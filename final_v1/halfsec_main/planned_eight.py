"""A fixed-size eight-aircraft placement and its verified half-second flight."""

from copy import copy
from dataclasses import asdict, replace
from functools import lru_cache
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from adaptive_joint_expanded import baseline
from adaptive_joint_expanded.contracts import DEPLOYMENT_BY_KEY
from fast_halfsec import ARTIFACTS, LEGACY
from fast_halfsec.controller import Option
from fast_halfsec.features import N_CANDIDATE
from fast_halfsec.navigation import DT, make_survey
from final_v1_fast.runner import intervals
from forecast_rebuild.contracts import read, sha
from forecast_shield.forecaster import geometry
from swarm_sim.config import Scenario
from swarm_sim.mission import MAX_MISSION
from usable.collect import prepare_city_without_unused_radio_cache

from .layout_planner import (HEIGHT_STEPS_M, STATION_GAP_M, closed_paths, elevation, margin_from_loss,
                             margins, risk_score, screening_frames, station_edges, station_xyz,
                             track_with_altitude)
from .mission import MainFlight
from .navigation_pass import PassCity
from .planned_eight_motion import (BASE_HEIGHT_M, COMMAND_DELAY_S, SAMPLE_S, VERTICAL_ACCEL_MPS2,
                                    VERTICAL_SPEED_MPS, HeightNavigator, HeightWindow, duration_s)
from .planning_motion import HeightNavigator as PlanningHeightNavigator
from .radio_layout import install_fixed
from .release import EXPANDED_REGISTRY, load_model
from .staged_passage import TrafficEvidence, route_confirmed
from .staged_schedule import nominal_trace


FLEET = DEPLOYMENT_BY_KEY['v8_s3_r5_40']
RISK_MARGIN_DB = 6.
DESCENT_MARGIN_DB = 2.
ALTITUDE_LEAD_S = duration_s(max(HEIGHT_STEPS_M)) + COMMAND_DELAY_S + 8.
MERGE_GAP_S = 60.
SITE_SPACING_M = 50.
STATION_FINALISTS = 2
LOWER_TIMES = np.array([0., 1., 2., 4., 6.])


@lru_cache(maxsize=1)
def registry():
    return {row['key']: row for row in read(EXPANDED_REGISTRY)['maps']}


@lru_cache(maxsize=1)
def scene_for(region, direction):
    """Read the frozen scene directly for evaluation while preserving its study role."""
    entry = registry()[region]
    path = LEGACY / entry['path']
    if sha(path) != entry['sha256']:
        raise ValueError(f'Registered scene changed: {region}')
    scenario = Scenario(direction=direction, motion_dt_s=3.).validate()
    if scenario.altitude_m != BASE_HEIGHT_M:
        raise ValueError('The altitude program expects the 50 m survey grid')
    city = prepare_city_without_unused_radio_cache(path, scenario, ARTIFACTS / 'cache/city')
    city.radio.enable_cache()
    with patch.object(baseline, 'ARTIFACTS', ARTIFACTS):
        path = baseline.ensure_nominal(city, FLEET, scenario)
    with np.load(path, allow_pickle=False) as archive:
        route = archive['nodes']
        survey = make_survey(city, FLEET.ns, route)
    return city, scenario, FLEET, survey, (len(route)-1)*3.


def station_survey(city, fleet, survey, station):
    if station is None:
        return survey
    relay, node = station['relay'], station['node']
    if not fleet.ns < relay < fleet.n:
        raise ValueError('The fixed position must use a non-coordinator relay')
    _, extended = install_fixed(city, fleet, survey, node)
    order = np.arange(fleet.n)
    order[relay] = fleet.n
    return replace(survey, initial=extended.initial[order], waypoints=extended.waypoints[:, order],
                   options=extended.options[:, order[fleet.ns:]-fleet.ns])


def graph_for(city, scenario, xy, heights=None):
    radio_city = city
    if heights is not None:
        radio_city = copy(city)
        radio_city.radio_altitudes_m = heights
    loss, shielding = geometry(radio_city, scenario, xy)
    return (margin_from_loss(np.maximum(loss, loss.swapaxes(-1, -2)), scenario), shielding)


def quality(graph):
    return margins(closed_paths(graph), FLEET.ns)


def score(quality_pair):
    return tuple(risk_score(*quality_pair).tolist())


def describe(quality_pair, steps):
    network, observer = quality_pair
    return dict(seconds=steps*SAMPLE_S, survey_below_0_seconds=float((network < 0.).sum()*SAMPLE_S),
                survey_below_6_seconds=float((network < 6.).sum()*SAMPLE_S),
                observer_below_6_seconds=float((observer < 6.).sum()*SAMPLE_S),
                survey_p10_db=float(np.quantile(network, .1)), complete_route=True)


def representative_nodes(city, xy):
    gap = cKDTree(xy.reshape(-1, 2)).query(city.points)[0]
    chosen = {}
    for node in np.flatnonzero(gap >= STATION_GAP_M):
        point = city.points[node]
        cell = tuple(np.floor(point/SITE_SPACING_M).astype(int))
        center = (np.asarray(cell)+.5)*SITE_SPACING_M
        rank = (float(np.linalg.norm(point-center)), -gap[node], int(node))
        if cell not in chosen or rank < chosen[cell]:
            chosen[cell] = rank
    return np.asarray(sorted(rank[2] for rank in chosen.values()), dtype=int)


def rank_stations(city, scenario, xy, graph, before):
    nodes = representative_nodes(city, xy)
    if not len(nodes):
        return [], 0
    frames = screening_frames(*before)
    track = track_with_altitude(xy[frames], BASE_HEIGHT_M)
    station_links = station_edges(city, scenario, track, station_xyz(city, scenario, nodes))
    reference = graph[frames]
    ranked = []
    for relay in range(FLEET.ns+1, FLEET.n):
        altered = np.broadcast_to(reference, (len(nodes),*reference.shape)).copy()
        altered[..., relay, :] = altered[..., :, relay] = station_links
        values = risk_score(*quality(altered))
        ranked.extend((tuple(row.tolist()), relay, int(node)) for row, node in zip(values, nodes))
    ranked.sort()
    return ranked, len(nodes)


def hazard_windows(before, stages):
    bad = np.minimum(*before) < RISK_MARGIN_DB
    groups = []
    for start, stop in intervals(~bad, SAMPLE_S):
        begin = np.searchsorted(stages, stages[max(0, round((start-ALTITUDE_LEAD_S)/SAMPLE_S))])
        entry = round(start/SAMPLE_S)
        end = round(stop/SAMPLE_S)
        if groups and (start - groups[-1][2]*SAMPLE_S <= MERGE_GAP_S or stages[begin] <= stages[groups[-1][2]-1]):
            groups[-1] = (groups[-1][0], groups[-1][1], end)
        else:
            groups.append((begin, entry, end))
    return groups


def assign_heights(city, scenario, xy, graph, stages, groups, station):
    if not groups:
        return []
    excluded = FLEET.ns if station is None else station['relay']
    frames = np.unique(np.concatenate([np.arange(entry, end) for _, entry, end in groups]))
    track = track_with_altitude(xy[frames], BASE_HEIGHT_M)
    ref_graph = graph[frames]
    baseline_quality = quality(ref_graph)
    reference = np.minimum(*baseline_quality)
    slices = [slice(*(np.searchsorted(frames, [entry, end]).tolist())) for _, entry, end in groups]
    masks = [reference[s] < RISK_MARGIN_DB for s in slices]
    rank = None
    best = []
    for relay in range(FLEET.ns+1, FLEET.n):
        if relay == excluded:
            continue
        alternatives = {}
        for increase in HEIGHT_STEPS_M:
            height = BASE_HEIGHT_M + increase
            _, paths = elevation(city, scenario, track, ref_graph, relay, height)
            alternatives[height] = np.minimum(*margins(paths, FLEET.ns))
        selected = []
        improved, altitude_seconds = 0, 0.
        for (begin, entry, end), section, mask in zip(groups, slices, masks):
            ref = reference[section]
            choices = []
            for height, projected in alternatives.items():
                current = projected[section]
                if np.any((ref >= 0.) & (current < 0.)):
                    continue
                gain = int(np.count_nonzero(mask & (current >= RISK_MARGIN_DB)))
                if gain >= 2:
                    choices.append((-gain, height))
            if not choices:
                continue
            gain, height = min(choices)
            selected.append(HeightWindow(int(stages[begin]), int(stages[entry]), int(stages[end-1])+1, relay, height))
            improved -= gain
            altitude_seconds += (height-BASE_HEIGHT_M)*(end-begin)*SAMPLE_S
        candidate = (-improved, altitude_seconds, relay)
        if selected and (rank is None or candidate < rank):
            rank, best = candidate, selected
    return best


def can_descend(nav, scenario, relay, target, estimate=None):
    mask = nav.held()
    mask[relay] = True
    future = nav.predict_holds(nav.last_offsets, nav.last_offsets, 0, LOWER_TIMES, mask)
    proposed = nav.heights.copy()
    proposed[relay] = target
    shifted, _ = graph_for(nav.city, scenario, future, proposed)
    if estimate is not None:
        if not estimate.local_rf:
            return False
        old_loss, _ = geometry(nav.city, scenario, future[:1])
        received = estimate.power-(10.-old_loss[0])
        loss_residual = np.nan_to_num(np.clip(received, -6., 0.))
        adjustment = loss_residual - (estimate.noise+107.)[None, :] - np.where(estimate.measured, 0., 3.)[None, :]
        shifted += np.minimum(adjustment, adjustment.T)[None]
    network, observer = quality(shifted)
    return bool((network >= DESCENT_MARGIN_DB).all() and (observer >= DESCENT_MARGIN_DB).all())


def preplanned_trace(city, scenario, survey, windows):
    # 배치안 확인용 모의 비행도 제출본 원래 이동(지점마다 정지)으로 한다 (planning_motion).
    nav = PlanningHeightNavigator(city, FLEET.ns, survey, windows,
                          lambda now, since, _: now-since >= SAMPLE_S-1e-8,
                          lambda now, relay, goal: can_descend(nav, scenario, relay, goal))
    positions, heights = [], []
    offsets = np.zeros(FLEET.nr, np.int32)
    for tick in range(round(MAX_MISSION/DT)):
        nav.update(tick*DT)
        if nav.finished_at is not None:
            break
        if tick % round(SAMPLE_S/DT) == 0:
            positions.append(nav.motion.positions()[0])
            heights.append(nav.heights.copy())
        nav.step(offsets)
    return np.asarray(positions), np.asarray(heights), nav


def plan_for(region, direction):
    start = perf_counter()
    original, scenario, fleet, survey, _ = scene_for(region, direction)
    xy, stages, _ = nominal_trace(original, fleet, survey)
    details = dict(region=region, direction=direction, map_sha256=original.sha256, station=None, windows=[])
    if not len(xy) or len(xy)*SAMPLE_S >= MAX_MISSION:
        return dict(**details, reason='nominal route did not finish', planning_wall_s=perf_counter()-start)
    base_graph, _ = graph_for(original, scenario, xy)
    before = quality(base_graph)
    details['before'] = describe(before, len(xy))
    if not (np.minimum(*before) < RISK_MARGIN_DB).any():
        return dict(**details, reason='nominal route already meets the margin targets', planning_wall_s=perf_counter()-start)
    ranked, site_count = rank_stations(original, scenario, xy, base_graph, before)
    details['screened_sites'] = site_count
    variants = [(None, original, survey, xy, stages, base_graph, before)]
    used_sites = set()
    for _, relay, node in ranked:
        if node in used_sites:
            continue
        used_sites.add(node)
        changed_city = copy(original)
        station = dict(relay=relay, node=node, xy_m=original.points[node].tolist())
        try:
            changed = station_survey(changed_city, fleet, survey, station)
        except ValueError:
            continue
        np.testing.assert_array_equal(changed.waypoints[:,:fleet.ns], survey.waypoints[:,:fleet.ns])
        alternative, alt_stages, _ = nominal_trace(changed_city, fleet, changed)
        if not len(alternative) or len(alternative)*SAMPLE_S >= MAX_MISSION:
            continue
        alt_graph, _ = graph_for(changed_city, scenario, alternative)
        variants.append((station, changed_city, changed, alternative, alt_stages, alt_graph, quality(alt_graph)))
        if len(variants)-1 == STATION_FINALISTS:
            break

    best_score, choice = score(before), dict(reason='no placement improved the full-route check')
    for station, city, survey, xy, stages, graph, reference in variants:
        groups = hazard_windows(reference, stages)
        windows = assign_heights(city, scenario, xy, graph, stages, groups, station)
        planned, final, checks = xy, reference, []
        if windows:
            planned, heights, nav = preplanned_trace(copy(city), scenario, survey, windows)
            if nav.finished_at is None or nav.covered[city.observable].sum() != survey.observable:
                continue
            final = quality(graph_for(city, scenario, planned, heights)[0])
            checks = nav.windows_log
            if not all(row['outcome'] == 'crossed' for row in checks) or len(checks) != len(windows):
                continue
        rank = score(final)
        if rank[:4] < best_score[:4] and rank[0] <= best_score[0]:
            best_score = rank
            choice = dict(station=station, windows=[asdict(w) for w in windows],
                          after=describe(final, len(planned)), preflight_windows=checks)
    return {**details, **choice, 'planning_wall_s': perf_counter()-start}


class PlannedFlight(MainFlight):
    def __init__(self, model, threshold, plan=None, explore=False):
        super().__init__(model, threshold)
        self.plan = plan or dict(station=None, windows=[])
        self.explore = explore
        self.survey_factory = scene_for
        if self.plan['windows']:
            self.packet_factory = TrafficEvidence

    def survey(self, *args):
        city, self.scenario, fleet, survey, anchor = super().survey(*args)
        # 통과 비행 안전 검사용 감싸개(PassCity)를 벗겨 고정 중계 위치를 넣은 뒤 다시 감싼다.
        wrapped = isinstance(city, PassCity)
        city = copy(city._city if wrapped else city)
        if self.plan.get('map_sha256', city.sha256) != city.sha256:
            raise ValueError('The placement belongs to another map')
        changed = station_survey(city, fleet, survey, self.plan['station'])
        self.history.bind(city, fleet)
        return (PassCity(city) if wrapped else city), self.scenario, fleet, changed, anchor

    def navigator(self, city, ns, survey):
        windows = [HeightWindow(**data) for data in self.plan['windows']]
        if self.plan['station'] and any(w.relay == self.plan['station']['relay'] for w in windows):
            raise ValueError('Station and climbing roles must use different aircraft')
        self.nav = HeightNavigator(city, ns, survey, windows, self.confirm, self.descent_safe)
        return self.nav

    def confirm(self, now, since, relay):
        sensor, packet = self.sensor, self.packets
        return all(route_confirmed(now, since, anchor, self.nav.ns, packet.hops_seen,
                                   sensor.remote_stamp, packet.last_received) for anchor in (self.nav.ns, relay))

    def descent_safe(self, now, relay, target):
        return can_descend(self.nav, self.scenario, relay, target, self.sensor.estimate(now))

    def options_for(self, nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery):
        if nav.entry_waiting:
            hold = np.zeros(N_CANDIDATE, np.float32)
            hold[-1] = 1.
            return [Option(requested.copy(), hold, 0., None)], 0
        return super().options_for(nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery)


def run(spec, plan=None, model=None, threshold=None, explore=False, max_seconds=MAX_MISSION):
    if explore and (registry()[spec['region']]['study_role'], registry()[spec['region']]['joint_split']) != ('train', 'train'):
        raise ValueError('Learning traces may only use the original training geography')
    if plan and (plan['region'], plan['direction']) != (spec['region'], spec['direction']):
        raise ValueError('The placement belongs to another survey direction')
    if model is None:
        model, threshold = load_model()
    flight = PlannedFlight(model, threshold, plan, explore)
    row, arrays = flight.run(spec, max_seconds)
    nav = flight.nav
    arrays['altitudes'] = np.asarray(nav.samples[:len(arrays['network'])], np.float32)
    if arrays['altitudes'].shape != (len(arrays['network']), 8) or row['total_drones'] != 8:
        raise ValueError('Eight-aircraft flight violated the altitude trace contract')
    station = flight.plan['station']
    if station is not None and not np.allclose(arrays['positions'][:, station['relay']], station['xy_m'], atol=1e-5, rtol=0.):
        raise ValueError('The preassigned stationary relay left its site')
    if nav.max_speed > VERTICAL_SPEED_MPS+1e-6 or nav.max_acceleration > VERTICAL_ACCEL_MPS2+1e-6:
        raise ValueError('Vertical motion violated the simulated physical limits')
    row.update(controller='halfsec_planned_eight', station=station, altitude_windows=nav.windows_log,
               scheduled_altitude_windows=len(nav.windows),
               altitude_moves=nav.moves, vertical_travel_m=nav.vertical_travel, end_altitudes_m=nav.heights.tolist(),
               max_vertical_speed_mps=nav.max_speed, max_vertical_accel_mps2=nav.max_acceleration,
               climbed=sum(x['kind']=='climb' and x['end_s']<=row['duration_s'] for x in nav.moves),
               descended=sum(x['kind']=='descent' and x['end_s']<=row['duration_s'] for x in nav.moves))
    return row, arrays
