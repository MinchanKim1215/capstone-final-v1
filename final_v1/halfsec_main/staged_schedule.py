"""Plan relay stations from the map and the nominal survey, before any radio draw."""

import argparse
from dataclasses import asdict

import numpy as np
from scipy.sparse.csgraph import dijkstra

from adaptive_joint_expanded.features import THRESHOLD, widest
from fast_halfsec import context
from final_v1_fast.motion import AMAX, DT, VMAX
from final_v1_fast.runner import intervals
from forecast_rebuild.contracts import now, sha, write
from forecast_shield.forecaster import geometry
from swarm_sim.mission import MAX_MISSION

from .navigation import ClearanceNavigator
from .release import map_scope
from .reserve_trial import references
from .staged_passage import Passage, reserve_survey
from .staged_trial import TRIAL


STEP_S = .5
LEAD_S = 30.
NOISE_DBM = -97.
MAX_PASSAGES = 3


def nominal_trace(city, fleet, survey):
    nav = ClearanceNavigator(city, fleet.ns, survey)
    offsets = np.zeros(fleet.n - fleet.ns, np.int32)
    positions, stages, committed = [], [], []
    for tick in range(round(MAX_MISSION / DT)):
        nav.update(tick * DT)
        if nav.finished_at is not None:
            break
        if tick % round(STEP_S / DT) == 0:
            positions.append(nav.motion.positions()[0])
            stages.append(nav.stage)
            committed.append(nav.motion.end.copy())
        nav.step(offsets)
    return np.asarray(positions), np.asarray(stages), np.asarray(committed)


def arrival_graph(graph):
    travel = graph.copy()
    meters = travel.data
    seconds = np.where(meters <= VMAX**2 / AMAX, 2 * np.sqrt(meters / AMAX), meters / VMAX + VMAX / AMAX)
    travel.data = np.ceil(seconds / DT - 1e-8) * DT
    return travel


def network_margin(graph, ns):
    return widest(graph, ns).min(axis=-1)


def candidates(city, scenario, ns, xyz, current, safe_nodes, relay):
    base = current[None, :, :, :]
    for ids in np.array_split(safe_nodes, max(1, (len(safe_nodes) + 63) // 64)):
        if not len(ids):
            continue
        station = np.column_stack((city.points[ids], np.full(len(ids), scenario.altitude_m)))
        mount = station[:, None, None]
        tracks = xyz[None]
        shape = (len(ids), len(xyz), xyz.shape[1])
        tx = city.radio.propagation(mount, tracks, scenario)[0]
        rx = city.radio.propagation(tracks, mount, scenario)[0]
        edges = (10. - np.maximum(tx, rx).reshape(shape)) - NOISE_DBM - THRESHOLD
        graph = np.broadcast_to(base, (len(ids), *current.shape)).copy()
        graph[:, :, relay, :] = graph[:, :, :, relay] = edges
        yield ids, network_margin(graph, ns)


def choose_station(city, scenario, fleet, survey, xyz, stages, committed, graph, base, flight_graph, first, last):
    ns, n = fleet.ns, fleet.n
    before = max(0, first - round(LEAD_S / STEP_S))
    end_frame = min(len(stages) - 1, last + round(4. / STEP_S))
    begin, entry, end = int(stages[before]), int(stages[first]), int(stages[end_frame]) + 1
    if begin >= entry or end - entry > 18:
        return None
    frame_start = int(np.searchsorted(stages, entry))
    lead = (frame_start - before) * STEP_S - 5.
    if lead < 0.:
        return None
    section = slice(frame_start, end_frame + 1)
    current = base[section]
    risk = current < 0.
    pair = int(np.argmin(widest(graph[first], ns)))
    a, b = np.triu_indices(ns, 1)
    receiving = graph[first][[a[pair], b[pair]]]
    peer = int([a[pair], b[pair]][np.argmin(np.sum(receiving >= 0., axis=1))])
    reserved = np.unique(survey.waypoints[begin:end, :ns])
    points = city.points
    plans = []
    for relay in range(ns, n):
        travel = dijkstra(flight_graph, indices=int(committed[before, relay]), directed=False, limit=lead)
        sites = np.flatnonzero(np.isfinite(travel))
        sites = sites[~np.isin(sites, reserved)]
        if not len(sites):
            continue
        other = xyz[section, np.arange(n) != relay, :2]
        gaps = np.linalg.norm(points[sites, None, None] - other[None], axis=-1)
        sites = sites[gaps.min(axis=(1, 2)) >= 7.5]
        for ids, margins in candidates(city, scenario, ns, xyz[section], graph[section], sites, relay):
            for node, margin in zip(ids, margins):
                if np.any((margin < 0.) & ~risk):
                    continue
                gained = int((risk & (margin >= 0.)).sum())
                if gained < 2:
                    continue
                plan = Passage(begin, entry, end, peer, relay, int(node))
                rank = (gained, float(np.percentile(margin[risk], 20)), -float(travel[node]))
                plans.append((rank, plan))
    for _, plan in sorted(plans, key=lambda item: item[0], reverse=True):
        try:
            reserve_survey(survey, ns, plan)
        except ValueError:
            continue
        return plan
    return None


def plan_for(spec):
    with map_scope(spec['region']):
        city, scenario, fleet, survey, _ = context.survey_for(spec['region'], spec['direction'])
        xy, stages, committed = nominal_trace(city, fleet, survey)
        loss, shielding = geometry(city, scenario, xy)
        xyz = np.concatenate((xy, np.full((*xy.shape[:2], 1), scenario.altitude_m)), axis=-1)
        graph = 10. - np.maximum(loss, loss.swapaxes(-1, -2)) - NOISE_DBM - THRESHOLD
        open_loss = loss - shielding
        without_buildings = 10. - np.maximum(open_loss, open_loss.swapaxes(-1, -2)) - NOISE_DBM - THRESHOLD
        base = network_margin(graph, fleet.ns)
        hazard = (base < 0.) & (network_margin(without_buildings, fleet.ns) >= 0.)
        openings = [(int(a / STEP_S), int(b / STEP_S)) for a,b in intervals(~hazard, STEP_S)]
        openings.sort(key=lambda pair: -(pair[1] - pair[0]))
        flight_graph = arrival_graph(city.graph)
        chosen = []
        for a, b in openings:
            if len(chosen) >= MAX_PASSAGES:
                break
            if any(item.begin_stage <= int(stages[a]) < item.end_stage for item in chosen):
                continue
            plan = choose_station(city, scenario, fleet, survey, xyz, stages, committed, graph, base, flight_graph, a, b)
            if plan is not None and not any(
                    item.begin_stage < plan.end_stage and plan.begin_stage < item.end_stage for item in chosen):
                chosen.append(plan)
        selected = sorted(chosen, key=lambda item: item.begin_stage)
        return [asdict(p) for p in selected], len(openings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-focus', action='store_true', help='Include the two development maps')
    args = parser.parse_args()
    reference = references(True)
    if args.include_focus:
        reference.update(references(False))
    cases, hazard_counts = {}, {}
    for key, row in reference.items():
        cases[key], hazard_counts[key] = plan_for(row['spec'])
        print(row['spec']['region'], 'geometric hazards', hazard_counts[key], 'plans', cases[key], flush=True)
    write(TRIAL / 'map_only_plan.json', dict(built_at=now(), note='Plans use only each map, direction and the nominal trajectory.',
          code_sha256=sha(__file__), assumed_receiver_noise_dbm=NOISE_DBM, nominal_lead_s=LEAD_S,
          max_passages_per_map=MAX_PASSAGES, hazard_counts=hazard_counts, cases=cases))


if __name__ == '__main__':
    main()
