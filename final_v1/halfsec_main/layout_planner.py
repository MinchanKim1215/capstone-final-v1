"""Select a mixed-altitude relay layout from the map and planned search alone."""

from copy import copy
from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from adaptive_joint_expanded.features import THRESHOLD, widest
from fast_halfsec import context
from fast_halfsec.navigation import GUARD
from final_v1_fast.motion import VMAX
from forecast_shield.forecaster import geometry
from swarm_sim.mission import MAX_MISSION

from .radio_layout import RadioLayout, install_fixed, mobile_component
from .release import map_scope
from .staged_schedule import STEP_S, nominal_trace


HEIGHT_STEPS_M = (5., 10., 15., 20., 25., 30.)
NOISE_DBM = -107.
HEADROOM_DB = (0., 6., 12.)
SCREEN_BLOCKS = 80
FINALISTS = 12
STATION_BATCH = 24
STATION_GAP_M = GUARD + VMAX * STEP_S / 2.


def closed_paths(graph):
    """Return bottleneck margins for all pairs of a symmetric radio graph."""
    n = graph.shape[-1]
    a, b = np.triu_indices(n, 1)
    paths = np.empty_like(graph)
    paths[..., a, b] = paths[..., b, a] = widest(graph, n)
    paths[..., np.arange(n), np.arange(n)] = np.inf
    return paths


def margins(paths, ns):
    a, b = np.triu_indices(ns, 1)
    return paths[..., a, b].min(axis=-1), paths[..., :ns, ns].min(axis=-1)


def via_station(paths, edges, ns):
    """Close paths that may use the new station once, using mobile closure."""
    to_fixed = np.minimum(paths[..., :ns + 1, :], edges[..., None, :]).max(axis=-1)
    a, b = np.triu_indices(ns, 1)
    pair = np.maximum(paths[..., a, b], np.minimum(to_fixed[..., a], to_fixed[..., b]))
    observer = np.maximum(paths[..., :ns, ns], np.minimum(to_fixed[..., :ns], to_fixed[..., ns, None]))
    return pair.min(axis=-1), observer.min(axis=-1)


def margin_from_loss(loss, scenario):
    return scenario.tx_power_dbm + 2. * scenario.gain_dbi - NOISE_DBM - THRESHOLD - loss


def mobile_graph(city, scenario, xy):
    loss, _ = geometry(city, scenario, xy)
    return margin_from_loss(np.maximum(loss, loss.swapaxes(-1, -2)), scenario)


def edge_margin(city, scenario, left, right):
    outgoing = city.radio.propagation(left, right, scenario)[0]
    incoming = city.radio.propagation(right, left, scenario)[0]
    shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    return margin_from_loss(np.maximum(outgoing, incoming).reshape(shape), scenario)


def station_xyz(city, scenario, nodes):
    return np.column_stack((city.points[nodes], np.full(len(nodes), scenario.altitude_m)))


def station_edges(city, scenario, track, stations):
    parts = [edge_margin(city, scenario, stations[i:i + STATION_BATCH, None, None], track[None])
             for i in range(0, len(stations), STATION_BATCH)]
    return np.concatenate(parts)


def elevation(city, scenario, track, graph, relay, height):
    raised = track[:, relay].copy()
    raised[:, 2] = height
    changes = edge_margin(city, scenario, raised[:, None], track)
    altered = graph.copy()
    altered[:, relay, :] = altered[:, :, relay] = changes
    return raised, closed_paths(altered)


def screening_frames(network, observer):
    """Retain the weakest sample and the midpoint of each mission block."""
    risk = np.minimum(network, observer)
    blocks = np.array_split(np.arange(len(risk)), min(SCREEN_BLOCKS, len(risk)))
    return np.unique([point for block in blocks for point in (block[np.argmin(risk[block])], block[len(block) // 2])])


def risk_score(network, observer):
    """Rank missed margin targets before remaining deficits and lower-tail margin."""
    counts = [np.count_nonzero(network < reserve, axis=-1) for reserve in HEADROOM_DB]
    control = np.count_nonzero(observer < HEADROOM_DB[1], axis=-1)
    worst = np.minimum(network, observer)
    mean_gap = np.maximum(0., HEADROOM_DB[-1] - worst).mean(axis=-1)
    return np.stack((*counts, control, mean_gap, -np.quantile(worst, .1, axis=-1)), axis=-1)


def track_with_altitude(xy, altitude):
    return np.concatenate((xy, np.full((*xy.shape[:-1], 1), altitude)), axis=-1)


def candidate_rank(score, layout, clearance):
    return (*score, layout.height_m, -clearance[layout.fixed_node], layout.elevated_relay, layout.fixed_node)


def screen(city, scenario, fleet, track, graph, nodes, clearance):
    stations = station_xyz(city, scenario, nodes)
    at_baseline_height = station_edges(city, scenario, track, stations)
    orders = []
    for relay in range(fleet.ns, fleet.n):
        for increase in HEIGHT_STEPS_M:
            height = scenario.altitude_m + increase
            raised, paths = elevation(city, scenario, track, graph, relay, height)
            edges = at_baseline_height.copy()
            edges[:, :, relay] = station_edges(city, scenario, raised[:, None], stations)[:, :, 0]
            scores = risk_score(*via_station(paths, edges, fleet.ns))
            for node, score in zip(nodes, scores):
                layout = RadioLayout(relay, height, int(node))
                orders.append((candidate_rank(score, layout, clearance), layout))
    return sorted(orders, key=lambda item: item[0])


def score_full(city, scenario, track, graph, ns, layout):
    raised, paths = elevation(city, scenario, track, graph, layout.elevated_relay, layout.height_m)
    station = station_xyz(city, scenario, [layout.fixed_node])[0]
    links = edge_margin(city, scenario, station, track)
    links[:, layout.elevated_relay] = edge_margin(city, scenario, station, raised)
    return risk_score(*via_station(paths, links, ns))


def nominal_stats(network, observer):
    return dict(samples=len(network), network_below_headroom_seconds={
        str(db): float(np.count_nonzero(network < db) * STEP_S) for db in HEADROOM_DB},
        observer_below_6db_seconds=float(np.count_nonzero(observer < HEADROOM_DB[1]) * STEP_S),
        network_margin_p10_db=float(np.quantile(network, .1)))


def plan_for(region, direction):
    """Choose a station and elevation blind to outages, shadow draws and interference locations."""
    started = perf_counter()
    with map_scope(region):
        city, scenario, fleet, survey, _ = context.survey_for(region, direction)
        xy, _, _ = nominal_trace(city, fleet, survey)
        if not len(xy) or len(xy) * STEP_S >= MAX_MISSION:
            return dict(layout=None, reason='nominal mission did not finish', map_sha256=city.sha256)
        clearance = cKDTree(xy.reshape(-1, 2)).query(city.points)[0]
        nodes = np.flatnonzero(clearance >= STATION_GAP_M)
        if not len(nodes):
            return dict(layout=None, reason='no site clear of the nominal flight tracks', map_sha256=city.sha256)
        graph = mobile_graph(city, scenario, xy)
        base = margins(closed_paths(graph), fleet.ns)
        frames = screening_frames(*base)
        track = track_with_altitude(xy, scenario.altitude_m)
        ranked = screen(city, scenario, fleet, track[frames], graph[frames], nodes, clearance)
        sites, finalists = set(), []
        for _, layout in ranked:
            if layout.fixed_node in sites:
                continue
            sites.add(layout.fixed_node)
            try:
                mobile_component(city, survey, fleet.ns, layout.fixed_node)
            except ValueError:
                continue
            full_score = score_full(city, scenario, track, graph, fleet.ns, layout)
            finalists.append((candidate_rank(full_score, layout, clearance), layout))
            if len(finalists) == FINALISTS:
                break
        details = dict(map_sha256=city.sha256, screened_sites=len(nodes), screened_frames=len(frames),
                       screened_layouts=len(ranked), finalists=len(finalists), nominal_baseline=nominal_stats(*base))
        for _, layout in sorted(finalists, key=lambda item: item[0]):
            changed_city = copy(city)
            try:
                added, fixed = install_fixed(changed_city, fleet, survey, layout.fixed_node)
            except ValueError:
                continue
            np.testing.assert_array_equal(fixed.waypoints[:, :fleet.ns], survey.waypoints[:, :fleet.ns])
            planned, _, _ = nominal_trace(changed_city, added, fixed)
            if len(planned) * STEP_S >= MAX_MISSION:
                continue
            heights = np.full(added.n, scenario.altitude_m)
            heights[layout.elevated_relay] = layout.height_m
            changed_city.radio_altitudes_m = heights
            planned_graph = mobile_graph(changed_city, scenario, planned)
            combined = margins(closed_paths(planned_graph), fleet.ns)
            return dict(**details, layout=asdict(layout), station_xy_m=city.points[layout.fixed_node].tolist(),
                        station_nominal_clearance_m=float(clearance[layout.fixed_node]),
                        nominal_combined=nominal_stats(*combined), planning_wall_s=perf_counter() - started)
        return dict(**details, layout=None, reason='no finalist preserved the planned mission',
                    planning_wall_s=perf_counter() - started)
