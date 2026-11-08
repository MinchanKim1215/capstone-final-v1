"""Event-by-event attribution. Hidden radio state is confined to this module."""
from itertools import combinations
from pathlib import Path

import numpy as np

from . import ARTIFACTS
from .protocol import read, sha, write
from adaptive_joint_expanded.world import World
from cooperative.packets import hop_probabilities
from forecast_causal.causes import annotate
from forecast_service.replay import service_health

FACTORS = ('building', 'interference', 'shadow')
SUBSETS = [()] + [part for size in (1, 2, 3) for part in combinations(FACTORS, size)]


def reachability(adjacency):
    reach = np.array(adjacency, dtype=bool, copy=True)
    nodes = np.arange(reach.shape[-1])
    reach[..., nodes, nodes] = True
    for k in nodes:
        reach |= reach[..., :, k, None] & reach[..., None, k, :]
    return reach


def connected(links, n_search):
    return reachability(links)[..., :n_search, :n_search].all(axis=(-1, -2))


def outage_runs(healthy, minimum_frames=2):
    edges = np.diff(np.r_[False, ~np.asarray(healthy, bool), False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)
            if stop - start >= minimum_frames]


def factorial_radio(env, positions, links, frames):
    """Replay the exact same positions and exogenous state for 2^3 radio worlds."""
    if not len(frames):
        return {}, {}
    step, subframe = np.divmod(frames, 6)
    fraction = subframe[:, None, None] / 6
    xy = positions[step] * (1 - fraction) + positions[step + 1] * fraction
    xyz = np.concatenate((xy, np.full((*xy.shape[:-1], 1), 50.)), axis=-1)
    n = env.n
    losses, _, extra = env.city.radio.propagation(
        np.repeat(xyz, n, axis=1).reshape(-1, 3), np.tile(xyz, (1, n, 1)).reshape(-1, 3),
        env.scenario)
    losses = losses.reshape(-1, n, n)
    extra = extra.reshape(-1, n, n)
    jammer_loss = env.city.radio.propagation(env.jammer[None], xyz.reshape(-1, 3), env.scenario)[0].reshape(-1, n)
    times = frames * .5
    active = (times >= env.jam_start) & (times < env.jam_end)
    interference = 10 ** ((env.jam_power - jammer_loss) / 10) * active[:, None]
    scratch = np.empty((n, n))
    scratch_sinr = np.empty((n, n))
    zero_shadow = np.zeros((n, n))
    zero_interference = np.zeros(n)
    connectivity = {}
    cuts = {}
    for subset in SUBSETS:
        graph = []
        for i, frame in enumerate(frames):
            loss = losses[i] - extra[i] if 'building' in subset else losses[i]
            noise = zero_interference if 'interference' in subset else interference[i]
            shadow = zero_shadow if 'shadow' in subset else env.shadow[frame]
            p = hop_probabilities(loss, noise, shadow, scratch, scratch_sinr)
            graph.append((p >= .9) & (p.T >= .9))
        graph = np.asarray(graph)
        if not subset:
            np.testing.assert_array_equal(graph, links[frames])
        reach = reachability(graph)
        connectivity[subset] = reach[:, :env.ns, :env.ns].all(axis=(1, 2))
        if not subset:
            for i, frame in enumerate(frames):
                groups = set()
                for search in range(env.ns):
                    groups.add(tuple(np.flatnonzero(reach[i, search]).tolist()))
                cuts[int(frame)] = sorted(groups)
    return {int(frame): [subset for subset in SUBSETS[1:] if connectivity[subset][i]]
            for i, frame in enumerate(frames)}, cuts


def minimal_repairs(repairs):
    return [list(part) for part in repairs if not any(set(other) < set(part) for other in repairs)]


def analyze(env, positions, links, packet):
    healthy = connected(links, env.ns)
    np.testing.assert_array_equal(healthy, packet['physical_truth'][:, 0] > .5)
    settled = service_health(packet['packet_born'], packet['packet_status'], packet['packet_delay'], len(links))
    service = settled['service_healthy']
    network_runs = outage_runs(healthy)
    service_runs = outage_runs(service)
    physical_windows = {start: list(range(start, min(stop, start + 2))) for start, stop in network_runs}
    frames = np.unique(np.concatenate(list(physical_windows.values()))).astype(int) if physical_windows else np.empty(0, int)
    repairs, cuts = factorial_radio(env, positions, links, frames)
    network_events = []
    for start, stop in network_runs:
        first = physical_windows[start]
        stable = [subset for subset in SUBSETS[1:] if all(subset in repairs[frame] for frame in first)]
        network_events.append(dict(onset_s=start * .5, duration_s=(stop-start)*.5,
                                   left_censored=start == 0, right_censored=stop == len(links),
                                   search_components_with_relays=cuts[start],
                                   first_second_minimal_repairs=minimal_repairs(stable),
                                   onset_frame_minimal_repairs=minimal_repairs(repairs[start])))
    service_events = []
    for start, stop in service_runs:
        row = dict(onset_s=start*.5, duration_s=(stop-start)*.5,
                   network_down_at_onset=not bool(healthy[start]))
        # The old packet recorder measures the live mission; the drain has no
        # complete per-attempt evidence. Keep boundary events in the count.
        if 24 <= row['onset_s'] <= len(links)*.5-8.5:
            detail = annotate(packet, row['onset_s'])
            row.update(flags=detail['flags'], late_packets=detail['bad_packets'],
                       settled_packets=detail['total_settled_packets'],
                       mean_queue_before_deadline=detail['mean_queue_before_deadline'])
        else:
            row['flags'] = None
        service_events.append(row)
    link_runs = sum(len(outage_runs(links[:, i, j])) for i, j in combinations(range(env.n), 2))
    return dict(network_events=network_events, service_events=service_events,
                individual_undirected_link_runs=link_runs, verified_factorial_frames=len(frames),
                duration_s=len(links)*.5, disconnected_s=float((~healthy).sum()*.5))


def archived(item):
    """Analyze an existing full mission and re-simulate each cut's first second."""
    source = item['source_record']
    path = ARTIFACTS / 'causes' / (item['key'] + '.json')
    contract = dict(raw_sha256=source['array_sha256'], packet_replay_sha256=item['replay_sha256'],
                    implementation_sha256=sha(__file__))
    if path.exists():
        previous = read(path)
        if previous['contract'] != contract:
            raise ValueError('Archived diagnosis source changed')
        return dict(path=str(path), sha256=sha(path), region=item['region'], kind=source['spec']['kind'])
    if sha(source['array_path']) != contract['raw_sha256'] or sha(item['replay_path']) != contract['packet_replay_sha256']:
        raise ValueError('Original flight or packet evidence changed')
    env = World(**source['spec'], deployment=source['deployment'], observer=None)
    with np.load(source['array_path'], allow_pickle=False) as raw:
        positions = raw['positions']
        links = raw['links']
    with np.load(item['replay_path'], allow_pickle=False) as packet:
        fields = ('physical_truth', 'packet_born', 'packet_status', 'packet_delay',
                  'packet_end', 'packet_evidence_before_deadline', 'packet_queue_before_deadline',
                  'packet_service_before_deadline', 'jam_start')
        evidence = {name: packet[name] for name in fields}
    result = analyze(env, positions, links, evidence)
    write(path, dict(contract=contract, spec=source['spec'], deployment=source['deployment'],
                     provenance='archived_full_flight_exact_radio_onset_replay', **result))
    return dict(path=str(path), sha256=sha(path), region=item['region'], kind=source['spec']['kind'])
