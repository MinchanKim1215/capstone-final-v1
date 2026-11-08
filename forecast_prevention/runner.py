"""Paired full missions using the original flight and packet engines."""
from collections import Counter
from contextlib import nullcontext
import gc
from pathlib import Path
import time

import numpy as np
import torch

from . import ARTIFACTS, ROOT
from .protocol import BASELINE, OLD, read, sha, write
from .controller import Inbox, decide
from .diagnostics import analyze, connected, outage_runs
from adaptive_joint_expanded.contracts import DEPLOYMENTS, deployment_mask, entry_for, public_context, seed_all
from adaptive_joint_expanded.experiment import load_checkpoint
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.model import tensor_observation
from adaptive_joint_expanded.observation import Observer, policy_inputs
from adaptive_joint_expanded.rollout import trajectory_audit
from adaptive_joint_expanded.world import World
from forecast_causal.recorder import Recorder
from forecast_navigation.wire import attach_navigation
from forecast_service.replay import service_health

MODEL = None
CONFIG = None


def initialize():
    global MODEL, CONFIG
    seed_all(1)
    protocol = read(OLD / 'protocol.json')
    MODEL, _, _ = load_checkpoint(BASELINE, protocol, 'cpu')
    MODEL.eval()
    CONFIG = protocol['config']


def deployment_for(spec):
    entry = entry_for(spec['region'])
    ctx = public_context(entry, spec['direction'], CONFIG['available_drones'], CONFIG['minimum_search'])
    valid = deployment_mask(CONFIG['available_drones'], CONFIG['minimum_search'])
    with torch.no_grad():
        index = int(MODEL.select(torch.from_numpy(ctx), torch.from_numpy(valid)).argmax())
    return DEPLOYMENTS[index]


def outcome(links, n_search, packet):
    network = connected(links, n_search)
    health = service_health(packet['packet_born'], packet['packet_status'], packet['packet_delay'], len(links))
    service = health['service_healthy']
    cuts = outage_runs(network)
    service_cuts = outage_runs(service)
    return dict(persistent_onsets=sum(start > 0 for start, _ in cuts),
                initial_persistent_cut=bool(cuts and cuts[0][0] == 0),
                persistent_cut_seconds=sum((stop-start)*.5 for start, stop in cuts),
                persistent_cut_intervals=[[start*.5, stop*.5] for start, stop in cuts],
                service_onsets=sum(start > 0 for start, _ in service_cuts),
                service_bad_seconds=float((~service).sum()*.5)), network, service


def run(spec, arm, max_steps=None, diagnosis=False):
    """The existing actor proposes every action. The overlay can replace one."""
    start = time.perf_counter()
    seed_all(spec['seed'] + 9173)
    fleet = deployment_for(spec)
    # The 1 Hz V7 logging call is redundant for actuation. policy() recomputes
    # the same predictor from the raw observation history at each 3 s boundary.
    env = World(**spec, deployment=fleet.key, observer=Observer(), max_steps=max_steps,
                fleet_cost=CONFIG['fleet_cost_per_step'])
    telemetry = attach_navigation(env) if arm != 'baseline' else None
    inbox = Inbox(env.n, env.ns) if telemetry is not None else None
    recorder = Recorder(env) if diagnosis else None
    commands = []
    original_actions = []
    applied_actions = []
    intervention = []
    counters = Counter()
    times = []
    receipt_count = 0
    with recorder.installed() if recorder is not None else nullcontext():
        while not env.done:
            observation = policy_inputs(env)
            with torch.no_grad():
                logits, _ = MODEL.policy(tensor_observation(observation), 'joint')
            logits = logits.numpy()
            actions = logits.argmax(axis=-1)
            original_actions.append(actions.copy())
            now = env.k * 3.
            detail = None
            if inbox is not None:
                for k in range(receipt_count, len(telemetry.nav_tx)):
                    received = telemetry.plan_rx[k][env.ns].copy()
                    received[env.ns] = False
                    inbox.receive(now, telemetry.nav_tx[k][received])
                receipt_count = len(telemetry.nav_tx)
                state = inbox.state(now, env.last_x[env.ns], env.last_mask[env.ns])
                if now >= 21. and arm != 'navigation_control':
                    tick = time.perf_counter()
                    action, detail = decide(arm, env.city, env.scenario, state, env.ns,
                                            env.proposal['nominal'][env.ns], int(actions[0]),
                                            logits[0], observation['action_mask'][0])
                    times.append((time.perf_counter() - tick) * 1000)
                    counters[detail['reason']] += 1
                    if action != actions[0]:
                        detail.update(time_s=now, base_action=int(actions[0]), action=action)
                        actions[0] = action
                    else:
                        detail = None
            command = np.zeros((env.n, 2))
            command[env.ns:] = OFFSETS[actions]
            commands.append(command)
            env.step(command)
            protected = env.history[-1]['protected']
            env.previous_actions = np.zeros(fleet.nr, int) if protected else actions.copy()
            applied_actions.append(env.previous_actions.copy())
            if detail is not None:
                detail['motion_guard_blocked'] = protected
                detail['actual_endpoint'] = env.city.points[env.nodes[env.ns]].tolist()
                intervention.append(detail)
    links = np.asarray(env.channel_available)
    nodes = np.asarray(env.node_log)
    positions = env.city.points[nodes]
    generated = int(env.packets.ic[0])
    packet = dict(packet_born=env.packets.born[:generated], packet_status=env.packets.status[:generated],
                  packet_delay=env.packets.delays[:generated])
    metrics, network, service = outcome(links, env.ns, packet)
    for event in intervention:
        end = int(event['time_s']*2)
        event['actual_pre_action_connected'] = bool(network[end-1])
        event['actual_own_displacement_m'] = float(np.linalg.norm(
            positions[end//6+1, env.ns] - positions[end//6, env.ns]))
    data = dict(nodes=nodes, commands=np.asarray(commands), links=links,
                original_actions=np.asarray(original_actions), applied_actions=np.asarray(applied_actions),
                network=network, service=service)
    rewards = np.asarray([env.settled[k]['reward'] for k in range(env.k)], dtype=np.float32)
    terms = np.stack([env.settled[k]['reward_terms'] for k in range(env.k)])
    audit = trajectory_audit(env, dict(data, rewards=rewards, reward_terms=terms))
    if not audit['passed']:
        raise ValueError('Actual trajectory audit failed: ' + str(audit))
    row = dict(env.summary, **metrics, spec=spec, arm=arm, intervention=intervention,
               decision_counts=dict(counters), audited=audit, wall_s=time.perf_counter()-start,
               control_p95_ms=float(np.percentile(times, 95)) if times else None)
    if recorder is not None:
        row['cause_analysis'] = analyze(env, positions, links, recorder.arrays())
        from .patterns import compare
        row['observed_precursors'] = compare(env.telemetry.x[:env.t, env.ns], env.telemetry.mask[:env.t, env.ns], network)
    return row, data


def simulate(task):
    spec, arm, scope = task
    diagnosis = scope == 'historical'
    prefix = ARTIFACTS / scope / (spec['region'] + '_' + spec['kind'] + '_' + arm)
    path = prefix.with_suffix('.json')
    archive = prefix.with_suffix('.npz')
    contract = dict(spec=spec, arm=arm, scope=scope, policy=sha(BASELINE),
                    sources={k: sha(ROOT / k) for k in ('runner.py', 'controller.py', 'diagnostics.py')})
    if path.exists():
        old = read(path)
        if old['contract'] != contract or sha(archive) != old['archive_sha256']:
            raise ValueError('Completed mission changed')
    else:
        row, data = run(spec, arm, diagnosis=diagnosis)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        temp = prefix.with_suffix('.pending.npz')
        np.savez_compressed(temp, **data)
        temp.replace(archive)
        write(path, dict(row, contract=contract, archive_path=str(archive), archive_sha256=sha(archive)))
    gc.collect()
    return dict(path=str(path), sha256=sha(path), region=spec['region'], kind=spec['kind'], arm=arm)


def archived_baseline(item):
    source = item['source_record']
    if sha(source['array_path']) != source['array_sha256'] or sha(item['replay_path']) != item['replay_sha256']:
        raise ValueError('Paired baseline archive changed')
    with np.load(source['array_path'], allow_pickle=False) as raw:
        links = raw['links']
    with np.load(item['replay_path'], allow_pickle=False) as evidence:
        metrics, _, _ = outcome(links, source['n_search'], evidence)
    return dict(read(source['metadata_path']), **metrics, arm='baseline', intervention=[],
                archive_path=source['array_path'], archive_sha256=source['array_sha256'])
