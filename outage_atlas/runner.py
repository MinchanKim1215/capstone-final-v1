"""Run new native missions, with map access confined to this diagnostic process."""
from pathlib import Path
from time import perf_counter

from . import ARTIFACTS
from .protocol import BASELINE, LEGACY, V7, maps, read, sha

from adaptive_joint_expanded import baseline, map_features, world
from adaptive_joint_expanded.contracts import DEPLOYMENTS, deployment_mask, public_context, seed_all, digest
from adaptive_joint_expanded.experiment import load_checkpoint
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.model import tensor_observation
from adaptive_joint_expanded.observation import Observer, policy_inputs
from adaptive_joint_expanded.rollout import trajectory_audit
from forecast_causal.recorder import Recorder
from forecast_prevention.diagnostics import outage_runs

from .events import Observations, network_events, service_events
from .radio import GeometryTrace, interference_regimes, packet_summary, packets_for, radio_graph, scenarios

import numpy as np
import torch


MODEL = CONFIG = REGISTRY = None
OLD_CACHE = V7/'artifacts'/'cache'
ORIGINAL_CITY = world.prepare_city_without_unused_radio_cache
ORIGINAL_NOMINAL = baseline.ensure_nominal


def entry_for_diagnostics(region):
    entry = REGISTRY[region]
    if sha(LEGACY / entry['path']) != entry['sha256']:
        raise ValueError('Map changed during the diagnostic: ' + region)
    return entry


def prepare_city(path, scenario, cache_dir):
    entry = next(m for m in REGISTRY.values() if LEGACY/m['path'] == path)
    old_path = OLD_CACHE/'city'/f"{entry['key']}_{entry['sha256'][:12]}_visibility.npz"
    if old_path.is_file():
        return ORIGINAL_CITY(path, scenario, old_path.parent)
    return ORIGINAL_CITY(path, scenario, cache_dir)


def nominal(city, deployment, scenario):
    spec = dict(map=city.sha256, deployment=deployment.key, direction=scenario.direction,
                engine=sha(baseline.__file__), motion='cooperative.motion-v5-unchanged')
    old = OLD_CACHE/'nominal'/(digest(spec)[:24]+'.npz')
    if old.is_file() and old.with_suffix('.json').is_file():
        meta = read(old.with_suffix('.json'))
        if meta['spec'] != spec or meta['sha256'] != sha(old):
            raise ValueError('Archived nominal trajectory changed')
        return old
    return ORIGINAL_NOMINAL(city, deployment, scenario)


def initialize():
    global MODEL, CONFIG, REGISTRY
    seed_all(1)
    REGISTRY = maps()
    world.entry_for = map_features.entry_for = entry_for_diagnostics
    world.ARTIFACTS = baseline.ARTIFACTS = map_features.ARTIFACTS = ARTIFACTS
    world.prepare_city_without_unused_radio_cache = prepare_city
    world.ensure_nominal = nominal
    protocol = read(V7/'artifacts'/'protocol.json')
    MODEL, _, _ = load_checkpoint(BASELINE, protocol, 'cpu')
    MODEL.eval()
    CONFIG = protocol['config']


def run(spec, max_steps=None):
    started = perf_counter()
    seed_all(spec['seed']+9173)
    ctx = public_context(entry_for_diagnostics(spec['region']), spec['direction'],
                         CONFIG['available_drones'], CONFIG['minimum_search'])
    valid = deployment_mask(CONFIG['available_drones'], CONFIG['minimum_search'])
    with torch.no_grad():
        selected = int(MODEL.select(torch.from_numpy(ctx), torch.from_numpy(valid)).argmax())
    fleet = DEPLOYMENTS[selected]
    env = world.World(**spec, deployment=fleet.key, observer=Observer(), max_steps=max_steps,
                      fleet_cost=CONFIG['fleet_cost_per_step'])
    geometry = GeometryTrace(env)
    recorder = Recorder(env)
    commands = []
    with recorder.installed():
        while not env.done:
            observation = policy_inputs(env)
            with torch.no_grad():
                logits, _ = MODEL.policy(tensor_observation(observation), 'joint')
            actions = logits.numpy().argmax(axis=-1)
            command = np.zeros((env.n, 2))
            command[env.ns:] = OFFSETS[actions]
            commands.append(command)
            geometry.calls = 0
            with geometry.record():
                env.step(command)
            env.previous_actions = np.zeros(fleet.nr, int) if env.history[-1]['protected'] else actions.copy()
    nodes = np.asarray(env.node_log)
    links = np.asarray(env.channel_available)
    commands = np.asarray(commands)
    terms = np.stack([env.settled[k]['reward_terms'] for k in range(env.k)])
    rewards = np.asarray([env.settled[k]['reward'] for k in range(env.k)], np.float32)
    evidence = dict(nodes=nodes, commands=commands, actual_links=links)
    audit = trajectory_audit(env, dict(evidence, rewards=rewards, reward_terms=terms))
    if not audit['passed']:
        raise ValueError('Flight engine safety or conservation check failed')
    positions = env.telemetry.starts[:env.t, :, :2]
    rf = geometry.arrays()
    physical_packet = recorder.arrays()
    noise, jammer = interference_regimes(env, rf['jammer'], nominal)
    observe = Observations(env)
    cases = {}
    for kind, p, repairs in scenarios(env, rf, noise):
        packets = packets_for(env, p, nodes)
        health, packet_stats = packet_summary(packets, env.t)
        actual = spec['kind'] == kind
        if actual:
            np.testing.assert_array_equal(radio_graph(p), links)
            np.testing.assert_array_equal(repairs[0], physical_packet['physical_truth'][:,0]>.5)
            np.testing.assert_array_equal(env.packets.ic, packets.ic)
            for key in ('born', 'status', 'delays'):
                a = getattr(env.packets, key)[:packets.ic[0]]
                b = getattr(packets, key)[:packets.ic[0]]
                np.testing.assert_array_equal(a, b)
        events, individual = network_events(env, p, repairs, rf, noise[kind], positions, observe if actual else None)
        cases[kind] = dict(native=actual, network_events=events, individual_link_runs=individual,
                           disconnected_seconds=float((~repairs[0]).sum()*.5), packet=packet_stats)
        if actual:
            cases[kind]['service_events'] = service_events(health, repairs[0], physical_packet)
            if len(cases[kind]['service_events']) != packet_stats['service_runs']:
                raise ValueError('Service event tally mismatch')
        evidence['healthy_'+kind] = np.packbits(repairs, axis=-1)
        evidence['service_'+kind] = np.packbits(health)
    evidence['frames'] = np.asarray(env.t)
    evidence['phases'] = np.asarray([row['phase'] for row in env.history], dtype=np.int8)
    return dict(spec=spec, mission=env.summary, audited=audit, jammer=jammer, rf_cases=cases,
                native_radio_exact=True, native_packet_exact=True, elapsed_s=perf_counter()-started), evidence
