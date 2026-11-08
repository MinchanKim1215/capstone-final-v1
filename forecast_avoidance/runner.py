"""Run the actual policy, motion, radio and packet stack for each arm."""
from collections import Counter
from time import perf_counter

import numpy as np
import torch

from .controller import Planner, choose
from forecast_prevention import runner as parent
from forecast_prevention.controller import Inbox
from forecast_prevention.diagnostics import factorial_radio, minimal_repairs, outage_runs
from forecast_navigation.preview import capture
from forecast_navigation.wire import attach_navigation
from adaptive_joint_expanded.contracts import seed_all
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.model import tensor_observation
from adaptive_joint_expanded.observation import Observer, policy_inputs
from adaptive_joint_expanded.rollout import trajectory_audit
from adaptive_joint_expanded.world import World


def run(spec, arm, shadow_arm=None, max_steps=None, check_every_projection=False, branch_at=None):
    if arm == 'geometry':
        return parent.run(spec, arm, max_steps=max_steps)
    tick = perf_counter()
    seed_all(spec['seed'] + 9173)
    fleet = parent.deployment_for(spec)
    env = World(**spec, deployment=fleet.key, observer=Observer(), max_steps=max_steps,
                fleet_cost=parent.CONFIG['fleet_cost_per_step'])
    wire = attach_navigation(env)
    inbox = Inbox(env.n, env.ns)
    issue = wire.issue
    previous_nav = None

    def issued(t, desired):
        nonlocal previous_nav
        issue(t, desired)
        # The original planner has just issued the segment. Copy only navigation state.
        previous_nav = capture(env.stream)

    wire.issue = issued
    commands, actions_log, interventions, times, forecasts = [], [], [], [], []
    reasons = Counter()
    receipt_count = 0
    checked = 0
    while not env.done:
        observation = policy_inputs(env)
        with torch.no_grad():
            logits, _ = parent.MODEL.policy(tensor_observation(observation), 'joint')
        logits = logits.numpy()
        base = logits.argmax(axis=-1)
        actions = base
        now = env.k * 3.
        expected = None
        detail = None
        for frame in range(receipt_count, len(wire.nav_tx)):
            received = wire.plan_rx[frame][env.ns].copy()
            received[env.ns] = False
            inbox.receive(now, wire.nav_tx[frame][received])
        receipt_count = len(wire.nav_tx)
        state = inbox.state(now, env.last_x[env.ns], env.last_mask[env.ns])
        if previous_nav is not None and (now >= 21 or check_every_projection):
            planner = Planner(env.city, env.design, env.scenario, previous_nav, env.k-1,
                              env.baseline_duration, state)
            if check_every_projection:
                expected = planner.path(base)[1]
            if now >= 21 and (branch_at is None or now == branch_at):
                start = perf_counter()
                candidate, result = choose(shadow_arm or arm, planner, base, logits, observation['action_mask'])
                times.append((perf_counter() - start)*1000.)
                reasons[result['reason']] += 1
                if shadow_arm is not None:
                    forecasts.append(dict(time_s=now, reason=result['reason'], expert=result.get('expert'),
                                          predicted=result.get('predicted_onsets_in_s', []),
                                          estimated_connected=result.get('estimated_connected')))
                elif not np.array_equal(candidate, base):
                    actions = candidate
                    expected = np.asarray(result['projected_fleet_endpoint'])
                    relay = result['changed_relay'] - env.ns
                    detail = dict(result, time_s=now, base_action=int(base[relay]), action=int(actions[relay]))
        command = np.zeros((env.n, 2))
        command[env.ns:] = OFFSETS[actions]
        commands.append(command)
        actions_log.append(base)
        env.step(command)
        protected = env.history[-1]['protected']
        env.previous_actions = np.zeros(fleet.nr, int) if protected else actions.copy()
        if expected is not None:
            np.testing.assert_array_equal(env.city.points[env.nodes], expected)
            checked += 1
        if detail is not None:
            relay = detail['changed_relay']
            detail['motion_guard_blocked'] = protected
            detail['actual_endpoint'] = env.city.points[env.nodes[relay]].tolist()
            interventions.append(detail)
    links = np.asarray(env.channel_available)
    nodes = np.asarray(env.node_log)
    positions = env.city.points[nodes]
    generated = int(env.packets.ic[0])
    packets = dict(packet_born=env.packets.born[:generated], packet_status=env.packets.status[:generated],
                   packet_delay=env.packets.delays[:generated])
    metrics, network, service = parent.outcome(links, env.ns, packets)
    for item in interventions:
        step = int(item['time_s']//3)
        item['actual_pre_action_connected'] = bool(network[step*6-1])
        relay = item['changed_relay']
        item['actual_own_displacement_m'] = float(np.linalg.norm(positions[step+1, relay] - positions[step, relay]))
    evidence = dict(nodes=nodes, commands=np.asarray(commands), original_actions=np.asarray(actions_log),
                    links=links, network=network, service=service)
    terms = np.stack([env.settled[k]['reward_terms'] for k in range(env.k)])
    rewards = np.asarray([env.settled[k]['reward'] for k in range(env.k)], np.float32)
    audit = trajectory_audit(env, dict(evidence, rewards=rewards, reward_terms=terms))
    if not audit['passed']:
        raise ValueError('Actual trajectory audit failed')
    row = dict(env.summary, **metrics, spec=spec, arm=arm, intervention=interventions,
               decision_counts=dict(reasons), audited=audit, exact_endpoint_checks=checked,
               wall_s=perf_counter()-tick, control_p95_ms=float(np.percentile(times, 95)) if times else None)
    if shadow_arm is not None:
        runs = outage_runs(network)
        frames = np.asarray([f for start, _ in runs for f in (start, start+1)], dtype=int)
        repairs, components = factorial_radio(env, positions, links, frames)
        row['cut_evidence'] = [dict(onset_s=start*.5, stop_s=stop*.5, components=components[start],
                                    repairs=minimal_repairs([r for r in repairs[start] if r in repairs[start+1]]))
                               for start, stop in runs]
        row['advisory'] = forecasts
    return row, evidence
