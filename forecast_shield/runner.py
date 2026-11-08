"""Native one-second forecasts and unchanged three-second PPO dispatch."""
from time import perf_counter

import numpy as np
import torch

from .forecaster import predict
from forecast_avoidance.controller import Planner
from forecast_navigation.preview import capture
from forecast_navigation.wire import attach_navigation
from forecast_prevention import runner as previous
from forecast_prevention.controller import Inbox
from forecast_prevention.diagnostics import factorial_radio, minimal_repairs, outage_runs
from adaptive_joint_expanded.contracts import seed_all
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.model import tensor_observation
from adaptive_joint_expanded.observation import Observer, policy_inputs
from adaptive_joint_expanded.rollout import trajectory_audit
from adaptive_joint_expanded.world import World


COLUMNS = ('time_s', 'v18', 'planner', 'shield', 'short', 'onset_in_s', 'building', 'building_change',
           'missing_search', 'unknown_rf', 'elapsed_ms')


class FlightWarnings(Observer):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.next_receipt = 0
        self.nav = None

    def attach(self, env):
        self.env = env
        self.inbox = Inbox(env.n, env.ns)
        self.wire = attach_navigation(env)
        old_issue = self.wire.issue

        def issue(t, desired):
            old_issue(t, desired)
            self.nav = capture(env.stream)

        self.wire.issue = issue

    def read(self, now, values, masks):
        relay = self.env.ns
        for i in range(self.next_receipt, len(self.wire.nav_tx)):
            received = self.wire.plan_rx[i][relay].copy()
            received[relay] = False
            self.inbox.receive(now, self.wire.nav_tx[i][received])
        self.next_receipt = len(self.wire.nav_tx)
        return self.inbox.state(now, values[relay], masks[relay])

    def warn(self, now, state, knots, elapsed):
        env = self.env
        tick = perf_counter()
        result = predict(env.city, env.scenario, state, env.ns, knots, elapsed)
        self.rows.append([now, result.legacy_score if not elapsed else np.nan, result.planner_score,
                          result.shield_score, result.short_score, result.onset_in_s, result.building_support,
                          result.changing_building_support, result.missing_search_beacons,
                          result.unknown_rf_receivers, (perf_counter()-tick)*1000.])
        return result

    def observe(self, now, values, masks, intents):
        answer = super().observe(now, values, masks, intents)
        if now >= 21 and now % 3 in (1., 2.):
            # This plan was published before these RF observations arrived.
            if self.wire.issued_time != now-now%3:
                raise ValueError('Committed plan and RF observations are misaligned')
            state = self.read(now, values, masks)
            self.warn(now, state, self.wire.mission_nodes, now%3)
        return answer


def run(spec, reference=None, max_steps=None):
    started = perf_counter()
    seed_all(spec['seed']+9173)
    fleet = previous.deployment_for(spec)
    observer = FlightWarnings()
    env = World(**spec, deployment=fleet.key, observer=observer, max_steps=max_steps,
                fleet_cost=previous.CONFIG['fleet_cost_per_step'])
    observer.attach(env)
    commands = []
    checked = 0
    while not env.done:
        if reference is None:
            with torch.no_grad():
                logits, _ = previous.MODEL.policy(tensor_observation(policy_inputs(env)), 'joint')
            actions = logits.numpy().argmax(axis=-1)
            command = np.zeros((env.n,2))
            command[env.ns:] = OFFSETS[actions]
        else:
            # Only this decision's pre-recorded command is supplied to the predictor.
            command = reference[env.k]
            equal = (OFFSETS[None] == command[env.ns:,None]).all(axis=-1)
            if not equal.any(axis=1).all():
                raise ValueError('Archived controller issued an unknown action')
            actions = equal.argmax(axis=-1)
        now = env.k*3.
        first = None
        if now >= 21:
            state = observer.read(now, env.last_x, env.last_mask)
            planner = Planner(env.city, env.design, env.scenario, observer.nav, env.k-1,
                              env.baseline_duration, state)
            knots = planner.path(actions)
            first = knots[1]
            observer.warn(now, state, knots, 0.)
        commands.append(command.copy())
        env.step(command)
        env.previous_actions = np.zeros(fleet.nr, int) if env.history[-1]['protected'] else actions.copy()
        if first is not None:
            np.testing.assert_array_equal(env.city.points[env.nodes], first)
            checked += 1
    links = np.asarray(env.channel_available)
    nodes = np.asarray(env.node_log)
    positions = env.city.points[nodes]
    generated = int(env.packets.ic[0])
    packet = dict(packet_born=env.packets.born[:generated], packet_status=env.packets.status[:generated],
                  packet_delay=env.packets.delays[:generated])
    metrics, network, service = previous.outcome(links, env.ns, packet)
    scores = np.asarray(observer.rows).reshape(-1,len(COLUMNS))
    np.testing.assert_array_equal(scores[:,0], np.arange(21., env.t*.5))
    evidence = dict(nodes=nodes, commands=np.asarray(commands), links=links, network=network, service=service,
                    scores=scores, columns=np.asarray(COLUMNS))
    terms = np.stack([env.settled[k]['reward_terms'] for k in range(env.k)])
    rewards = np.asarray([env.settled[k]['reward'] for k in range(env.k)], np.float32)
    audit = trajectory_audit(env, dict(evidence, rewards=rewards, reward_terms=terms))
    if not audit['passed']:
        raise ValueError('Actual trajectory failed safety audit')
    cuts = outage_runs(network)
    frames = np.asarray([f for start,_ in cuts for f in (start,start+1)], dtype=int)
    repairs, _ = factorial_radio(env, positions, links, frames)
    events = [dict(onset_s=start*.5, duration_s=(stop-start)*.5,
                   repairs=minimal_repairs([r for r in repairs[start] if r in repairs[start+1]]))
              for start,stop in cuts if start]
    return dict(env.summary, **metrics, spec=spec, audited=audit, events=events,
                exact_endpoints=checked, live_hz=1, wall_s=perf_counter()-started), evidence
