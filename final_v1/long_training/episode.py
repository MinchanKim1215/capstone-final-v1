"""Collect on-policy proposals with the current Final V1 safety gates."""
from time import perf_counter

import numpy as np
import torch

from adaptive_joint_expanded.contracts import DEPLOYMENTS, deployment_mask, entry_for, public_context, seed_all
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.model import Controller, tensor_observation
from adaptive_joint_expanded.observation import Observer, critic_state, policy_inputs
from adaptive_joint_expanded.rollout import trajectory_audit
from adaptive_joint_expanded.targets import timing_labels
from adaptive_joint_expanded.world import World
from final_v1 import runner
from final_v1.transformer_upgrade.bridge import FleetGate
from final_v1.transformer_upgrade.capture import watch
from forecast_avoidance.controller import Planner
from forecast_graph.trace import attach
from forecast_navigation.preview import capture as capture_navigation
from forecast_prevention.controller import Inbox
from forecast_prevention.diagnostics import outage_runs

OBS_KEYS = ('context', 'candidates', 'action_mask', 'teacher', 'gain', 'geometry', 'support')


class TrainingController(Controller):
    def __init__(self, temperature=1.):
        super().__init__(warm_start=False)
        self.temperature = temperature

    def select(self, public_context, feasible):
        return super().select(public_context, feasible) / self.temperature


@torch.no_grad()
def episode(model, network_model, forecast_profile, spec, config, deterministic=False, max_steps=None):
    started = perf_counter()
    seed_all(spec['seed'] + 9173)
    model.eval()
    ctx = public_context(entry_for(spec['region']), spec['direction'], config['available_drones'],
                         config['minimum_search'])
    feasible = deployment_mask(config['available_drones'], config['minimum_search'])
    selector = model.select(torch.from_numpy(ctx), torch.from_numpy(feasible))
    distribution = torch.distributions.Categorical(logits=selector)
    selection = selector.argmax() if deterministic else distribution.sample()
    design = DEPLOYMENTS[int(selection)]
    observer = Observer()
    env = World(**spec, deployment=design.key, observer=observer, max_steps=max_steps,
                fleet_cost=config['fleet_cost_per_step'])
    wire = attach(env)
    issue = wire.issue
    nav = None

    def on_issue(time_s, desired):
        nonlocal nav
        issue(time_s, desired)
        nav = capture_navigation(env.stream)

    wire.issue = on_issue
    inbox = Inbox(env.n, env.ns)
    seen = overrides = 0
    columns = OBS_KEYS + ('state', 'value', 'chosen', 'logp', 'old_logits', 'endpoints',
                          'commands', 'executed')
    records = {key: [] for key in columns}
    with watch(model) as tap:
        gate = FleetGate(tap, model.forecaster.timing, network_model, forecast_profile['threshold'],
                         forecast_profile['network_weight'])
        while not env.done:
            now = env.k * 3.
            b = policy_inputs(env)
            logits, _ = model.policy(tensor_observation(b), 'joint')
            action_distribution = torch.distributions.Categorical(logits=logits)
            proposal = logits.argmax(dim=-1) if deterministic else action_distribution.sample()
            actions = proposal.numpy()
            for frame in range(seen, len(wire.plan_tx)):
                received = wire.plan_rx[frame][env.ns].copy()
                received[env.ns] = False
                inbox.receive(now, wire.plan_tx[frame][received])
            seen = len(wire.plan_tx)
            local = env.last_x[env.ns]
            radio = inbox.state(now, local, env.last_mask[env.ns])
            if now >= 21.:
                planner = Planner(env.city, env.design, env.scenario, nav, env.k - 1,
                                  env.baseline_duration, radio)
                actions, _, _ = gate.choose('v21', planner, actions, logits.numpy(), b['action_mask'],
                                            local, env.proposal['phase'], now)
            overrides += not np.array_equal(actions, proposal.numpy())
            state = critic_state(env)
            for name in OBS_KEYS:
                records[name].append(b[name])
            entries = dict(state=state, value=float(model.value(torch.from_numpy(state))),
                           chosen=proposal.numpy(), logp=action_distribution.log_prob(proposal).numpy(),
                           old_logits=logits.numpy(), endpoints=env.k * 6, executed=actions)
            for name, value in entries.items():
                records[name].append(value)
            command = np.zeros((env.n, 2))
            command[env.ns:] = OFFSETS[actions]
            records['commands'].append(command)
            env.step(command)
            env.previous_actions = np.zeros(design.nr, dtype=int) if env.history[-1]['protected'] else actions.copy()

    arrays = {name: np.asarray(values) for name, values in records.items()}
    links = np.asarray(env.channel_available)
    count = int(env.packets.ic[0])
    packet = dict(packet_born=env.packets.born[:count], packet_status=env.packets.status[:count],
                  packet_delay=env.packets.delays[:count])
    metrics, network, _ = runner.parent.outcome(links, env.ns, packet)
    # Use realized sustained outages for the reward and supervised labels only.
    starts = [start // 6 for start, _ in outage_runs(network) if start > 0]
    penalty = -config['onset_cost'] * np.bincount(starts, minlength=env.k)
    reward_terms = np.c_[np.stack([env.settled[k]['reward_terms'] for k in range(env.k)]), penalty]
    nodes = np.asarray(env.node_log)
    endpoints = arrays['endpoints']
    labels, valid = timing_labels(links, endpoints.clip(1), env.ns)
    valid[endpoints == 0] = False
    aux = endpoints >= 40
    aux_labels, aux_valid = labels[aux], valid[aux].copy()
    early = (aux_labels >= 6) & (aux_labels < 14)
    draws = np.random.default_rng(spec['seed'] + 713).random(aux_labels.shape)
    aux_valid &= early | (draws < .25)
    arrays.update(design=b['design'], rf_values=np.asarray(observer.values), rf_masks=np.asarray(observer.masks),
                  rf_intents=np.asarray(observer.intents), links=links, network=network, nodes=nodes,
                  positions=env.city.points[nodes], rewards=reward_terms.sum(axis=1).astype(np.float32),
                  reward_terms=reward_terms, public_context=ctx, deployment_mask=feasible,
                  selector_choice=np.array(int(selection)), selector_logp=np.array(float(distribution.log_prob(selection))),
                  selector_value=np.array(float(model.selector_value(torch.from_numpy(ctx)).squeeze())),
                  labels=labels, label_valid=valid, aux_endpoints=endpoints[aux],
                  aux_geometry=arrays['geometry'][aux], aux_support=arrays['support'][aux],
                  aux_labels=aux_labels, aux_valid=aux_valid)
    audit = trajectory_audit(env, arrays)
    if not audit['passed']:
        raise ValueError(f'Joint training flight failed safety or packet conservation: {audit}')
    row = dict(env.summary, **metrics, spec=spec, arm='joint_proposals', audited=audit,
               deterministic_policy=deterministic, selector_on_policy=not deterministic,
               selector_return=float(arrays['rewards'].sum() / 2400.), overrides=overrides,
               wall_s=perf_counter() - started)
    return row, arrays
