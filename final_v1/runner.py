"""Native full-mission loop with the four-stage study's selectable controls."""
from collections import Counter
from time import perf_counter

import numpy as np
import torch

from adaptive_joint_expanded.contracts import seed_all
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.model import tensor_observation
from adaptive_joint_expanded.observation import Observer, critic_state, policy_inputs
from adaptive_joint_expanded.rollout import trajectory_audit
from adaptive_joint_expanded.world import World
from forecast_avoidance.controller import Planner
from forecast_graph.trace import attach
from forecast_learning.features import extract
from forecast_navigation.preview import capture
from forecast_prevention import runner as parent
from forecast_prevention.controller import Inbox

from .control import choose
from .policy import N_FEATURES, Policy, tensor
from .transport import CommandChannel

OVERLAYS = ("reactive", "v21", "physics_trigger", "no_short", "no_backup")
TRAIN_KEYS = ("raw", "context", "candidates", "base", "critic", "value", "chosen", "logp", "estimate")


def initialize():
    torch.set_num_threads(1)
    parent.initialize()
    parent.MODEL.requires_grad_(False)


def run(spec, arm, state_dict=None, training=False, transport=None, max_steps=None):
    started = perf_counter()
    seed_all(spec["seed"]+9173)
    fleet = parent.deployment_for(spec)
    env = World(**spec, deployment=fleet.key, observer=Observer(), max_steps=max_steps,
                fleet_cost=parent.CONFIG["fleet_cost_per_step"])
    if arm.startswith("ppo_"):
        policy = Policy(arm == "ppo_continuous")
        policy.load_state_dict(state_dict)
        policy.eval()
    else:
        policy = None
    channel = CommandChannel(spec["seed"], env.n, env.ns, env.baseline_end_step) if transport else None
    needs_forecast = arm in OVERLAYS or arm in ("v7", "ppo_continuous") or transport
    wire = attach(env) if needs_forecast else None
    inbox = Inbox(env.n, env.ns) if wire else None
    nav = None
    if wire:
        issue = wire.issue

        def on_issue(t, desired):
            nonlocal nav
            issue(t, desired)
            nav = capture(env.stream)

        wire.issue = on_issue
    index, checked = 0, 0
    commands, original, requested, interventions = [], [], [], []
    receipt_log, forced_log, hop_log, fallback_log, reasons, times = [], [], [], [], Counter(), []
    phase_log, alert_inputs, observed = [], [], []
    records = {key:[] for key in TRAIN_KEYS} if training else None
    missing = np.full(N_FEATURES, np.nan, dtype=np.float32)

    while not env.done:
        tick = perf_counter()
        now = env.k*3.
        phase = env.proposal["phase"]
        phase_log.append(phase)
        b = policy_inputs(env) if arm != "nominal" else None
        if b is not None:
            with torch.no_grad():
                stock, _ = parent.MODEL.policy(tensor_observation(b), "joint")
            logits = stock.numpy()
            base = logits.argmax(-1)
        else:
            base = np.zeros(env.n-env.ns, dtype=int)
        actions = base
        planner, base_data, detail = None, None, None
        raw = missing
        if wire:
            for frame in range(index, len(wire.plan_tx)):
                received = wire.plan_rx[frame][env.ns].copy()
                received[env.ns] = False
                inbox.receive(now, wire.plan_tx[frame][received])
            index = len(wire.plan_tx)
            local = env.last_x[env.ns]
            radio = inbox.state(now, local, env.last_mask[env.ns])
            if now >= 21.:
                planner = Planner(env.city, env.design, env.scenario, nav, env.k-1, env.baseline_duration, radio)
                if radio.local_rf_available and arm in ("v7", "ppo_continuous"):
                    base_data = extract(planner, base, local, phase)
                    raw = base_data[0]
            if arm == "v7" and not transport:
                alert_inputs.append(raw)
            if not training:
                observed.append((float(radio.local_rf_available), float((~radio.known[:env.ns]).sum()),
                                 float(radio.noise[env.ns]), float(local[0, 11])))
        if policy is not None:
            state = critic_state(env)
            with torch.no_grad():
                value = parent.MODEL.value(tensor(state))
                adjusted, estimate = policy(tensor(raw), tensor(b["context"]), tensor(b["candidates"]),
                                             stock, tensor(state), value)
                dist = torch.distributions.Categorical(logits=adjusted)
                selected = dist.sample() if training else adjusted.argmax(-1)
            actions = selected.numpy()
            if training:
                payload = dict(raw=raw, context=b["context"], candidates=b["candidates"], base=logits,
                               critic=state, value=float(value), chosen=actions,
                               logp=dist.log_prob(selected).numpy(), estimate=float(estimate))
                for key, data in payload.items():
                    records[key].append(data)
            logits = adjusted.numpy()

        if planner is not None:
            if arm in OVERLAYS:
                actions, detail, reason = choose(arm, planner, actions, logits, b["action_mask"], local,
                                                 phase, now, base_data, transport != "local_no_recovery")
                reasons[reason] += 1
            elif transport and policy is not None and transport != "local_no_recovery":
                actions, detail, reason = choose("reactive", planner, actions, logits, b["action_mask"], local, phase, now)
                reasons[reason] += 1

        original.append(base)
        requested.append(actions)
        if channel is not None:
            mask = env.last_mask[env.ns:]
            values = env.last_x[env.ns:]
            local_rf = (mask[:, :, 28] & mask[:, :, 29] & mask[:, :, 30] & (values[:, :, 30] <= 2.)).any(1)
            actions, received, hops, forced, reasons_at_tick = channel.apply(
                env.k, env.p_now, actions, b, local_rf, "neutral" if transport == "neutral" else "local")
            receipt_log.append(received)
            hop_log.append(hops)
            forced_log.append(forced)
            fallback_log.append(reasons_at_tick)
        command = np.zeros((env.n, 2))
        command[env.ns:] = OFFSETS[actions]
        commands.append(command)
        env.step(command)
        protected = env.history[-1]["protected"]
        env.previous_actions = np.zeros(fleet.nr, dtype=int) if protected else actions.copy()
        if detail is not None:
            relay = detail["changed_relay"]
            expected = detail["projected_fleet_endpoint"]
            if np.array_equal(actions, requested[-1]):
                np.testing.assert_array_equal(env.city.points[env.nodes], expected)
                checked += 1
            interventions.append(dict(detail, command_received=bool(received[relay-env.ns]) if channel else True,
                                      executed_as_requested=bool(np.array_equal(actions, requested[-1]) and not protected),
                                      motion_guard_blocked=protected,
                                      realized_endpoint=env.city.points[env.nodes[relay]].tolist()))
        times.append((perf_counter()-tick)*1000.)

    links = np.asarray(env.channel_available)
    generated = int(env.packets.ic[0])
    packet = dict(packet_born=env.packets.born[:generated], packet_status=env.packets.status[:generated],
                  packet_delay=env.packets.delays[:generated])
    metrics, network, service = parent.outcome(links, env.ns, packet)
    for event in interventions:
        f = int(event["time_s"]*2)
        event["actual_pre_action_connected"] = bool(network[f-1])
    terms = np.stack([env.settled[k]["reward_terms"] for k in range(env.k)])
    rewards = np.asarray([env.settled[k]["reward"] for k in range(env.k)], dtype=np.float32)
    nodes = np.asarray(env.node_log)
    data = dict(positions=env.city.points[nodes], nodes=nodes, links=links, network=network, service=service,
                commands=np.asarray(commands), requested=np.asarray(requested), original=np.asarray(original),
                phases=np.asarray(phase_log), rewards=rewards, reward_terms=terms)
    audit = trajectory_audit(env, data)
    if not audit["passed"]:
        raise ValueError("Final V1 native motion audit failed")
    row = dict(env.summary, **metrics, spec=spec, arm=arm, transport=transport, audited=audit,
               intervention=interventions, exact_endpoints=checked, decision_counts=dict(reasons),
               wall_s=perf_counter()-started, simulator_step_p95_ms=float(np.percentile(times, 95)))
    if training:
        return row, {**{k:np.asarray(v) for k,v in records.items()}, "rewards":rewards, "network":network}
    if alert_inputs:
        data["forecast_features"] = np.asarray(alert_inputs)
    if observed:
        data["observed"] = np.asarray(observed, dtype=np.float32)
    if channel:
        data.update(received=np.asarray(receipt_log), forced=np.asarray(forced_log), hops=np.asarray(hop_log),
                    fallback=np.asarray(fallback_log))
        row["stress_windows"] = [[i*3., (i+4)*3., r+env.ns] for i,r in channel.windows]
    return row, data
