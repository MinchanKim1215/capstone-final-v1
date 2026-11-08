"""Run a complete survey with continuous motion and half-second RF samples."""
from time import perf_counter

import numpy as np
import torch

from adaptive_joint_expanded.baseline import paired_shadow
from adaptive_joint_expanded.features import THRESHOLD, widest
from cooperative.packets import PacketSim, hop_probabilities
from final_v1_fast.motion import poses
from final_v1_fast.runner import intervals
from final_v1_fast.sensing import Commands, Sensors
from forecast_shield.forecaster import geometry
from swarm_sim.mission import MAX_MISSION, SHARE_SECONDS

from . import actor
from .context import interference, survey_for
from .controller import PLANNER_DELAY_TICKS, options_for, state_at
from .features import N_FEATURES, RadioHistory, physical_risk
from .navigation import DT, Navigator

MODES = ('physics_3s', 'physics_05s', 'learned_05s', 'ppo_05s')
REWARD = dict(time_per_s=-.002, cut_per_s=-.06, persistent_start=-.65,
              timely=.02, generated=-.02, relay_meter=-.0007, command=-.015, incomplete=-5.)


def load_actor(path):
    model = actor.ActorCritic()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state'])
    return model.eval()


def full_mission(spec, mode, models=None, policy=None, explore=False, collect=False,
                 max_seconds=MAX_MISSION, command_latency=.1):
    if mode not in MODES or (explore and (mode != 'ppo_05s' or policy is None)):
        raise ValueError('Invalid half-second policy mode')
    if mode in ('learned_05s', 'ppo_05s') and models is None:
        raise ValueError('Learned policy needs its frozen forecast heads')
    if mode == 'ppo_05s' and policy is None:
        raise ValueError('PPO policy checkpoint is required')
    started = perf_counter()
    city, scenario, fleet, survey, anchor_s = survey_for(spec['region'], spec['direction'])
    n, ns = fleet.n, fleet.ns
    nav = Navigator(city, ns, survey)
    motion = nav.motion
    steps = int(round(max_seconds / DT))
    if not 0 < steps <= int(MAX_MISSION / DT):
        raise ValueError('Unexpected mission time budget')
    seed = spec['seed']
    shadow = paired_shadow(seed, steps//5 + 1, n, .5, scenario.shadow_sigma_db,
                           scenario.shadow_correlation_s)
    jammer, jam_start, jam_stop, jam_power = interference(spec, city, anchor_s)
    sensor = Sensors(n, ns, seed+113, 1.)
    sensor.random = np.random.default_rng(seed+113).random((steps, n, n), dtype=np.float32)
    commands = Commands(n, ns, np.zeros(n-ns, np.int32), seed+227, command_latency)
    commands.random = np.random.default_rng(seed+227).random((steps, 3, n, n), dtype=np.float32)
    delivery_ticks = int(np.ceil(command_latency / DT - 1e-8))
    packet = PacketSim(n, ns, city.visible, DT, seed+331,
                       max_duration_s=max_seconds+3., reserved_slots=n)
    history = RadioHistory(n)
    rng = np.random.default_rng(seed+27101)
    plan_ticks = 30 if mode == 'physics_3s' else 5
    pending = None
    broadcast_until = -1
    last_change = -100
    x0, velocity = motion.positions()
    pair = np.triu_indices(n, 1)

    def gap(xy):
        return float(np.sqrt(np.sum((xy[pair[0]] - xy[pair[1]])**2, axis=1)).min())

    minimum = gap(x0)
    maximum_speed = maximum_accel = 0.
    travel = np.zeros(n)
    previous_xy = x0
    previous_velocity = velocity
    used_edges = set()
    network, links, positions, margin_log, offset_log = [], [], [], [], []
    events, applied_events, forecasts, times, physics_scores = [], [], [], [], []
    decisions, rewards = [], []
    reserved = tested = assessments = 0
    timings = []
    p = np.empty((n,n))
    received = np.empty((n,n))
    sinr = np.empty((n,n))
    net = True
    for frame in range(steps):
        now = frame / 10.
        nav.update(now)
        if nav.finished_at is not None and now-nav.finished_at >= SHARE_SECONDS-1e-8 and frame % 5 == 0:
            break
        sensor.deliver(now)
        before = commands.applied.copy()
        commands.deliver(now)
        for relay in np.flatnonzero(before != commands.applied):
            applied_events.append([now, int(relay)+ns, int(commands.applied[relay]),
                                   float(commands.last_stamp[relay])])
        if pending is not None and frame >= pending[0]:
            _, requested, event = pending
            commands.requested = requested.copy()
            broadcast_until = frame+2
            event['published_s'] = now
            pending = None
        current = previous_xy
        onset = False
        if frame % 5 == 0:
            path_loss, _ = geometry(city, scenario, current[None])
            noise = np.zeros(n)
            if jam_start <= now < jam_stop:
                receivers = np.c_[current, np.full(n, scenario.altitude_m)]
                loss = city.radio.propagation(jammer[None], receivers, scenario)[0]
                noise = 10**((jam_power-loss)/10.)
            p = hop_probabilities(path_loss[0], noise, shadow[frame//5], received, sinr)
            graph = (p >= .9) & (p.T >= .9)
            net = bool(widest(np.where(graph, 1., -1.), ns).min() >= 0.)
            network.append(net)
            links.append(graph)
            positions.append(current.copy())
            margin_log.append(float(widest(np.minimum(sinr, sinr.T)-THRESHOLD, ns).min()))
            offset_log.append(commands.applied.copy())
            onset = len(network) >= 3 and network[-3] and not network[-2] and not net
        intervention_cost = 0.
        if frame % plan_ticks == 0:
            tick = perf_counter()
            estimate = sensor.estimate(now)
            history.accept(now, sensor.remote_sinr[:, ns], sensor.remote_stamp[:, ns])
            if estimate.local_rf:
                assessments += 1
                state = state_at(nav, scenario, now, commands.applied, estimate, history)
                raw = state[0]
                early, short = models.predict(raw) if models else (0., 0.)
                if collect:
                    forecasts.append(raw)
                    times.append(now)
                    physics_scores.append(physical_risk(state[1]))
                if frame-last_change >= 5:
                    options, count = options_for(nav, scenario, now, commands.requested, commands.applied,
                                                 estimate, state, bool(models and early >= models.threshold), delivery_ticks)
                    tested += count
                    if len(options) > 1:
                        if policy is None:
                            selected = int(options[1].physical_value > 0.)
                        else:
                            action_inputs = actor.inputs(raw, early, short, options)
                            selected, value, logp = actor.select(policy, action_inputs, explore, rng)
                            if explore:
                                decisions.append((frame, *action_inputs, selected, value, logp))
                        option = options[selected]
                        if option.event is not None:
                            last_change = frame
                            event = dict(option.event, actual_connected_at_request=net,
                                         probability_early=early if models else None,
                                         probability_short=short if models else None,
                                         ready_s=(frame+PLANNER_DELAY_TICKS)/10.,
                                         earliest_delivery_s=(frame+PLANNER_DELAY_TICKS)/10.+command_latency)
                            events.append(event)
                            pending = (frame+PLANNER_DELAY_TICKS, option.offsets.copy(), event)
                            intervention_cost = REWARD['command']
            timings.append((perf_counter()-tick)*1000.)
        beacon_slots = sensor.transmit(frame, now, p, received, sinr)
        command_slots = 0
        if frame <= broadcast_until or (frame % 10 == 0 and commands.requested.any()):
            command_slots = commands.transmit(frame, now, p)
        reserved_now = beacon_slots+command_slots
        packet.reserved_slots = (frame*20) % 200 + reserved_now
        ids = city.tree.query(current)[1].astype(np.int32)
        generated_before, timely_before = int(packet.ic[0]), int(packet.ic[8])
        packet.interval(p, ids)
        reserved += reserved_now
        xy, velocity = nav.step(commands.applied)
        midpoint, _ = poses(city.points, motion.start, motion.end, np.maximum(0., motion.elapsed-.05))
        minimum = min(minimum, gap(xy), gap(midpoint))
        maximum_speed = max(maximum_speed, float(np.linalg.norm(velocity, axis=1).max()))
        maximum_accel = max(maximum_accel, float(np.linalg.norm(velocity-previous_velocity, axis=1).max()/DT))
        distance = np.linalg.norm(xy-previous_xy, axis=1)
        travel += distance
        for a, b in zip(motion.start, motion.end):
            if a != b:
                used_edges.add((int(a),int(b)))
        previous_xy, previous_velocity = xy, velocity
        if explore:
            rewards.append((REWARD['time_per_s'] + REWARD['cut_per_s'] * (not net))*DT +
                           REWARD['persistent_start'] * onset +
                           REWARD['timely'] * (int(packet.ic[8])-timely_before) +
                           REWARD['generated'] * (int(packet.ic[0])-generated_before) +
                           REWARD['relay_meter'] * float(distance[ns:].sum()) + intervention_cost)
    else:
        frame = steps
    elapsed = frame / 10.
    packet.drain()
    coverage = int(nav.covered[city.observable].sum())/survey.observable
    completed = bool(nav.finished_at is not None and elapsed-nav.finished_at >= SHARE_SECONDS-1e-8 and coverage == 1.)
    if not completed and rewards:
        rewards[-1] += REWARD['incomplete']
    network = np.asarray(network, bool)
    cuts = intervals(network, .5)
    safe = bool(minimum >= 5.-1e-7 and maximum_speed <= 5.+1e-7 and maximum_accel <= 2.5+1e-6 and
                all(city.graph[a,b] > 0 for a,b in used_edges))
    conserved = int(packet.ic[0]) == int(packet.ic[1]+packet.ic[5]+packet.ic[6]+packet.ic[7])
    if not safe or not conserved:
        raise ValueError(f'Half-second physics audit failed: {minimum=} {maximum_speed=} {maximum_accel=} {conserved=}')
    row = dict(spec=spec, mode=mode, deployment=fleet.key, total_drones=n, search_drones=ns,
               duration_s=elapsed, nominal_duration_s=survey.baseline_s, mission_completed=completed,
               coverage=coverage,
               waypoints_reached=nav.stage, waypoints_total=len(survey.waypoints),
               onsets=sum(a>0 for a,b in cuts), cuts=cuts, initial_cut=bool(cuts and cuts[0][0] == 0.),
               cut_seconds=sum(b-a for a,b in cuts), network_fraction=float(network.mean()),
               generated=int(packet.ic[0]), timely=int(packet.ic[8]), delivered=int(packet.ic[1]),
               timely_pdr=float(packet.ic[8]/max(1,packet.ic[0])),
               relay_travel_m=float(travel[ns:].sum()), search_travel_m=float(travel[:ns].sum()),
               safety_passed=safe, packet_conserved=conserved, min_separation_m=minimum,
               max_speed_mps=maximum_speed, max_acceleration_mps2=maximum_accel,
               assessments=assessments, candidates_tested=tested, events=events, applied_changes=applied_events,
               prevention_requests=sum(event['actual_connected_at_request'] for event in events),
               planner_p95_ms=float(np.quantile(timings,.95)),
               planner_over_200ms=sum(ms>200. for ms in timings), plan_ticks=len(timings),
               command_relay_receipts=commands.received, command_relay_transmissions=commands.sent,
               beacon_count=sensor.beacons, reserved_airtime=reserved*.005/elapsed,
               wall_s=perf_counter()-started)
    arrays = dict(network=network, links=np.asarray(links, bool), positions=np.asarray(positions, np.float32),
                  margin_db=np.asarray(margin_log, np.float32), applied_offsets=np.asarray(offset_log, np.int8),
                  times=np.asarray(times, np.float32), features=np.asarray(forecasts, np.float32).reshape(-1,N_FEATURES),
                  physics=np.asarray(physics_scores, np.float32))
    if explore:
        arrays['rewards'] = np.asarray(rewards, np.float32)
        for index, name in enumerate(('action_frames', 'states', 'options', 'priors', 'valid', 'actions', 'values', 'logp')):
            arrays[name] = np.asarray([record[index] for record in decisions])
    return row, arrays
