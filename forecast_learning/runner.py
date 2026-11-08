"""Run the native V7 world with causal features and optional guarded actuation."""
from collections import Counter
from time import perf_counter

import numpy as np
import torch

from adaptive_joint_expanded.contracts import seed_all
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.model import tensor_observation
from adaptive_joint_expanded.observation import Observer, policy_inputs
from adaptive_joint_expanded.rollout import trajectory_audit
from adaptive_joint_expanded.world import World
from forecast_avoidance.controller import Planner
from forecast_graph.trace import attach
from forecast_navigation.preview import capture
from forecast_prevention import runner as parent
from forecast_prevention.controller import Inbox
from forecast_prevention.diagnostics import factorial_radio, minimal_repairs, outage_runs

from .features import COLUMNS, extract


def run(spec, reference=None, arm='baseline', profile=None, max_steps=None, diagnosis=True):
    started = perf_counter()
    seed_all(spec['seed']+9173)
    fleet = parent.deployment_for(spec)
    env = World(**spec,deployment=fleet.key,observer=Observer(),max_steps=max_steps,
                fleet_cost=parent.CONFIG['fleet_cost_per_step'])
    wire = attach(env)
    inbox = Inbox(env.n,env.ns)
    nav = None
    original_issue = wire.issue

    def issue(t,desired):
        nonlocal nav
        original_issue(t,desired)
        nav = capture(env.stream)

    wire.issue = issue
    commands, x, scores, events, times = [], [], [], [], []
    observations = Counter()
    index = checked = 0
    while not env.done:
        logits = None
        if reference is None:
            obs = policy_inputs(env)
            with torch.no_grad():
                logits, _ = parent.MODEL.policy(tensor_observation(obs),'joint')
            logits = logits.numpy()
            actions = logits.argmax(axis=-1)
        else:
            equal = (reference[env.k,env.ns:,None,:]==OFFSETS[None]).all(axis=-1)
            if not equal.any(axis=1).all():
                raise ValueError('Archived relay command is unknown')
            actions = equal.argmax(axis=-1)
        now = env.k*3.
        for i in range(index,len(wire.plan_tx)):
            received = wire.plan_rx[i][env.ns].copy()
            received[env.ns] = False
            inbox.receive(now,wire.plan_tx[i][received])
        index = len(wire.plan_tx)
        state = inbox.state(now,env.last_x[env.ns],env.last_mask[env.ns])
        endpoint = None
        change = None
        if now>=21:
            tick = perf_counter()
            planner = Planner(env.city,env.design,env.scenario,nav,env.k-1,env.baseline_duration,state)
            local,phase = env.last_x[env.ns],env.proposal['phase']
            if arm=='learned':
                from .control import choose
                actions,change,endpoint,reason = choose(profile,planner,actions,logits,obs['action_mask'],local,phase,now)
                observations[reason] += 1
            elif state.local_rf_available:
                features,warning,_,_,endpoint = extract(planner,actions,local,phase)
                x.append(features)
                scores.append([now,warning.shield_score,warning.short_score,warning.legacy_score])
            else:
                x.append(np.full(len(COLUMNS),np.nan,np.float32))
                scores.append([now,np.nan,np.nan,np.nan])
            times.append((perf_counter()-tick)*1000.)
        command = np.zeros((env.n,2))
        command[env.ns:] = OFFSETS[actions]
        commands.append(command)
        env.step(command)
        protect = env.history[-1]['protected']
        env.previous_actions = np.zeros(fleet.nr,int) if protect else actions.copy()
        if endpoint is not None:
            np.testing.assert_array_equal(env.city.points[env.nodes],endpoint)
            checked += 1
        if change is not None:
            events.append(dict(change,time_s=now,motion_guard_blocked=protect))
    links = np.asarray(env.channel_available)
    nodes = np.asarray(env.node_log)
    generated = int(env.packets.ic[0])
    packet = dict(packet_born=env.packets.born[:generated],packet_status=env.packets.status[:generated],
                  packet_delay=env.packets.delays[:generated])
    metrics, network, service = parent.outcome(links,env.ns,packet)
    for event in events:
        event['actual_pre_action_connected'] = bool(network[int(event['time_s']*2)-1])
    values = np.asarray(scores).reshape(-1,4)
    evidence = dict(nodes=nodes,commands=np.asarray(commands),links=links,network=network,service=service,
                    x=np.asarray(x,np.float32).reshape(-1,len(COLUMNS)),scores=values)
    if arm!='learned':
        np.testing.assert_array_equal(values[:,0],np.arange(21.,env.t*.5,3.))
    terms = np.stack([env.settled[k]['reward_terms'] for k in range(env.k)])
    rewards = np.asarray([env.settled[k]['reward'] for k in range(env.k)],np.float32)
    audit = trajectory_audit(env,dict(evidence,rewards=rewards,reward_terms=terms))
    if not audit['passed']:
        raise ValueError('Native safety audit failed')
    actual = outage_runs(network)
    repairs = {}
    if diagnosis:
        frames = np.asarray([f for start,_ in actual for f in (start,start+1)],dtype=int)
        repairs,_ = factorial_radio(env,env.city.points[nodes],links,frames)
    cuts = [dict(onset_s=start*.5,duration_s=(stop-start)*.5,
                 repairs=minimal_repairs([r for r in repairs[start] if r in repairs[start+1]]) if start in repairs else [])
            for start,stop in actual if start]
    return dict(env.summary,**metrics,spec=spec,arm=arm,events=cuts,intervention=events,
                decision_counts=dict(observations),audited=audit,exact_endpoints=checked,
                wall_s=perf_counter()-started,control_p95_ms=float(np.percentile(times,95)) if times else None),evidence
