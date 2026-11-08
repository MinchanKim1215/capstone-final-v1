"""Reproduce frozen missions and extract observations available before each query."""
from time import perf_counter

import numpy as np
import torch

from adaptive_joint_expanded.contracts import seed_all
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.model import tensor_observation
from adaptive_joint_expanded.observation import Observer,policy_inputs
from adaptive_joint_expanded.rollout import trajectory_audit
from adaptive_joint_expanded.world import World
from forecast_avoidance.controller import Planner
from forecast_graph.trace import attach
from forecast_learning.features import COLUMNS,extract
from forecast_navigation.preview import capture
from forecast_prevention import runner as parent
from forecast_prevention.controller import Inbox

from .features import NAMES,observed_features

LEADS = (.5,1.,2.,3.,5.,7.)


def initialize():
    parent.initialize()
    parent.MODEL.requires_grad_(False)


def run(spec, reference=None, expected_features=None,max_steps=None):
    started = perf_counter()
    seed_all(spec["seed"]+9173)
    fleet = parent.deployment_for(spec)
    env = World(**spec,deployment=fleet.key,observer=Observer(),max_steps=max_steps,
                fleet_cost=parent.CONFIG["fleet_cost_per_step"])
    wire = attach(env)
    nav = None
    issue = wire.issue

    def issued(t,desired):
        nonlocal nav
        issue(t,desired)
        nav = capture(env.stream)

    wire.issue = issued
    inbox = Inbox(env.n,env.ns)
    records,commands,physical = [],[],[]
    index = 0
    while not env.done:
        now = env.k*3.
        if reference is not None:
            command = reference[env.k]
            indices = (command[env.ns:,None] == OFFSETS[None]).all(-1)
            if not indices.any(-1).all():
                raise ValueError("Unknown original command")
            actions = indices.argmax(-1)
        else:
            b = policy_inputs(env)
            with torch.no_grad():
                logits,_ = parent.MODEL.policy(tensor_observation(b),"joint")
            actions = logits.numpy().argmax(-1)
            command = np.zeros((env.n,2))
            command[env.ns:] = OFFSETS[actions]
        if expected_features is None:
            for frame in range(index,len(wire.plan_tx)):
                received = wire.plan_rx[frame][env.ns].copy()
                received[env.ns] = False
                inbox.receive(now,wire.plan_tx[frame][received])
            index = len(wire.plan_tx)
            state = inbox.state(now,env.last_x[env.ns],env.last_mask[env.ns])
            if now >= 21.:
                if state.local_rf_available:
                    planner = Planner(env.city,env.design,env.scenario,nav,env.k-1,env.baseline_duration,state)
                    features,warning,_,_,_ = extract(planner,actions,env.last_x[env.ns],env.proposal["phase"])
                    physical.append(features)
                    records.append([now,warning.shield_score])
                else:
                    physical.append(np.full(len(COLUMNS),np.nan))
                    records.append([now,np.nan])
        commands.append(command)
        env.step(command)
        env.previous_actions = np.zeros(fleet.nr,dtype=int) if env.history[-1]["protected"] else actions.copy()
    if expected_features is not None:
        physical,saved = expected_features
        times = saved[:,0]
    else:
        saved = np.asarray(records)
        times = saved[:,0]
    links = np.asarray(env.channel_available)
    count = int(env.packets.ic[0])
    packet = dict(packet_born=env.packets.born[:count],packet_status=env.packets.status[:count],
                  packet_delay=env.packets.delays[:count])
    metrics,network,service = parent.outcome(links,env.ns,packet)
    values = wire.x[:env.t,env.ns]
    masks = wire.mask[:env.t,env.ns]
    received = observed_features(values,masks,env.ns)
    observed = received[(times*2).astype(int)-1]
    events = np.asarray([cut[0] for cut in metrics["persistent_cut_intervals"] if cut[0]>0.])
    aligned = np.full((len(events),len(LEADS),len(NAMES)),np.nan,dtype=np.float32)
    # Offline event alignment only chooses past queries. Its future labels never enter observed_features.
    for i,onset in enumerate(events):
        for j,lead in enumerate(LEADS):
            query = onset-lead
            if query > 0.:
                aligned[i,j] = received[int(round(query*2))-1]
    terms = np.stack([env.settled[k]["reward_terms"] for k in range(env.k)])
    rewards = np.asarray([env.settled[k]["reward"] for k in range(env.k)],dtype=np.float32)
    nodes = np.asarray(env.node_log)
    data = dict(nodes=nodes,commands=np.asarray(commands),links=links,network=network,service=service,
                positions=env.city.points[nodes],x=np.asarray(physical,dtype=np.float32),rf=observed,
                times=times,event_aligned=aligned,onsets=events,physical_scores=saved[:,1])
    audit = trajectory_audit(env,dict(data,rewards=rewards,reward_terms=terms))
    if not audit["passed"]:
        raise ValueError("Replay native safety/packet audit failed")
    row = dict(env.summary,**metrics,spec=spec,audited=audit,wall_s=perf_counter()-started,
               input_frames=env.t,queries=len(times),new_features=len(NAMES))
    # Two direct regression checks on real received measurements.
    query = min(env.t-2,128)
    prefix = observed_features(values[:query],masks[:query],env.ns)
    np.testing.assert_array_equal(prefix,received[:query])
    hidden = values[:query].copy()
    hidden[~masks[:query]] = 123456.
    np.testing.assert_array_equal(prefix,observed_features(hidden,masks[:query],env.ns))
    row["future_prefix_invariant"] = row["masked_value_invariant"] = True
    return row,data
