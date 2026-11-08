"""One alternate continuation from the same issued, collision-resolved segment."""
import copy

import numpy as np

from adaptive_joint_expanded.features import OFFSETS
from forecast_navigation.preview import preview
from swarm_sim.mission import MOVE_DT, move_group, unique_goals
from usable.motion import recovery_step, reserved_goals


def advance(planner, correction):
    """Advance a copy of the existing navigation snapshot by one proposal."""
    city, scenario = planner.city, planner.scenario
    s = copy.deepcopy(planner.nav)
    ns = planner.design.n_search
    nodes, offset = s['nodes'], s['offset']
    phase = 0
    if s['cursor'] < len(city.route):
        remaining = scenario.reference_speed_mps*MOVE_DT
        while remaining > 1e-9 and s['cursor'] < len(city.route):
            target = city.route[s['cursor']]
            delta = target-s['reference']
            length = np.linalg.norm(delta)
            if length <= remaining+1e-9:
                s['reference'] = target.copy()
                s['cursor'] += 1
                remaining -= length
            else:
                s['reference'] += delta*remaining/length
                remaining = 0.
    else:
        missing = city.observable & ~s['observed']
        phase = 1 if missing.any() else 2
        if phase == 1:
            goal = s['goal']
            if goal < 0 or nodes[0] == goal or not np.any(city.visible[goal] & missing):
                gain = city.visible[:,missing].sum(axis=1)
                s['goal'] = goal = int(np.argmax(gain/(10+city.distance[nodes[0]])))
                s['recovery'] = False
            s['reference'] = city.points[city.next_node(nodes[0],goal)].copy()-offset[0]
    nominal = s['reference'][None]+offset
    lag = np.linalg.norm(city.points[nodes[:ns]]-nominal[:ns],axis=1).max()/40
    if (planner.previous_step+1)*3>planner.baseline_duration*1.25 or lag>1.5 or phase==2:
        correction = np.zeros_like(correction)
    desired = np.clip(nominal+correction,-200+scenario.grid_m/2,200-scenario.grid_m/2)
    goal = s['goal']
    goals = reserved_goals(city,desired,goal) if phase==1 and goal>=0 else unique_goals(city,desired)
    s['age'] += 1
    priorities = s['age']+city.distance[nodes,goals]*.001
    if phase==1:
        priorities[0] = priorities.max()+100.
    new = move_group(nodes,goals,priorities,city.neighbors,city.distance,city.points,scenario.min_separation_m)
    history = s['trajectories']
    if phase==1 and len(history)>=12:
        if len(np.unique(np.asarray(history[-12:]),axis=0))<=4:
            s['recovery'] = True
        if s['recovery']:
            new = recovery_step(city,nodes,goal,scenario.min_separation_m)
        rng = np.random.default_rng()
        rng.bit_generator.state = s['rng']
        if s['since_gain']>=60 and s['escape_left']==0:
            s['escape_left'] = 30
            s['escape_goals'] = rng.choice(len(city.points),len(nodes),replace=False).astype(np.int32)
            s['since_gain'] = 0
        if s['escape_left']>0:
            order = rng.permutation(len(nodes)).astype(float)
            new = move_group(nodes,s['escape_goals'],order,city.neighbors,city.distance,city.points,scenario.min_separation_m)
            s['escape_left'] -= 1
            if s['escape_left']==0:
                s['recovery'] = False
                s['goal'] = -1
        s['rng'] = rng.bit_generator.state
    s['age'][new==goals] = 0
    before = int(s['observed'].sum())
    s['observed'] |= np.any(city.visible[new[:ns]],axis=0)
    s['since_gain'] = 0 if s['observed'].sum()>before else s['since_gain']+1
    s['nodes'] = new
    s['trajectories'] = (history+[new.copy()])[-12:]
    s['phase'] = phase
    return s


def continuations(planner, actions):
    held = planner.path(actions)
    if not actions.any():
        return held,held
    correction = np.zeros_like(held[0])
    correction[planner.design.n_search:] = OFFSETS[actions]
    state = advance(planner,correction)
    np.testing.assert_array_equal(planner.city.points[state['nodes']],held[1])
    state['correction'] = np.zeros_like(correction)
    tail = preview(planner.city,planner.design,planner.scenario,state,
                   planner.baseline_duration,planner.previous_step+1)
    return held,np.concatenate([held[:1],tail[:3]],axis=0)
