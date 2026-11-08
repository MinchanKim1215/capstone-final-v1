"""Prospective mission planning from the issuing controller's current nav state.

No RF, packet truth, flight archive, future policy actions or future positions
are accepted. The held relay correction is a declared hypothesis.
"""
import copy
import numpy as np
from swarm_sim.mission import MOVE_DT,unique_goals,move_group
from usable.motion import reserved_goals,recovery_step

FIELDS=('reference','cursor','offset','nodes','observed','age','goal','recovery','escape_left','escape_goals','since_gain','phase','correction','trajectories','rng')


def capture(generator):
    current=generator.gi_frame.f_locals
    state={}
    for field in FIELDS:
        value=current[field]
        if field=='rng':state[field]=copy.deepcopy(value.bit_generator.state)
        elif field=='trajectories':state[field]=[np.array(x) for x in value[-12:]]
        elif isinstance(value,np.ndarray):state[field]=value.copy()
        else:state[field]=value
    return state


def preview(city,design,scenario,state,baseline_duration,world_step):
    s=copy.deepcopy(state);rng=np.random.default_rng();rng.bit_generator.state=s['rng']
    nodes=s['nodes'].copy();reference=s['reference'].copy();cursor=s['cursor'];offset=s['offset']
    observed=s['observed'].copy();age=s['age'].copy();goal=s['goal'];recovery=s['recovery']
    escape_left=s['escape_left'];escape_goals=s['escape_goals'];since_gain=s['since_gain']
    history=s['trajectories'];held=s['correction'];paths=[city.points[nodes]]
    for horizon in range(1,4):
        phase=0
        if cursor<len(city.route):
            remaining=scenario.reference_speed_mps*MOVE_DT
            while remaining>1e-9 and cursor<len(city.route):
                target=city.route[cursor];distance=np.linalg.norm(target-reference)
                if distance<=remaining+1e-9:reference=target.copy();remaining-=distance;cursor+=1
                else:reference+=(target-reference)*remaining/distance;remaining=0
        else:
            phase=1;missing=city.observable&~observed
            if not np.any(missing):phase=2
            else:
                if goal<0 or nodes[0]==goal or not np.any(city.visible[goal]&missing):
                    gains=np.sum(city.visible[:,missing],axis=1);goal=int(np.argmax(gains/(10+city.distance[nodes[0]])));recovery=False
                reference=city.points[city.next_node(nodes[0],goal)].copy()-offset[0]
        nominal=reference[None]+offset
        lag=np.linalg.norm(city.points[nodes[:design.n_search]]-nominal[:design.n_search],axis=1).max()/40
        protect=bool(np.any(held) and ((world_step+horizon)*3>baseline_duration*1.25 or lag>1.5 or phase==2))
        correction=np.zeros_like(held) if protect else held
        desired=nominal+correction;bounded=np.clip(desired,-200+scenario.grid_m/2,200-scenario.grid_m/2)
        goals=reserved_goals(city,bounded,goal) if phase==1 and goal>=0 else unique_goals(city,bounded)
        age+=1;priorities=age+city.distance[nodes,goals]*.001
        if phase==1:priorities[0]=priorities.max()+100
        new=move_group(nodes,goals,priorities,city.neighbors,city.distance,city.points,scenario.min_separation_m)
        if phase==1 and len(history)>=12:
            if len(np.unique(np.asarray(history[-12:]),axis=0))<=4 and not recovery:recovery=True
            if recovery:new=recovery_step(city,nodes,goal,scenario.min_separation_m)
            if since_gain>=60 and escape_left==0:
                escape_left=30;escape_goals=rng.choice(len(city.points),len(nodes),replace=False).astype(np.int32);since_gain=0
            if escape_left>0:
                new=move_group(nodes,escape_goals,rng.permutation(len(nodes)).astype(float),city.neighbors,city.distance,city.points,scenario.min_separation_m)
                escape_left-=1
                if escape_left==0:recovery=False;goal=-1
        age[new==goals]=0;before=int(observed.sum());nodes=new
        observed|=np.any(city.visible[nodes[:design.n_search]],axis=0)
        since_gain=0 if int(observed.sum())>before else since_gain+1
        history.append(nodes.copy());paths.append(city.points[nodes])
    return np.asarray(paths)

