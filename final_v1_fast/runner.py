"""10 Hz radio/data/command loop; all cadence arms share physical clocks."""
from time import perf_counter
import numpy as np
from adaptive_joint_expanded.features import THRESHOLD,widest
from cooperative.packets import PacketSim,hop_probabilities
from forecast_shield.forecaster import geometry
from .motion import Fleet,poses
from .sensing import Sensors,Commands
from .controller import Controller
from .protocol import CONFIG,ARMS
from .source import program_for,fine_shadow,context as source_context


def intervals(network,dt=.1,minimum_s=1.):
    a=np.r_[True,network,True]
    bounds=np.flatnonzero(a[1:]!=a[:-1])
    return [[round(float(start*dt),6),round(float(stop*dt),6)] for start,stop in zip(bounds[::2],bounds[1::2])
            if (stop-start)*dt>=minimum_s-1e-8]


def pair_min(xy):
    i,j=np.triu_indices(len(xy),1)
    return float(np.linalg.norm(xy[i]-xy[j],axis=-1).min())


def run(case,arm,latency=.1):
    started=perf_counter()
    initial,program,context=program_for(case)
    meta,city,scenario,fleet,jammer=context
    cfg=ARMS[arm];n=fleet.n;ns=fleet.ns
    motion=Fleet(city,initial,ns)
    shadow=fine_shadow(case,n)
    sensor=Sensors(n,ns,case['fine_seed']+113,cfg['beacon_s'],cfg.get('hybrid',False))
    commands=Commands(n,ns,program.initial_offsets,case['fine_seed']+227,latency)
    control=Controller(city,scenario,program,fleet,latency+CONFIG['planning_compute_s'])
    packet=PacketSim(n,ns,city.visible,.1,case['fine_seed']+331,
                     max_duration_s=CONFIG['branch_duration_s']+3.,reserved_slots=n)
    xyz=[];velocity=[];network=[];links=[];margin=[];requested=[];applied=[];decisions=[]
    reserved=0;separation=pair_min(city.points[initial]);midmin=separation
    plan_ticks=int(round(cfg['plan_s']*10))
    emergency_ticks=int(round(cfg['emergency_s']*10)) if cfg['emergency_s'] else None
    previous_velocity=np.zeros((n,2));max_accel=0.;max_speed=0.
    old_xy,_=motion.positions()
    xyz.append(old_xy.copy());velocity.append(previous_velocity.copy())
    # Warm compiled numerical paths before measuring control latency.
    motion.preview(program.schedule(0.,commands.applied),np.array([0.,.1]))
    results_pending=[];busy_until=-1.;busy_skips=0
    elapsed_cpu=[]
    used_edges=set()
    for frame in range(int(CONFIG['branch_duration_s']*10)):
        tick=perf_counter();now=frame*.1
        sensor.deliver(now);commands.deliver(now)
        remaining=[]
        for due,result,critical,projected in results_pending:
            if due<=now+1e-8:
                if result is not None:
                    commands.requested=result.copy()
                commands.requested_fast[:]=False
                commands.requested_fast[list(critical)]=True
                control.published_onset=projected
            else:
                remaining.append((due,result,critical,projected))
        results_pending=remaining
        estimate=sensor.estimate(now)
        sensor.fast_mask=commands.fast_mask.copy()
        regular=frame%plan_ticks==0
        emergency=bool(not regular and emergency_ticks and frame%emergency_ticks==0 and control.emergency_due(now,estimate))
        event=None
        if (regular or emergency) and now<busy_until-1e-8:
            busy_skips+=1
        elif regular or emergency:
            candidate,event=control.decide(motion,now,commands.requested,commands.applied,estimate,emergency)
            busy_until=now+CONFIG['planning_compute_s']
            results_pending.append((busy_until,candidate.copy() if event else None,
                                    control.critical_nodes,control.projected_onset))
            if event:
                event['result_ready_s']=round(busy_until,6)
                event['earliest_apply_s']=round(busy_until+latency,6)
                decisions.append(dict(event,frame=frame))
        current_xy,current_velocity=motion.positions()
        loss,_=geometry(city,scenario,current_xy[None])
        absolute=case['start_s']+now
        active=jammer['intermittent_start_s']<=absolute<jammer['intermittent_stop_s']
        noise=np.zeros(n)
        if active:
            receiver=np.c_[current_xy,np.full(n,50.)]
            jl=city.radio.propagation(np.array([jammer['jammer_xy_m']+[2.]]),receiver,scenario)[0]
            noise=10**((jammer['tx_power_dbm']-jl)/10)
        received=np.empty((n,n));sinr=np.empty((n,n))
        p=hop_probabilities(loss[0],noise,shadow[frame],received,sinr)
        graph=(p>=.9)&(p.T>=.9)
        if frame==0:
            original=source_context(case['source_path'])[5]['links'][int(round(case['start_s']*2))]
            np.testing.assert_array_equal(graph,original)
        net=bool((widest(np.where(graph,1.,-1.),ns).min())>=0.)
        true_margin=float(widest(np.minimum(sinr,sinr.T)-THRESHOLD,ns).min())
        network.append(net);links.append(graph);margin.append(true_margin)
        if event:
            event['actual_connected_at_request']=net
            decisions[-1]['actual_connected_at_request']=net
        beacon_slots=sensor.transmit(frame,now,p,received,sinr)
        command_slots=commands.transmit(frame,now,p)
        reserved_now=beacon_slots+command_slots
        if reserved_now>=20:
            raise ValueError('Control/beacon packets exhausted the frame')
        # Existing packet engine reserves [0,reserved_slots) within each second.
        # Setting the end to the current 100 ms frame's offset+count reserves
        # exactly count slots in this frame without changing its equations.
        packet.reserved_slots=(frame*20)%200+reserved_now
        ids=city.tree.query(current_xy)[1].astype(np.int32)
        packet.interval(p,ids)
        reserved+=reserved_now
        goals=program.goals(now,commands.applied)
        new_xy,new_velocity=motion.step(goals)
        used_edges.update((int(a),int(b)) for a,b in zip(motion.start,motion.end) if a!=b)
        midpoint,_=poses(city.points,motion.start,motion.end,np.maximum(0.,motion.elapsed-.05))
        midmin=min(midmin,pair_min(midpoint))
        separation=min(separation,pair_min(new_xy),pair_min(midpoint))
        max_accel=max(max_accel,float(np.linalg.norm(new_velocity-previous_velocity,axis=1).max()/.1))
        max_speed=max(max_speed,float(np.linalg.norm(new_velocity,axis=1).max()))
        previous_velocity=new_velocity.copy()
        xyz.append(new_xy.copy());velocity.append(new_velocity.copy())
        requested.append(commands.requested.copy());applied.append(commands.applied.copy())
        elapsed_cpu.append((perf_counter()-tick)*1000.)
    packet.drain()
    xyz=np.asarray(xyz);velocity=np.asarray(velocity);network=np.asarray(network,bool)
    cuts=intervals(network)
    travel=np.linalg.norm(np.diff(xyz,axis=0),axis=-1).sum(axis=0)
    reversals=0
    for i in range(ns,n):
        last=None
        for v in velocity[:,i]:
            if np.linalg.norm(v)<1e-6:
                continue
            direction=v/np.linalg.norm(v)
            if last is not None and direction@last<-.5:
                reversals+=1
            last=direction
    packet_conserved=int(packet.ic[0])==int(packet.ic[1]+packet.ic[5]+packet.ic[6]+packet.ic[7])
    if not packet_conserved or separation<5.-1e-7 or max_speed>5.+1e-7 or max_accel>2.5+1e-6:
        raise ValueError(f'Safety/conservation failure: {separation=} {max_speed=} {max_accel=} {packet_conserved=}')
    if any(city.graph[a,b]<=0 for a,b in used_edges):
        raise ValueError('Motion left the swept-polygon-safe graph')
    # Every traveled edge belongs to the original swept-polygon-safe graph.
    row=dict(case_key=case['key'],spec=case['spec'],start_s=case['start_s'],anchor_kind=case['anchor_kind'],
        source_pattern=case['source_pattern'],arm=arm,latency_s=latency,duration_s=CONFIG['branch_duration_s'],
        total_drones=n,search_drones=ns,
        initial_radio_matches_source=True,
        onsets=sum(a>0 for a,b in cuts),initial_cut=bool(cuts and cuts[0][0]==0),
        cut_seconds=sum(b-a for a,b in cuts),any_cut_seconds=float((~network).sum()*.1),cuts=cuts,
        network_connected_fraction=float(network.mean()),generated_packets=int(packet.ic[0]),
        delivered_packets=int(packet.ic[1]),timely_packets=int(packet.ic[8]),
        timely_pdr=float(packet.ic[8]/max(1,packet.ic[0])),
        data_pdr=float(packet.ic[1]/max(1,packet.ic[0])),
        ttl_drops=int(packet.ic[5]),retry_drops=int(packet.ic[6]),overflow_drops=int(packet.ic[7]),
        reserved_airtime_fraction=float(reserved*.005/CONFIG['branch_duration_s']),
        relay_travel_m=float(travel[ns:].sum()),search_travel_m=float(travel[:ns].sum()),
        reversals=reversals,blocked_edge_attempts=motion.blocked,
        interventions=decisions,prevention_requests=sum(e['reason']=='prevention' for e in decisions),
        emergency_requests=sum(e['emergency'] for e in decisions),
        connected_requests=sum(e['actual_connected_at_request'] for e in decisions),
        assessments=control.assessments,candidates=control.candidates,
        planning_compute_s=CONFIG['planning_compute_s'],busy_planning_skips=busy_skips,
        decision_p50_ms=float(np.median(control.calls)) if control.calls else 0.,
        decision_p95_ms=float(np.quantile(control.calls,.95)) if control.calls else 0.,
        decision_max_ms=float(max(control.calls)) if control.calls else 0.,
        decisions_over_budget=sum(t>CONFIG['prediction_time_budget_ms'] for t in control.calls),
        frame_p95_ms=float(np.quantile(elapsed_cpu,.95)),
        command_received=commands.received,command_sent=commands.sent,
        actual_command_delay_p95_s=float(np.quantile(commands.delivery_delays,.95)) if commands.delivery_delays else None,
        beacons_sent=sensor.beacons,beacons_received=sensor.delivered,
        minimum_separation_m=separation,maximum_speed_mps=max_speed,maximum_acceleration_mps2=max_accel,
        packet_conserved=packet_conserved,safety_passed=True,wall_s=perf_counter()-started)
    data=dict(positions=xyz.astype(np.float32),velocity=velocity.astype(np.float32),network=network,
              links=np.asarray(links),margin_db=np.asarray(margin,np.float32),
              requested=np.asarray(requested,np.float32),applied=np.asarray(applied,np.float32),
              source_nodes=initial,shadow=shadow.astype(np.float32),
              used_edges=np.asarray(sorted(used_edges),np.int32).reshape(-1,2))
    return row,data
