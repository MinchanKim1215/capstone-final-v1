"""Reconstruct source flights without changing motion, RF, packets or beacons."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from . import ROOT,ARTIFACTS,WORKSPACE,PARENT,NAV
from .contracts import read,write,sha,now,verify,status
from .recorder import Recorder
from .wire import extend_trace,original_packet
from forecast_navigation.wire import attach_navigation

def one(parent,limit=None):
    verify();r=parent['source_record'];key=parent['key'];folder=ARTIFACTS/('fixture_v2' if limit is not None else 'replay_v2');folder.mkdir(exist_ok=True)
    target=folder/(key+('s'+str(limit) if limit is not None else '')+'.npz');meta=target.with_suffix('.json')
    pins={k:sha(ROOT/k) for k in ('recorder.py','instrumented_packet.py','wire.py','replay.py')}
    if meta.exists():
        item=read(meta)
        if item['source_pins']!=pins or sha(target)!=item['sha256']:raise ValueError('Instrumented replay cache changed')
        return item
    if shutil.disk_usage(ARTIFACTS).free<18*1024**3:raise RuntimeError('Preserve18GiB disk reserve')
    from adaptive_joint_expanded.world import World
    began=time.monotonic();source_meta=read(r['metadata_path']);maximum=source_meta['request'].get('max_steps')
    env=World(**r['spec'],deployment=r['deployment'],observer=None,max_steps=maximum);tel=attach_navigation(env);rec=Recorder(env)
    with np.load(r['array_path']) as a:commands=a['commands'];nodes=a['nodes'];links=a['links']
    with rec.installed():
        for k,command in enumerate(commands if limit is None else commands[:limit]):
            env.step(command);np.testing.assert_equal(env.nodes,nodes[k+1]);np.testing.assert_equal(np.asarray(env.channel_available[-6:]),links[k*6:k*6+6])
    values=rec.arrays();tx=np.asarray(tel.nav_tx);aug=extend_trace(tx,values['observed'])
    for a,b in zip(tx.reshape(-1,128),aug.reshape(-1,128)):np.testing.assert_equal(a,original_packet(b))
    trace=WORKSPACE/'forecast_graph/artifacts/traces'/(key+'.npz');delta=NAV/'deltas'/(key+'.npz')
    if r['role']!='confirmation':
        from forecast_navigation.wire import extend_trace as nav_extend
        with np.load(trace) as a:base=a['tx'];rx=a['rx']
        with np.load(delta) as a:base=nav_extend(base,a['delta'])
        np.testing.assert_equal(tx,base[:len(tx)]);np.testing.assert_equal(np.packbits(np.asarray(tel.plan_rx),axis=-1),rx[:len(tx)])
        if limit is None:
            service=WORKSPACE/'forecast_service/artifacts/service_cache'/(key+'.npz')
            with np.load(service) as a:
                for field in ('packet_born','packet_status','packet_delay'):np.testing.assert_equal(values[field],a[field])
    if limit is None:
        if not env.done:raise ValueError('Source mission incomplete')
        for field in ('generated_packets','delivered_packets','retries','ttl_drops','retry_drops','overflow_drops','timely_pdr'):
            if env.summary[field]!=source_meta[field]:raise ValueError('Instrumentation changed data-plane outcome: '+field)
    # Physiology/packettruth and measured state are separate named arrays; only
    # observed+received state can enter feature.py in subsequent stages.
    values.update(tx=aug,rx=np.packbits(np.asarray(tel.plan_rx),axis=-1),n=np.array(env.n),ns=np.array(env.ns),frames=np.array(env.t))
    temp=target.with_suffix('.working.npz');np.savez_compressed(temp,**values);temp.replace(target)
    item=dict(key=key,role=r['role'],region=r['region'],source_record=r,parent_item=parent,path=str(target),sha256=sha(target),source_pins=pins,created_at=now(),frames=env.t,
        bytes=target.stat().st_size,wall_seconds=time.monotonic()-began,full_mission=limit is None,RF_and_commands_exact=True,original_128B_packets_restore_exactly=True,
        packet_outcomes_exact=limit is None,diagnostic_truth_never_feature_input=True)
    write(meta,item);return item

def main(scope='selection',workers=3):
    parents=read(PARENT/'data_manifest.json')['records'];parents=[p for p in parents if scope=='all' or p['role']==scope]
    done={};status('stage1_selection_replay' if scope=='selection' else 'stage2_development_replay',complete=0,total=len(parents),next_action='재생 결과 일치 검사 후 장애 원인표를 생성한다.')
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(one,p):i for i,p in enumerate(parents)}
        for f in as_completed(jobs):
            done[jobs[f]]=f.result();write(ARTIFACTS/(scope+'_replay_manifest.json'),dict(records=[done[i] for i in sorted(done)],created_at=now()))
            status('stage1_selection_replay' if scope=='selection' else 'stage2_development_replay',complete=len(done),total=len(parents));print(scope,len(done),'/',len(parents),flush=True)
    write(ARTIFACTS/(scope+'_replay_status.json'),dict(state='completed',total=len(done),created_at=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',default='selection');p.add_argument('--workers',type=int,default=3);args=p.parse_args();main(args.scope,args.workers)
