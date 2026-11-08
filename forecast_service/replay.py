"""Exact commanded-flight replay to recover settled packet service labels."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,now,verify,status,V9

LABEL_VERSION='settled_birth_deadline250ms_rolling1s_pdr90_v1'


def service_health(born,statuses,delays,frames,deadline=.25,window=1.,minimum_pdr=.9):
    # At physical time q*.5 only births whose deadlines have elapsed count.
    born=np.asarray(born,float);timely=(np.asarray(statuses)==1)&(np.asarray(delays)<=deadline+1e-10)
    cumulative=np.r_[0,np.cumsum(timely,dtype=np.int64)]
    cutoff=np.arange(frames)*.5-deadline
    hi=np.searchsorted(born,cutoff,side='right');lo=np.searchsorted(born,cutoff-window,side='right')
    total=hi-lo;good=cumulative[hi]-cumulative[lo]
    ratio=np.divide(good,total,out=np.full(frames,np.nan),where=total>0)
    valid=(np.arange(frames)*.5>=deadline+window)&(total>=20)
    healthy=np.where(valid,ratio>=minimum_pdr,True)
    return dict(service_healthy=healthy,timely_pdr_1s=ratio,service_valid=valid,settled_packet_count=total)


def replay(record):
    from adaptive_joint_expanded.world import World
    from forecast_rebuild.dataset import Episode
    folder=ARTIFACTS/'service_cache';folder.mkdir(exist_ok=True);path=folder/(record['key']+'.npz');meta=path.with_suffix('.json')
    request=dict(source_sha256=record['array_sha256'],spec=record['spec'],deployment=record['deployment'],label_version=LABEL_VERSION)
    if meta.exists():
        prior=read(meta)
        if prior['request']!=request or sha(path)!=prior['sha256']:raise ValueError('Committed service replay differs')
        return prior
    if sha(record['array_path'])!=record['array_sha256']:raise ValueError('Physical source archive changed')
    episode=Episode(record)
    with np.load(record['array_path'],allow_pickle=False) as raw:commands=raw['commands'];nodes=raw['nodes']
    env=World(**record['spec'],deployment=record['deployment'],observer=None)
    started=time.monotonic()
    for step,command in enumerate(commands):
        env.step(command)
        if not np.array_equal(env.nodes,nodes[step+1]):raise ValueError('Command replay changed trajectory')
        begin=step*6;end=begin+6
        np.testing.assert_equal(np.asarray(env.channel_available[-6:]),np.asarray(episode.links[begin:end]))
        np.testing.assert_equal(env.last_x[env.ns:].astype(np.float16),episode.values[end-1])
        np.testing.assert_equal(env.last_mask[env.ns:],episode.masks[end-1])
        np.testing.assert_equal(env.last_intent[env.ns:].astype(np.float16),episode.intents[end-1])
    if not env.done or len(episode.values)!=env.t:raise ValueError('Replay did not cover exactly the mission')
    original=read(record['metadata_path']);summary=env.summary
    for key in ('generated_packets','delivered_packets','ttl_drops','retry_drops','overflow_drops','retries'):
        if summary[key]!=original[key]:raise ValueError('Packet conservation replay differs: '+key)
    for key in ('pdr','timely_pdr','delay_mean_s','delay_p95_s'):
        if summary[key] is None or original[key] is None:
            if summary[key]!=original[key]:raise ValueError('Undefined packet outcome differs: '+key)
        elif not np.isclose(summary[key],original[key],atol=1e-12,rtol=1e-12):raise ValueError('Packet outcome replay differs: '+key)
    g=int(env.packets.ic[0]);packets=env.packets
    arrays=dict(packet_source=packets.source[:g],packet_dest=packets.dest[:g],packet_born=packets.born[:g],
        packet_status=packets.status[:g],packet_delay=packets.delays[:g],packet_hops=packets.hops[:g],
        packet_frames=np.asarray(env.packet_frames),issued_targets=np.array(env.telemetry.desired),
        jam_active=(np.arange(env.t)*.5>=env.jam_start)&(np.arange(env.t)*.5<env.jam_end))
    arrays.update(service_health(arrays['packet_born'],arrays['packet_status'],arrays['packet_delay'],env.t))
    temporary=path.with_suffix('.working.npz');np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    result=dict(key=record['key'],role=record['role'],region=record['region'],path=str(path),sha256=sha(path),request=request,
        frames=env.t,generated_packets=g,physical_link_frames_exact=True,positions_exact=True,local_telemetry_endpoint_exact=True,
        final_packet_outcomes_exact=True,wall_seconds=time.monotonic()-started,created_at=now())
    write(meta,result);return result


def main(workers=4):
    verify();data=read(V9/'dataset.json');records=[r for r in data['rows'] if r['source_kind']=='fresh_v9' and r['role'] in ('train','selection','calibration')]
    if len(records)!=335:raise ValueError('Expected245/45/45 fresh physical sources')
    status('stage2_exact_packet_replay',complete=0,total=len(records));results={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(replay,r):i for i,r in enumerate(records)}
        for future in as_completed(pending):
            index=pending[future];results[index]=future.result()
            write(ARTIFACTS/'replay_manifest.json',dict(records=[results[i] for i in sorted(results)],label_version=LABEL_VERSION,created_at=now()))
            status('stage2_exact_packet_replay',complete=len(results),total=len(records))
            print(len(results),'/',len(records),flush=True)
    write(ARTIFACTS/'replay_status.json',dict(state='completed',complete=len(results),total=len(records),created_at=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    try:main(args.workers)
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',stage='packet_replay',error=repr(error),updated_at=now()));raise
