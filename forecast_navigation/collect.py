"""Replay335unchangedflights; save48-byte plan deltas, never duplicate RF truth."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,source_records,PARENT,verify,status
from .wire import attach_navigation


def one(record):
    from adaptive_joint_expanded.world import World
    folder=ARTIFACTS/'deltas';folder.mkdir(exist_ok=True);path=folder/(record['key']+'.npz');meta=path.with_suffix('.json')
    base=PARENT/'traces'/(record['key']+'.npz')
    contract=dict(base_trace_sha256=sha(base),source_sha256=record['array_sha256'],
        source_pins={name:sha(ROOT/name) for name in ('preview.py','wire.py','collect.py')})
    if meta.exists():
        old=read(meta)
        if old['contract']!=contract or sha(path)!=old['sha256']:raise ValueError('Mission delta changed')
        return old
    env=World(**record['spec'],deployment=record['deployment'],observer=None);tel=attach_navigation(env)
    with np.load(record['array_path'],allow_pickle=False) as a:commands=a['commands'];nodes=a['nodes']
    for k,c in enumerate(commands):env.step(c);np.testing.assert_equal(env.nodes,nodes[k+1])
    with np.load(base,allow_pickle=False) as a:
        np.testing.assert_equal(np.asarray(tel.plan_tx),a['tx'])
        np.testing.assert_equal(np.packbits(np.asarray(tel.plan_rx),axis=-1),a['rx'])
        np.testing.assert_equal(tel.sinr[:env.t].astype(np.float16),a['sinr'])
    original=read(record['metadata_path'])
    for key in ('generated_packets','delivered_packets','retries','ttl_drops','retry_drops','overflow_drops','timely_pdr'):
        if env.summary[key]!=original[key]:raise ValueError('Navigation preview changed data-plane outcome')
    revised=np.asarray(tel.nav_tx);delta=np.concatenate([revised[...,30:66],revised[...,114:126]],-1)
    tmp=path.with_suffix('.working.npz');np.savez_compressed(tmp,delta=delta);tmp.replace(path)
    value=dict(key=record['key'],role=record['role'],region=record['region'],path=str(path),sha256=sha(path),contract=contract,
        original_packets_reconstruct_exactly=True,RF_and_RX_exact=True,trajectories_and_packets_exact=True,bytes=path.stat().st_size,created_at=now())
    write(meta,value);return value


def main():
    verify();records=source_records();items={};status('mission_plan_delta_collection',complete=0,total=len(records))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(one,r):i for i,r in enumerate(records)}
        for future in as_completed(pending):
            items[pending[future]]=future.result();status('mission_plan_delta_collection',complete=len(items),total=len(records))
            write(ARTIFACTS/'delta_manifest.json',dict(records=[items[i] for i in sorted(items)],created_at=now()));print(len(items),'/',len(records),flush=True)
    write(ARTIFACTS/'delta_status.json',dict(state='completed',complete=len(records),created_at=now()))


if __name__=='__main__':main()
