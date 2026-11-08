"""Reuse exact V11 query/target rows; only three mission-view input arrays differ."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status,PARENT
from .wire import extend_trace
from forecast_graph.features import LocalHistory,ReceivedTrace,build
from forecast_graph.data import TARGET_KEYS
from forecast_rebuild.dataset import Episode
from forecast_rebuild.planning import public_context

INPUT_KEYS=('node_shared','edge_shared','physics','global')
REPLACED=('node_shared','physics','global')


def arrays(item,profile,positions=None):
    with np.load(item['base_path'],allow_pickle=False) as a:
        keys=INPUT_KEYS+TARGET_KEYS+('query',)
        result={key:a[key] if positions is None else a[key][positions] for key in keys}
    if profile=='mission':
        with np.load(item['mission_path'],allow_pickle=False) as a:
            for key in REPLACED:result[key]=a[key] if positions is None else a[key][positions]
    elif profile!='original':raise ValueError('Unknown plan-view profile')
    return result


def cohort_arrays(ep,base,health,n,ns):
    q=base['query'];t,r=q.T
    if r.any():raise ValueError('Cohort cache requires the first relay')
    ends=np.asarray(ep.ends)[t];energy=np.asarray(ep.energy[ends-1,0])
    support=base['link_support'][:,:n]
    qualified=support&(energy[...,3]>.5)&(energy[...,4]>.5)&np.isfinite(energy[...,0])&np.isfinite(energy[...,1])&np.isfinite(energy[...,5])
    qualified &= (energy[...,2]>=1)&(energy[...,2]<=2)&(energy[...,5]>=.5)&(energy[...,5]<=1)
    return dict(ends=ends.astype(np.int32),links=np.asarray(ep.links),energy=energy,health=health,
        link_y=base['link_y'][:,:n],link_up=base['link_up'][:,:n],link_common=support,link_qualified=qualified,
        service_y=base['service_y'],service_up=base['service_up'],service_common=base['service_support'],n=np.array(n),ns=np.array(ns))


def build_one(parent):
    folder=ARTIFACTS/'views';folder.mkdir(exist_ok=True);path=folder/(parent['key']+'.npz');meta=path.with_suffix('.json')
    record=parent['source_record'];delta_path=ARTIFACTS/'deltas'/(parent['key']+'.npz');trace_path=PARENT/'traces'/(parent['key']+'.npz')
    contract=dict(base_data_sha256=parent['sha256'],delta_sha256=sha(delta_path),base_trace_sha256=sha(trace_path),
        source_pins={name:sha(ROOT/name) for name in ('data.py','preview.py','wire.py')})
    if meta.exists():
        old=read(meta)
        if old['contract']!=contract or sha(path)!=old['mission_sha256']:raise ValueError('Navigation view cache changed')
        return old
    if shutil.disk_usage(ARTIFACTS).free<18*1024**3:raise RuntimeError('Preserve18GiB reserve')
    with np.load(parent['path'],allow_pickle=False) as a:base={key:a[key] for key in a.files}
    ep=Episode(record);q=base['query'];ends=ep.ends[q[:,0]]
    with np.load(trace_path,allow_pickle=False) as a:tx=a['tx'];rx=a['rx']
    with np.load(delta_path,allow_pickle=False) as a:tx=extend_trace(tx,a['delta'])
    received=ReceivedTrace(tx,rx,ep.values.shape[2],record['n_search']);local=LocalHistory(ep.values,ep.masks,ep.intents,ep.values.shape[2],record['n_search'])
    city,_=public_context(record);city.radio.enable_cache();parts={key:[] for key in REPLACED}
    for start in range(0,len(q),128):
        f=build(local,received,np.c_[ends[start:start+128],q[start:start+128,1]],city)
        for key in ('node_local','edge_local','edge_shared'):np.testing.assert_equal(f[key],base[key][start:start+128])
        for key in REPLACED:parts[key].append(f[key])
    tmp=path.with_suffix('.working.npz');np.savez_compressed(tmp,**{k:np.concatenate(v) for k,v in parts.items()});tmp.replace(path)
    cohort=None
    if record['role']!='train':
        from forecast_service import ARTIFACTS as V10
        with np.load(V10/'service_cache'/(record['key']+'.npz'),allow_pickle=False) as a:health=a['service_healthy']
        cohort=ARTIFACTS/'cohorts'/(record['key']+'.npz');cohort.parent.mkdir(exist_ok=True)
        np.savez_compressed(cohort,**cohort_arrays(ep,base,health,local.n,local.ns))
    item=dict(key=parent['key'],role=parent['role'],region=parent['region'],source_record=record,queries=len(q),
        base_path=parent['path'],base_sha256=parent['sha256'],mission_path=str(path),mission_sha256=sha(path),
        cohort_path=str(cohort) if cohort else None,cohort_sha256=sha(cohort) if cohort else None,
        contract=contract,created_at=now(),bytes=path.stat().st_size)
    write(meta,item);return item


def main():
    verify()
    while not (ARTIFACTS/'delta_status.json').exists():
        p=ARTIFACTS/'collect.stderr.log'
        if p.exists() and 'Traceback (most recent call last)' in p.read_text(encoding='utf-8',errors='replace'):raise RuntimeError('Delta collection failed')
        time.sleep(5)
    parents=read(PARENT/'data_manifest.json')['records'];items={};status('mission_graph_feature_views',complete=0,total=len(parents))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(build_one,r):i for i,r in enumerate(parents)}
        for f in as_completed(jobs):
            items[jobs[f]]=f.result();status('mission_graph_feature_views',complete=len(items),total=len(parents))
            write(ARTIFACTS/'data_manifest.json',dict(records=[items[i] for i in sorted(items)],created_at=now()));print(len(items),'/',len(parents),flush=True)
    write(ARTIFACTS/'data_status.json',dict(state='completed',complete=len(parents),created_at=now()))


if __name__=='__main__':main()
