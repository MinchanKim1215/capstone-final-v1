"""Compact once-per-time graph cache and exact inherited event query cohort."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
from . import ROOT,ARTIFACTS,PARENT,NAV
from .contracts import read,write,sha,now,verify,status
from .features import build
from forecast_rebuild.dataset import Episode
from forecast_graph.episodes import RawEpisode
from forecast_graph.features import LocalHistory,ReceivedTrace
from forecast_navigation.wire import extend_trace
from forecast_rebuild.planning import public_context

def prepare_one(parent):
    verify();began=time.monotonic();key=parent['key'];role=parent['role'];r=parent['source_record']
    path=ARTIFACTS/'data'/(key+'.npz');path.parent.mkdir(exist_ok=True);meta=path.with_suffix('.json')
    pins={name:sha(ROOT/name) for name in ('data.py','features.py')}
    if meta.exists():
        old=read(meta)
        if old['source_pins']!=pins or old['parent_sha256']!=parent['sha256'] or sha(path)!=old['sha256']:raise ValueError('Temporal cache changed')
        return old
    if shutil.disk_usage(ARTIFACTS).free<18*1024**3:raise RuntimeError('Preserve18GiB disk reserve')
    if sha(parent['path'])!=parent['sha256']:raise ValueError('Parent cohort changed')
    with np.load(parent['path'],allow_pickle=False) as a:
        values={k:a[k] for k in ('ends','query','uniform','bags','y','up','support','links','logits')}
    ep=Episode(r) if r.get('cache_path') else RawEpisode(r);n=ep.values.shape[2];ns=r['n_search']
    trace_root=ARTIFACTS/'fresh_traces' if role=='confirmation' else ROOT.parent/'forecast_graph/artifacts/traces'
    delta_root=ARTIFACTS/'deltas' if role=='confirmation' else NAV/'deltas'
    trace_path=trace_root/(key+'.npz');delta_path=delta_root/(key+'.npz')
    with np.load(trace_path,allow_pickle=False) as a:tx=a['tx'];rx=a['rx']
    with np.load(delta_path,allow_pickle=False) as a:tx=extend_trace(tx,a['delta'])
    city,_=public_context(r);city.radio.enable_cache();local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns);trace=ReceivedTrace(tx,rx,n,ns)
    f=build(local,trace,values['ends'],city)
    values.update(f,n=np.full(len(values['ends']),n,np.uint8),ns=np.full(len(values['ends']),ns,np.uint8),observer=np.full(len(values['ends']),ns,np.uint8))
    temp=path.with_suffix('.working.npz');np.savez_compressed(temp,**values);temp.replace(path)
    item=dict(key=key,role=role,region=r['region'],source_record=r,path=str(path),sha256=sha(path),queries=len(values['ends']),
        uniform_queries=int(values['uniform'].sum()),train_event_bags=len(values['bags']),source_pins=pins,parent_path=parent['path'],parent_sha256=parent['sha256'],
        trace_path=str(trace_path),trace_sha256=sha(trace_path),delta_path=str(delta_path),delta_sha256=sha(delta_path),created_at=now(),bytes=path.stat().st_size,wall_seconds=time.monotonic()-began)
    write(meta,item);return item

def main():
    verify();parents=read(PARENT/'data_manifest.json')['records'];done={};status('temporal_graph_data',complete=0,total=len(parents))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(prepare_one,x):i for i,x in enumerate(parents)}
        for f in as_completed(jobs):
            done[jobs[f]]=f.result();write(ARTIFACTS/'data_manifest.json',dict(records=[done[i] for i in sorted(done)],created_at=now()))
            status('temporal_graph_data',complete=len(done),total=len(parents));print(len(done),'/',len(parents),flush=True)
    write(ARTIFACTS/'data_status.json',dict(state='completed',complete=len(done),created_at=now()))

if __name__=='__main__':main()
