"""Flight-balanced supervised examples; truth is constructed outside feature APIs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import os
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,digest,now,verify,status,source_records,SEED,V10
from .features import LocalHistory,ReceivedTrace,build,THRESHOLD
from .codec import MAX_N
from forecast_rebuild.dataset import Episode
from forecast_rebuild.targets import labels
from forecast_rebuild.planning import public_context

FEATURE_KEYS=('node_local','node_shared','edge_local','edge_shared','physics','global')
TARGET_KEYS=('link_y','link_up','link_support','service_y','service_up','service_support','aux','n','ns','observer')


def plan():
    p=verify();value=dict(created_at=now(),protocol_sha256=sha(ARTIFACTS/'protocol.json'),
        queries_per_training_flight=p['per_flight_train_queries'],
        primary_physical_observer='First relay, matched against every baseline on the same flights and unchanged directed-link event rule',
        primary_service_observer='First relay, unchanged V10 target',
        training_observers='All relay observers; deterministic uniform queries',
        validation_queries='All eligible1Hz endpoints of first relay; no negative subsampling in final metrics',
        inference_scope='A single relay local/received-only view; not a fused omniscient ground station',
        note='Primary link reports use the matched first-relay subset, not previous all-relay aggregate numbers.')
    path=ARTIFACTS/'data_plan.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Data plan changed')
    else:write(path,value)
    return value


def queries(record,ep):
    good=np.flatnonzero(np.asarray(ep.valid).any((1,2)))
    if record['role']=='train':
        grid=np.stack(np.meshgrid(good,np.arange(ep.values.shape[1]),indexing='ij'),-1).reshape(-1,2)
        rng=np.random.default_rng(int(digest(['v11-query',SEED,record['key']])[:16],16))
        return grid[rng.choice(len(grid),min(plan()['queries_per_training_flight'],len(grid)),replace=False)]
    return np.c_[good,np.zeros(len(good),int)]


def target_arrays(record,ep,q,sinr):
    n=ep.values.shape[2];ns=record['n_search'];t,r=q.T;end=np.asarray(ep.ends)[t]
    with np.load(V10/'service_cache'/(record['key']+'.npz'),allow_pickle=False) as a:health=a['service_healthy']
    return target_values(ep,q,sinr,health,n,ns)


def target_values(ep,q,sinr,health,n,ns):
    t,r=q.T;end=np.asarray(ep.ends)[t];count=len(q)
    sy=np.zeros((len(health),2,2),bool);sy[:,1,0]=health
    service=labels(sy,ep.ends,1)
    ly=np.zeros((count,MAX_N,6),np.uint8);ly[:,:n]=ep.targets[t,r]
    up=np.zeros((count,MAX_N),bool);up[:,:n]=ep.currently_up[t,r]
    support=np.zeros_like(up);support[:,:n]=ep.legacy_support[t,r]
    future=np.asarray(sinr)[end[:,None]+np.arange(9)[None,:]*2]
    margin=np.minimum(future,future.transpose(0,1,3,2))-THRESHOLD
    aux=np.zeros((count,9,MAX_N,MAX_N),np.float16);aux[:,:,:n,:n]=np.clip(margin,-90,90)/30
    return dict(link_y=ly,link_up=up,link_support=support,service_y=service['targets'][t,0,0].astype(np.uint8),
        service_up=service['currently_up'][t,0,0],service_support=np.asarray(ep.legacy_support)[t,r].any(1),
        aux=aux,n=np.full(count,n,np.int16),ns=np.full(count,ns,np.int16),observer=(ns+r).astype(np.int16))


def build_one(record):
    folder=ARTIFACTS/'data';folder.mkdir(exist_ok=True);path=folder/(record['key']+'.npz');meta=path.with_suffix('.json')
    trace=ARTIFACTS/'traces'/(record['key']+'.npz')
    contract=dict(source_sha256=record['array_sha256'],trace_sha256=sha(trace),
        features_sha256=sha(ROOT/'features.py'),data_sha256=sha(ROOT/'data.py'),data_plan_sha256=sha(ARTIFACTS/'data_plan.json'))
    if meta.exists():
        old=read(meta)
        if old['contract']!=contract or sha(path)!=old['sha256']:raise ValueError('Data cache differs')
        return old
    if shutil.disk_usage(ARTIFACTS).free<18*1024**3:raise RuntimeError('Keep18GiB reserve; do not remove prior artifacts')
    ep=Episode(record);q=queries(record,ep);ends=np.asarray(ep.ends)[q[:,0]]
    with np.load(trace,allow_pickle=False) as a:tx=a['tx'];rx=a['rx'];sinr=a['sinr']
    received=ReceivedTrace(tx,rx,ep.values.shape[2],record['n_search'])
    local=LocalHistory(ep.values,ep.masks,ep.intents,ep.values.shape[2],record['n_search'])
    city,_=public_context(record);city.radio.enable_cache(200000)
    parts={key:[] for key in FEATURE_KEYS};started=time.monotonic()
    for start in range(0,len(q),128):
        features=build(local,received,np.c_[ends[start:start+128],q[start:start+128,1]],city)
        for key,value in features.items():parts[key].append(value)
    arrays={key:np.concatenate(value) for key,value in parts.items()}
    arrays.update(target_arrays(record,ep,q,sinr));arrays['query']=q.astype(np.int32)
    temporary=path.with_suffix('.working.npz');np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    value=dict(key=record['key'],role=record['role'],region=record['region'],path=str(path),sha256=sha(path),
        queries=len(q),contract=contract,n=local.n,ns=local.ns,created_at=now(),wall_seconds=time.monotonic()-started,
        bytes=path.stat().st_size,source_record=record)
    write(meta,value);return value


def main():
    plan()
    while not (ARTIFACTS/'trace_status.json').exists():
        error=ARTIFACTS/'trace.stderr.log'
        if error.exists() and 'Traceback (most recent call last)' in error.read_text(encoding='utf-8',errors='replace'):
            raise RuntimeError('Trace collection failed; see its preserved log')
        time.sleep(5)
    rows=source_records();results={};status('received_graph_and_physics_features',complete=0,total=len(rows))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(build_one,r):i for i,r in enumerate(rows)}
        for future in as_completed(pending):
            results[pending[future]]=future.result();status('received_graph_and_physics_features',complete=len(results),total=len(rows))
            write(ARTIFACTS/'data_manifest.json',dict(records=[results[i] for i in sorted(results)],created_at=now()))
            print(len(results),'/',len(rows),flush=True)
    write(ARTIFACTS/'data_status.json',dict(state='completed',total=len(rows),created_at=now()))


if __name__=='__main__':main()

