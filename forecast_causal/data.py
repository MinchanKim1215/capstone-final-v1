"""Matched existing queries; diagnostic arrays never copied into model inputs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
from . import ROOT,ARTIFACTS,PARENT,WORKSPACE
from .contracts import read,write,sha,now,status,milestone,verify
from .state import from_replay
from .features import summarize,tree_features
from .physics import Geometry
from .wire import SLOTS
from forecast_service.features import build as old_tree
from forecast_rebuild.planning import public_context
from forecast_graph.episodes import physical
from forecast_graph.features import LocalHistory,ReceivedTrace
from forecast_recall.inference import predict as old_predict
from forecast_temporal.inference import parent_entry

INPUTS=('base','network','physics','network_history','ends','n','ns')

def prepare_one(record):
    verify();began=time.monotonic();parent=record['parent_item'];r=record['source_record'];key=record['key'];folder=ARTIFACTS/'data';folder.mkdir(exist_ok=True);path=folder/(key+'.npz');meta=path.with_suffix('.json')
    pins={name:sha(ROOT/name) for name in ('data.py','features.py','state.py','physics.py','wire.py')}
    contract=dict(replay_sha256=record['sha256'],parent_sha256=parent['sha256'],uncertainty_sha256=sha(ARTIFACTS/'path_uncertainty.json'),source_pins=pins)
    if meta.exists():
        item=read(meta)
        if item['contract']!=contract or sha(path)!=item['sha256']:raise ValueError('Causal input cache changed')
        return item
    if shutil.disk_usage(ARTIFACTS).free<18*1024**3:raise RuntimeError('Preserve18GiB disk reserve')
    ep=physical(r);n=ep.values.shape[2];ns=r['n_search'];network,known=from_replay(record['path']);history=summarize(network,known,n,ns)
    with np.load(parent['path']) as a:
        values={k:a[k] for k in ('ends','query','uniform','bags','y','up','support','links')};context={k:a[k] for k in ('embedding','logits','history','ends')}
    baseline=old_predict(parent_entry()['candidate']['model'],context,device='cpu');ends=values['ends'];q=values['query'];city,_=public_context(r);city.radio.enable_cache()
    with np.load(record['path']) as a:tx=a['tx'].copy();rx=a['rx']
    tx[...,SLOTS]=0;tx[...,127]=0;trace=ReceivedTrace(tx,rx,n,ns);local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns);geometry=Geometry(city)
    bp=[];ph=[]
    for start in range(0,len(q),128):
        batch=q[start:start+128];bp.append(old_tree(ep,np.c_[batch,np.zeros(len(batch),int)],ns,city));ph.append(geometry.build(local,trace,ends[start:start+128]))
    values.update(base=np.c_[np.concatenate(bp),baseline[:,[2,5]]].astype(np.float32),network=tree_features(history,ends),physics=np.concatenate(ph),
        network_history=history.astype(np.float16),n=np.full(len(q),n,np.uint8),ns=np.full(len(q),ns,np.uint8),baseline=baseline)
    temp=path.with_suffix('.working.npz');np.savez_compressed(temp,**values);temp.replace(path)
    item=dict(key=key,role=record['role'],region=record['region'],source_record=r,path=str(path),sha256=sha(path),parent_path=parent['path'],parent_sha256=parent['sha256'],
        replay_path=record['path'],replay_sha256=record['sha256'],queries=len(q),uniform_queries=int(values['uniform'].sum()),event_bags=len(values['bags']),contract=contract,
        created_at=now(),wall_seconds=time.monotonic()-began,bytes=path.stat().st_size)
    write(meta,item);return item

def main():
    from .path_uncertainty import main as fit
    fit();records=read(ARTIFACTS/'all_replay_manifest.json')['records'];done={};status('stage2_causal_feature_cache',complete=0,total=len(records),next_action='관측·수신 시각과 기존 입력 대조를 검사한 뒤 원인별 대조 학습을 진행한다.')
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(prepare_one,r):i for i,r in enumerate(records)}
        for f in as_completed(jobs):
            done[jobs[f]]=f.result();write(ARTIFACTS/'data_manifest.json',dict(records=[done[k] for k in sorted(done)],created_at=now()));status('stage2_causal_feature_cache',complete=len(done),total=len(records));print(len(done),'/',len(records),flush=True)
    write(ARTIFACTS/'data_status.json',dict(state='completed',complete=len(done),created_at=now()))

if __name__=='__main__':main()
