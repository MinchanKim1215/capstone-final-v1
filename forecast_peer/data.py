"""Direct observed histories, inherited labels; no confirmation prediction reads."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now,verify,status
from .features import steps
from forecast_graph.episodes import physical

def prepare_one(parent):
    verify();key=parent['key'];r=parent['source_record'];path=ARTIFACTS/'data'/(key+'.npz');path.parent.mkdir(exist_ok=True);meta=path.with_suffix('.json')
    pins={k:sha(ROOT/k) for k in ('data.py','features.py')}
    if meta.exists():
        item=read(meta)
        if item['source_pins']!=pins or item['parent_sha256']!=parent['sha256'] or sha(path)!=item['sha256']:raise ValueError('Peer cache changed')
        return item
    if shutil.disk_usage(ARTIFACTS).free<18*1024**3:raise RuntimeError('Preserve18GiB disk reserve')
    if sha(parent['path'])!=parent['sha256']:raise ValueError('Inherited data changed')
    with np.load(parent['path']) as a:values={k:a[k] for k in ('ends','query','uniform','bags','y','up','support','links')}
    ep=physical(r);n=ep.values.shape[2];ns=r['n_search']
    values['peer_history']=steps(ep.values[1::2,0],ep.masks[1::2,0],ep.intents[1::2,0],ep.energy[1::2,0],n,ns)
    values.update(n=np.full(len(values['ends']),n,np.uint8),ns=np.full(len(values['ends']),ns,np.uint8))
    temporary=path.with_suffix('.working.npz');np.savez_compressed(temporary,**values);temporary.replace(path)
    item=dict(key=key,role=parent['role'],region=r['region'],source_record=r,path=str(path),sha256=sha(path),parent_path=parent['path'],parent_sha256=parent['sha256'],
        queries=len(values['ends']),uniform_queries=int(values['uniform'].sum()),train_event_bags=len(values['bags']),source_pins=pins,created_at=now(),bytes=path.stat().st_size)
    write(meta,item);return item

def inputs(item):
    if sha(item['parent_path'])!=item['parent_sha256']:raise ValueError('Frozen graph and baseline context changed')
    with np.load(item['parent_path']) as a:result={k:a[k] for k in ('embedding','logits','history','ends')}
    with np.load(item['path']) as a:result.update({k:a[k] for k in ('peer_history','n','ns')})
    return result

def main():
    records=read(PARENT/'data_manifest.json')['records'];done={};status('direct_peer_history_data',complete=0,total=len(records))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(prepare_one,i):k for k,i in enumerate(records)}
        for f in as_completed(jobs):
            done[jobs[f]]=f.result();write(ARTIFACTS/'data_manifest.json',dict(records=[done[k] for k in sorted(done)],created_at=now()));status('direct_peer_history_data',complete=len(done),total=len(records));print(len(done),'/',len(records),flush=True)
    write(ARTIFACTS/'data_status.json',dict(state='completed',complete=len(done),created_at=now()))

if __name__=='__main__':main()
