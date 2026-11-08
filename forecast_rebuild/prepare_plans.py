"""Alternative public-plan/constant-velocity hypotheses, never actual future positions."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import os
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,now,digest
from .dataset import Episode
from .planning import profile,public_context,VERSION


def one(record,only_rows=None):
    folder=Path(record['cache_path']);manifest=folder/'planning_manifest.json';path=folder/'planning.npy'
    if manifest.exists():
        saved=read(manifest)
        if saved['version']!=VERSION or saved['source_sha256']!=record['array_sha256']:raise ValueError('Planning cache version mismatch')
        expected=None if only_rows is None else digest(np.asarray(only_rows).tolist())
        if saved.get('requested_rows_sha256')!=expected:raise ValueError('Sparse planning sample contract changed')
        return dict(region=record['region'],reused=True)
    episode=Episode(record);city,nominal=public_context(record)
    out=profile(city,nominal,episode.values,episode.masks,episode.intents,episode.ends,record['n_search'],only_rows=only_rows)
    temporary=folder/'planning.working.npy';np.save(temporary,out,allow_pickle=False);os.replace(temporary,path)
    write(manifest,dict(version=VERSION,source_sha256=record['array_sha256'],planning_sha256=sha(path),
                        public_plan_only=True,actual_future_positions_used=False,
                        requested_rows_sha256=None if only_rows is None else digest(np.asarray(only_rows).tolist()),
                        sparse_training_cache=only_rows is not None,created_at=now()))
    return dict(region=record['region'],reused=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--confirmation',action='store_true');a=p.parse_args()
    rows=read(ARTIFACTS/'dataset.json')['rows']
    if a.confirmation:
        from .calibration import verify_freeze
        verify_freeze()
        rows=[r for r in rows if r['role']=='confirmation']
    else:rows=[r for r in rows if r['role']!='confirmation']
    selected={}
    if not a.confirmation:
        samples=ARTIFACTS/'samples/expanded_core'
        training_index=np.load(samples/'train_index.npy',mmap_mode='r',allow_pickle=False)
        sample_records=read(samples/'manifest.json')['contract']['records']
        for e in np.unique(training_index[:,0]):
            selected[sample_records[int(e)]['key']]=np.asarray(training_index[training_index[:,0]==e,1:])
    count=0
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        pending=[pool.submit(one,r,selected.get(r['key']) if r['role']=='train' else None) for r in rows]
        for future in as_completed(pending):
            future.result();count+=1
            if count%8==0 or count==len(rows):
                status=dict(state='running',complete=count,total=len(rows),updated_at=now())
                write(ARTIFACTS/'planning_status.json',status);print(json.dumps(status),flush=True)
    write(ARTIFACTS/'planning_status.json',dict(state='completed',complete=count,total=len(rows),updated_at=now()))


if __name__=='__main__':main()
