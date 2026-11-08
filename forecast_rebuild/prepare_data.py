"""Convert compressed read-only archives to indexed local prediction caches."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import os
import time
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,now
from .inventory import build
from .targets import labels,past_support
from .sensors import simulate,VERSION as SENSOR_VERSION

VERSION='mmap_prediction_cache_v1'


def compile_one(record):
    folder=ARTIFACTS/'cache'/(record['key']+'_s2');manifest=folder/'manifest.json'
    if manifest.exists():
        saved=read(manifest)
        if saved['source_sha256']!=record['array_sha256'] or saved['version']!=VERSION or saved['sensor_version']!=SENSOR_VERSION:
            raise ValueError('Existing cache belongs to a different source/measurement version')
        return dict(record,cache_path=str(folder),frames=saved['frames'],endpoints=saved['endpoints'])
    source=Path(record['array_path'])
    if sha(source)!=record['array_sha256']:raise ValueError('Original RF archive changed')
    with np.load(source,allow_pickle=False) as z:
        values=z['rf_values'];masks=z['rf_masks'];intents=z['rf_intents'];links=z['links']
        if record['source_kind']=='bootstrap':
            if sha(record['geometry_path'])!=record['geometry_sha256']:raise ValueError('Historical causal geometry changed')
            with np.load(record['geometry_path'],allow_pickle=False) as g:
                ends=g['ends'];geometry=g['geometry']
            positions=None
        else:
            ends=z['aux_endpoints'];geometry=z['aux_geometry'];positions=z['positions']
    take=ends>=40;ends=ends[take];geometry=np.asarray(geometry[take],np.float32)
    truth=labels(links,ends,record['n_search'])
    support=past_support(values,masks,ends,record['n_search'])
    if record['source_kind']=='bootstrap':
        energy=np.full((*values.shape[:3],6),np.nan,np.float32);energy[...,3:5]=0
        sensor=dict(enabled=False,reason='No validated historical measurement replay')
    else:energy,sensor=simulate(record,values,links,positions)
    arrays=dict(values=values,masks=masks,intents=intents,links=links,ends=ends,geometry=geometry,
                targets=truth['targets'].astype(np.uint8),valid=truth['valid'],currently_up=truth['currently_up'],
                next_bin=truth['next_bin'].astype(np.uint8),legacy_support=support['legacy'],available=support['available'],
                energy=energy.astype(np.float16))
    folder.mkdir(parents=True,exist_ok=True)
    sizes={}
    for key,value in arrays.items():
        path=folder/(key+'.npy');temporary=folder/(key+'.working.npy')
        np.save(temporary,value,allow_pickle=False);os.replace(temporary,path);sizes[key]=path.stat().st_size
    write(manifest,dict(version=VERSION,sensor_version=SENSOR_VERSION,source_sha256=record['array_sha256'],
                         region=record['region'],role=record['role'],frames=len(values),endpoints=len(ends),
                         sensor=sensor,sizes=sizes,created_at=now()))
    return dict(record,cache_path=str(folder),frames=len(values),endpoints=len(ends))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--include-confirmation',action='store_true');args=parser.parse_args()
    inventory=build();rows=inventory['rows']
    if not args.include_confirmation:rows=[r for r in rows if r['role']!='confirmation']
    else:
        from .calibration import verify_freeze
        verify_freeze()
        marker=ARTIFACTS/'confirmation_access.json'
        if not marker.exists():write(marker,dict(opened_at=now(),freeze_sha256=sha(ARTIFACTS/'model_alarm_freeze.json'),
                                                scope='Development confirmation only; original 32 final-test maps remain sealed'))
    results={};started=time.monotonic();ctx=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx) as pool:
        pending={pool.submit(compile_one,row):row['key'] for row in rows}
        for future in as_completed(pending):
            key=pending[future];results[key]=future.result()
            if len(results)%8==0 or len(results)==len(rows):
                state=dict(state='preparing',complete=len(results),total=len(rows),wall_seconds=round(time.monotonic()-started,1),updated_at=now())
                write(ARTIFACTS/'preparation_status.json',state);print(json.dumps(state),flush=True)
    ordered=[results[r['key']] for r in rows]
    write(ARTIFACTS/'dataset.json',dict(version=VERSION,rows=ordered,created_at=now(),
        counts={k:sum(r['role']==k for r in ordered) for k in ('train','selection','calibration','confirmation')}))
    write(ARTIFACTS/'preparation_status.json',dict(state='completed',complete=len(rows),total=len(rows),updated_at=now()))


if __name__=='__main__':main()
