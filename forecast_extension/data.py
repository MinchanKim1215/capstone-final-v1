"""Fresh RF caches and sparse public-plan features, isolated from V8."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import multiprocessing as mp
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,digest,now,OLD_DEVELOPMENT,SEED,verify_protocol,maps
from forecast_rebuild.dataset import Episode
from forecast_rebuild.targets import labels,past_support
from forecast_rebuild.sensors import simulate,VERSION as SENSOR_VERSION
from forecast_rebuild.planning import profile,public_context


def compile_record(record):
    folder=ARTIFACTS/'cache'/record['key'];manifest=folder/'manifest.json'
    if manifest.exists():
        item=read(manifest)
        if item['source_sha256']!=record['array_sha256'] or item['sensor_version']!=SENSOR_VERSION:raise ValueError('Cached source differs')
        return dict(record,cache_path=str(folder),frames=item['frames'])
    if sha(record['array_path'])!=record['array_sha256']:raise ValueError('Fresh source changed')
    with np.load(record['array_path'],allow_pickle=False) as raw:
        values=raw['rf_values'];masks=raw['rf_masks'];intents=raw['rf_intents'];links=raw['links'];ends=raw['aux_endpoints'];geometry=raw['aux_geometry'];positions=raw['positions']
    take=ends>=40;ends=ends[take];geometry=geometry[take];truth=labels(links,ends,record['n_search'])
    support=past_support(values,masks,ends,record['n_search']);energy,sensor=simulate(record,values,links,positions)
    arrays=dict(values=values,masks=masks,intents=intents,links=links,ends=ends,geometry=geometry,
        targets=truth['targets'].astype(np.uint8),valid=truth['valid'],currently_up=truth['currently_up'],
        next_bin=truth['next_bin'].astype(np.uint8),legacy_support=support['legacy'],available=support['available'],energy=energy.astype(np.float16))
    folder.mkdir(parents=True,exist_ok=True)
    for name,value in arrays.items():np.save(folder/(name+'.npy'),value,allow_pickle=False)
    write(manifest,dict(source_sha256=record['array_sha256'],sensor_version=SENSOR_VERSION,sensor=sensor,frames=len(values),created_at=now()))
    return dict(record,cache_path=str(folder),frames=len(values))


def plan_record(record,rows=None):
    folder=Path(record['cache_path']);path=folder/'planning_manifest.json';expected=None if rows is None else digest(np.asarray(rows).tolist())
    if path.exists():
        value=read(path)
        if value['source_sha256']!=record['array_sha256'] or value['rows_sha256']!=expected:raise ValueError('Planning request changed')
        return record
    episode=Episode(record);city,nominal=public_context(record)
    out=profile(city,nominal,episode.values,episode.masks,episode.intents,episode.ends,record['n_search'],only_rows=rows)
    np.save(folder/'planning.npy',out,allow_pickle=False)
    write(path,dict(source_sha256=record['array_sha256'],rows_sha256=expected,planning_sha256=sha(folder/'planning.npy'),
                    public_plan_only=True,actual_future_positions_used=False,created_at=now()))
    return record


def pool_map(function,arguments,workers,stage):
    result={};status=ARTIFACTS/'preparation_status.json'
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(function,*args):i for i,args in enumerate(arguments)}
        for future in as_completed(pending):
            i=pending[future];result[i]=future.result()
            if len(result)%8==0 or len(result)==len(arguments):
                row=dict(state='running',stage=stage,complete=len(result),total=len(arguments),updated_at=now())
                write(status,row);print(row,flush=True)
    return [result[i] for i in range(len(arguments))]


def training_layout(record):
    regions=sorted((k for k,m in maps().items() if m['joint_split']=='train'),key=lambda k:digest(['augmentation',SEED,k]))[:84]
    eligible=Episode(record).eligible();rng=np.random.default_rng(int(digest(['v9-rows',SEED,record['key']])[:16],16))
    indices=eligible[rng.choice(len(eligible),min(2400,len(eligible)),replace=False)]
    variants=[];folder=ARTIFACTS/'row_indices';folder.mkdir(exist_ok=True)
    if record['region'] in regions:
        i=regions.index(record['region']);half=len(indices)//2
        setting=dict(gps_axis_sigma_m=(1,2,4)[i%3] if i<63 else 0,
                     energy_mode=('clean','noise','missing')[(i//3)%3] if i<63 else ('noise','missing')[i%2])
        key=digest(['augmentation',record['key'],setting])[:20]
        variant=dict(record,key=key,source_flight_key=record['key'],source_kind='augmented_v9',belief_profile=setting,
                     cache_path=str(ARTIFACTS/'cache'/key),clean_record_key=record['key'])
        variant_rows=indices[half:];indices=indices[:half]
        np.save(folder/(key+'.npy'),variant_rows,allow_pickle=False)
        variants.append((record,variant,variant_rows))
    np.save(folder/(record['key']+'.npy'),indices,allow_pickle=False)
    return indices,variants


def prepare(confirmation=False):
    verify_protocol();source=read(ARTIFACTS/'collection_manifest.json')['rows']
    if len(source)!=377:raise ValueError('Fresh collection is not complete')
    if confirmation:
        from .evaluate import verify_freeze
        verify_freeze();source=[r for r in source if r['role']=='confirmation']
        marker=ARTIFACTS/'confirmation_access.json'
        if not marker.exists():write(marker,dict(opened_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),scope='Fresh flights on known confirmation geographies only'))
        records=pool_map(compile_record,[(r,) for r in source],3,'confirmation_RF_cache')
        records=pool_map(plan_record,[(r,None) for r in records],2,'confirmation_public_plans')
        base=read(ARTIFACTS/'dataset.json');base['rows']=[r for r in base['rows'] if r['role']!='confirmation']+records
        base['confirmation_opened']=True;write(ARTIFACTS/'dataset_with_confirmation.json',base);return
    source=[r for r in source if r['role']!='confirmation']
    records=pool_map(compile_record,[(r,) for r in source],3,'fresh_RF_cache')
    train=[r for r in records if r['role']=='train'];augmented=[];row_indices={}
    for record in train:
        indices,variants=training_layout(record);row_indices[record['key']]=indices;augmented.extend(variants)
    records=pool_map(plan_record,[(r,row_indices[r['key']] if r['role']=='train' else None) for r in records],2,'fresh_public_plans')
    from .belief import create
    variants=pool_map(create,augmented,2,'training_observation_perturbations')
    old=read(OLD_DEVELOPMENT)['rows'];base=[r for r in old if r['role'] in ('train','selection')]
    rows=base+records+variants
    write(ARTIFACTS/'dataset.json',dict(rows=rows,created_at=now(),confirmation_opened=False,
        scope='Old training and selection plus fresh V9 train/selection/calibration. No old confirmation. Fresh confirmation withheld.',
        counts={k:sum(r['role']==k for r in rows) for k in ('train','selection','calibration','confirmation')}))
    write(ARTIFACTS/'preparation_status.json',dict(state='completed',stage='fresh_data_ready',updated_at=now(),fresh_flights=len(records),augmented_training_variants=len(variants)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--confirmation',action='store_true');a=p.parse_args();prepare(a.confirmation)
