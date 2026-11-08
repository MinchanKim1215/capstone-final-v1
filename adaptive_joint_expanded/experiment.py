"""Frozen provenance, checkpoints, deterministic case schedules, isolated workers."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import asdict
import json
import multiprocessing as mp
import os
import sys
import time
import numpy as np
import torch
from . import ROOT, LEGACY, ARTIFACTS, VERSION
from .contracts import (DEFAULT, PILOT, PLAN, DEPLOYMENTS, atomic_json, atomic_torch,
                         config_hash, digest, entry_for, legacy_hashes, plan, read_json,
                         rng_state, restore_rng, seed_all, sha, source_hashes)
from .contracts import PUBLIC_CONTEXT_DIM
from .model import Controller, optimizer
from .initialization import parent_initialization


def runtime_contract():
    import numba,scipy,sklearn,shapely
    return dict(python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,
                numpy=np.__version__,numba=numba.__version__,scipy=scipy.__version__,
                sklearn=sklearn.__version__,shapely=shapely.__version__,
                environment={k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES',
                    'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})


def prepare(config=None, force=False):
    config=dict(DEFAULT,**(config or {}))
    config.setdefault('prediction_mode','joint')
    validate_config(config)
    if not (ARTIFACTS/'legacy_before.json').exists():
        atomic_json(ARTIFACTS/'legacy_before.json',legacy_hashes())
    if read_json(ARTIFACTS/'legacy_before.json')!=legacy_hashes():
        raise ValueError('Original source/checkpoint preservation contract failed')
    # Merely listing the original split does not open test geometry/outcomes.
    p=plan()
    from .map_features import descriptors
    for m in p['maps']:
        if m['joint_split'] in ('train','validation'):
            descriptors(m)
    protocol=dict(version=VERSION,parent_initialization=parent_initialization(),runtime=runtime_contract(),config=config,config_sha256=config_hash(config),
        source_sha256=source_hashes(),legacy_sha256=legacy_hashes(),plan_sha256=sha(PLAN),
        warm_predictor=parent_initialization() or dict(status='pending_parent_completion'),
        reference_pilot=dict(path=str(PILOT),sha256=sha(PILOT),step=5500,scope='Replay reference only'),
        deployments=[asdict(d) for d in DEPLOYMENTS],
        train_regions=[m['key'] for m in p['maps'] if m['joint_split']=='train'],
        validation_regions=[m['key'] for m in p['maps'] if m['joint_split']=='validation'],
        development_geography=development_splits(),
        test_regions_sealed=[m['key'] for m in p['maps'] if m['joint_split']=='test'],
        public_selector_inputs=['building_count','tall_building_fraction','no_fly_fraction',
                               'nonimputed_height_fraction','imputed_height_fraction','direction_east_west',
                               'direction_north_south','available_fleet','required_search_fleet',
                               'height_mean_std_max_scaled','nofly_grid_8x8','building_footprint_grid_8x8'],
        public_selector_dimension=PUBLIC_CONTEXT_DIM,
        hidden_inputs_forbidden=['seed','jammer_state','future_channel','future_position',
                                 'outcome','split','map_identifier'],
        targets='Next >=1s outage onset: 16 half-second bins in [0,8), plus no onset; strict half-open summaries',
        prediction_s=1,action_s=3,radio_s=.5,
        physical_teacher='v5 communications-priority probability profile, generalized search/relay counts',
        selector_action='Before takeoff; fleet size cannot change in flight',
        selector_reward='Undiscounted episode cost / 2400, with explicit fleet-time cost; not a reported performance score',
        traffic='30 packets/s total, unchanged legacy queue/routing; actual N beacon slots reserved each second',
        paired_environment='Same absolute jammer schedule from 8-agent nominal anchor; shared first-eight shadow streams; packet RNG follows actions',
        final_test_opened=False)
    path=ARTIFACTS/'protocol.json'
    if path.exists() and read_json(path)!=protocol:
        if not force or (ARTIFACTS/'runs').exists():
            raise ValueError('Protocol differs; archive this study or prepare before any training run exists')
        atomic_json(ARTIFACTS/'readiness.json',dict(ready=False,reason='Protocol refreshed; rerun preflight',protocol_sha256=digest(protocol)))
    atomic_json(path,protocol)
    return protocol


def validate_config(c):
    for k in ('seed','updates','batch_episodes','epochs','minibatch','workers','validation_cases','available_drones','minimum_search'):
        if not isinstance(c[k],int) or c[k]<1:raise ValueError('Positive integer required: '+k)
    if c['batch_episodes']<4 or c['batch_episodes']%4:
        raise ValueError('Batch episodes must be a multiple of four for deployment balance')
    for k in ('actor_lr','encoder_lr','timing_lr','critic_lr','selector_lr','prediction_coef','target_kl'):
        if not np.isfinite(c[k]) or c[k]<=0:raise ValueError('Finite positive parameter required: '+k)
    for k in ('validation_every','patience','warmup_updates','keep_rollouts'):
        if not isinstance(c[k],int) or c[k]<0:raise ValueError('Nonnegative integer required: '+k)
    if c['mode'] not in ('adaptive','balanced'):raise ValueError('Unknown deployment training mode')
    if c.get('prediction_mode','joint') not in ('joint','fixed','none'):raise ValueError('Unknown predictor mode')
    if not 0<c['clip']<1 or not 0<c['gamma']<=1 or not 0<c['gae_lambda']<=1:
        raise ValueError('Invalid PPO clipping or return parameters')
    if not np.isfinite(c['fleet_cost_per_step']) or c['fleet_cost_per_step']<0:
        raise ValueError('Explicit fleet cost must be finite and nonnegative')
    if not 0<c['minimum_pdr']<=1 or not 1<=c['maximum_duration_ratio']<10:
        raise ValueError('Invalid explicit mission feasibility goals')
    from .contracts import deployment_mask
    feasible=deployment_mask(c['available_drones'],c['minimum_search'])
    if c['batch_episodes']%int(feasible.sum()):
        raise ValueError('Batch episodes must be divisible by the number of budget-feasible deployments')


def assert_protocol(protocol,allow_final_test=False):
    if protocol.get('parent_initialization')!=parent_initialization():raise ValueError('Parent initialization binding changed; prepare again before training')
    if protocol.get('runtime')!=runtime_contract():raise ValueError('Python/ML runtime differs from the frozen study')
    if protocol['version']!=VERSION or protocol['source_sha256']!=source_hashes():
        raise ValueError('New study source differs from its frozen protocol')
    if protocol['legacy_sha256']!=legacy_hashes() or protocol['plan_sha256']!=sha(PLAN):
        raise ValueError('Read-only legacy dependency or data split changed')
    if not allow_final_test and (protocol['final_test_opened'] or (ARTIFACTS/'test_access.json').exists()):
        raise ValueError('Final test is already open; development cannot continue in this study')


def save_checkpoint(folder, model, opt, config, step, protocol, history, purpose='training'):
    folder=Path(folder)
    item=dict(format=VERSION,purpose=purpose,model=model.state_dict(),optimizer=opt.state_dict(),config=config,
              config_sha256=config_hash(config),step=step,protocol_sha256=digest(protocol),
              rng=rng_state(),history=history)
    path=folder/'checkpoints'/f's{step:07d}.pt'
    atomic_torch(path,item)
    atomic_json(folder/'latest.json',dict(path=str(path.resolve()),sha256=sha(path),step=step))
    return path


def load_checkpoint(path, protocol, device='cpu', restore=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('format')!=VERSION or ck['protocol_sha256']!=digest(protocol):
        raise ValueError('Checkpoint architecture/provenance mismatch')
    if ck['config_sha256']!=config_hash(ck['config']):raise ValueError('Checkpoint configuration is corrupt')
    model=Controller(warm_start=False)
    model.load_state_dict(ck['model'],strict=True);model.to(device)
    opt=optimizer(model,ck['config']);opt.load_state_dict(ck['optimizer'])
    for state in opt.state.values():
        for k,v in state.items():
            if torch.is_tensor(v):state[k]=v.cpu() if k=='step' else v.to(device)
    if restore:restore_rng(ck['rng'])
    return model,opt,ck


def training_specs(seed, offset, count):
    keys=sorted(m['key'] for m in plan()['maps'] if m['joint_split']=='train')
    order=np.random.default_rng(seed).permutation(keys).tolist()
    out=[]
    for i in range(offset,offset+count):
        slot=i%len(order);cycle=i//len(order)
        token=digest(['adaptive-v7-expanded100-train',seed,i])
        out.append(dict(region=order[slot],direction=('east_west','north_south')[(slot+cycle)%2],
                        kind=('clear','intermittent','continuous')[(slot//2+cycle//2)%3],
                        seed=int(token[:8],16)%(2**31-3000004)))
    return out


def development_splits():
    groups={}
    for m in plan()['maps']:
        if m['joint_split']=='validation':groups.setdefault(m['joint_spatial_group'],[]).append(m['key'])
    folds=[[],[],[]]
    for group,keys in sorted(groups.items(),key=lambda v:(-len(v[1]),digest(['fold-v6',v[0]]))):
        fold=min(range(3),key=lambda i:len(folds[i]))
        folds[fold].extend(keys)
    return dict(zip(('selection','calibration','confirmation'),[sorted(f) for f in folds]))


def validation_specs(count,scope='selection'):
    if scope not in ('selection','calibration','confirmation'):raise ValueError('Unknown development scope')
    allowed=development_splits()[scope]
    specs=[s for s in plan()['validation'] if s['region'] in allowed]
    if count<=0 or count>=len(specs):return specs
    # Mix original and added geography in every small validation batch.
    groups={}
    for s in specs:groups.setdefault(s['region'],[]).append(s)
    metadata={m['key']:m for m in plan()['maps']}
    original=sorted(k for k in groups if metadata[k].get('source_dataset','legacy_v6')=='legacy_v6')
    by_province={}
    for k in groups:
        if k not in original:by_province.setdefault(metadata[k].get('province','unknown'),[]).append(k)
    added=[]
    for j in range(max([len(v) for v in by_province.values()]+[0])):
        for province in sorted(by_province):
            keys=sorted(by_province[province])
            if j<len(keys):added.append(keys[j])
    order=[]
    for j in range(max(len(original),len(added))):
        if j<len(added):order.append(added[j])
        if j<len(original):order.append(original[j])
    ordered=[]
    for j in range(max(map(len,groups.values()))):
        for k in order:
            if j<len(groups[k]):
                rotation=order.index(k)%len(groups[k])
                ordered.append(groups[k][(j+rotation)%len(groups[k])])
    return ordered[:count]


def prepare_assets(specs):
    """Parent-only preparation avoids concurrent writes to shared map caches."""
    from swarm_sim.config import Scenario
    from usable.collect import prepare_city_without_unused_radio_cache
    from .baseline import ensure_nominal
    records=[]
    for region,direction in sorted(set((s['region'],s['direction']) for s in specs)):
        entry=entry_for(region)
        scenario=Scenario(duration_s=7200,motion_dt_s=3,direction=direction).validate()
        city=prepare_city_without_unused_radio_cache(LEGACY/entry['path'],scenario,ARTIFACTS/'cache/city')
        for d in DEPLOYMENTS:
            path=ensure_nominal(city,d,scenario)
            item=read_json(path.with_suffix('.json'))
            records.append(dict(region=region,direction=direction,deployment=d.key,path=str(path.resolve()),
                                sha256=item['sha256'],completed=item['completed'],duration_s=item['duration_s'],
                                minimum_separation_m=item['minimum_separation_m']))
    return records


def _asset_worker(region):
    return prepare_assets([dict(region=region,direction=d) for d in ('east_west','north_south')])


def build_assets(protocol,workers=4):
    assert_protocol(protocol)
    regions=sorted(m['key'] for m in plan()['maps'] if m['joint_split'] in ('train','validation'))
    rows=[];errors=[]
    ctx=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as pool:
        pending={pool.submit(_asset_worker,k):k for k in regions}
        done=0
        for future in as_completed(pending):
            region=pending[future]
            try:rows.extend(future.result())
            except Exception as error:errors.append(dict(region=region,error=str(error)))
            done+=1
            atomic_json(ARTIFACTS/'assets_progress.json',dict(complete=done,total=len(regions),errors=errors))
            if done%10==0 or errors or done==len(regions):
                print(json.dumps(dict(stage='nominal_asset_preparation',complete=done,total=len(regions),errors=len(errors))),flush=True)
    result=dict(protocol_sha256=digest(protocol),ready=not errors and len(rows)==len(regions)*8,
                regions=len(regions),trajectories=len(rows),incomplete_nominal=sum(not r['completed'] for r in rows),
                errors=errors,rows=rows,test_geometry_opened=False,purpose='Static nominal planning only; no RF/learning performance results')
    atomic_json(ARTIFACTS/'assets_manifest.json',result)
    return result


_MODEL=None
_CHECKPOINT_SHA=None
_PROTOCOL_SHA=None
_PURPOSE=None
def _initialize(checkpoint, protocol):
    global _MODEL,_CHECKPOINT_SHA,_PROTOCOL_SHA,_PURPOSE
    assert_protocol(protocol,allow_final_test=bool(os.environ.get('ADAPTIVE_FINAL_TEST_FREEZE')))
    seed_all(1)
    _MODEL,_,ck=load_checkpoint(checkpoint,protocol,'cpu')
    _MODEL.eval()
    _CHECKPOINT_SHA=sha(checkpoint)
    _PROTOCOL_SHA=digest(protocol)
    _PURPOSE=ck.get('purpose','training')


def _worker(task):
    from .rollout import episode
    spec,config,forced,deterministic,max_steps,prefix=task
    request=dict(spec=spec,forced_deployment=forced,deterministic=deterministic,max_steps=max_steps,
                 prediction_mode=config.get('prediction_mode','joint'),
                 purpose=_PURPOSE,
                 checkpoint_sha256=_CHECKPOINT_SHA,protocol_sha256=_PROTOCOL_SHA)
    meta=Path(prefix).with_suffix('.json')
    if meta.exists():
        row=read_json(meta)
        if row.get('request')!=request or row['array_sha256']!=sha(meta.with_suffix('.npz')):
            raise ValueError('Existing committed rollout differs from this request')
        return row
    if forced=='legacy_v5':
        from .rollout import legacy_episode
        return legacy_episode(spec,config,max_steps,prefix,request)[1]
    return episode(_MODEL,spec,config,forced,deterministic,max_steps,
                   config.get('prediction_mode','joint'),prefix,request)[1]


def collect(checkpoint, protocol, specs, folder, forced=None, deterministic=False, max_steps=None,load_arrays=True):
    assert_protocol(protocol,allow_final_test=bool(os.environ.get('ADAPTIVE_FINAL_TEST_FREEZE')))
    config=protocol['config']
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    prepare_assets(specs)
    tasks=[]
    for i,spec in enumerate(specs):
        choice=forced[i] if isinstance(forced,list) else forced
        name=f'{i:04d}_{spec["region"]}_{choice or "adaptive"}'
        tasks.append((spec,config,choice,deterministic,max_steps,str(folder/name)))
    if config['workers']==1:
        _initialize(str(checkpoint),protocol)
        rows=[_worker(t) for t in tasks]
    else:
        ctx=mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=config['workers'],mp_context=ctx,
                                 initializer=_initialize,initargs=(str(checkpoint),protocol)) as pool:
            pending={pool.submit(_worker,t):i for i,t in enumerate(tasks)}
            indexed={}
            for future in as_completed(pending):
                i=pending[future];indexed[i]=future.result()
                print(json.dumps(dict(stage='collect',complete=len(indexed),total=len(tasks),
                    deployment=indexed[i]['deployment'],smoke_only=indexed[i]['smoke_only'])),flush=True)
            rows=[indexed[i] for i in range(len(tasks))]
    episodes=[]
    for task,row in zip(tasks,rows):
        prefix=Path(task[-1]);path=prefix.with_suffix('.npz')
        if sha(path)!=row['array_sha256']:raise ValueError('Rollout content changed')
        row['array_path']=str(path.resolve())
        if not load_arrays:
            episodes.append(row);continue
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        episodes.append((arrays,row))
    return episodes
