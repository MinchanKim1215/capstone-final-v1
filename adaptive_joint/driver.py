"""Long-training driver. Preflight never invokes this campaign."""
from pathlib import Path
import json
import re
import shutil
import time
import os
import numpy as np
import torch
from . import ARTIFACTS
from .contracts import (DEPLOYMENTS,atomic_json,config_hash,deployment_mask,digest,
                         lock,read_json,seed_all,sha)
from .experiment import (assert_protocol,collect,load_checkpoint,prepare,save_checkpoint,
                         training_specs,validation_specs)
from .learning import update
from .model import Controller,optimizer


def read_episodes(rows):
    for row in rows:
        path=Path(row['array_path'])
        if sha(path)!=row['array_sha256']:raise ValueError('Evaluation rollout hash changed')
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        yield a,row


def evaluate(checkpoint,protocol,folder,scope='selection',count=0,paired=False,explicit_specs=None,allow_final_test=False):
    from .metrics import summarize,paired_comparison,paired_interval
    assert_protocol(protocol,allow_final_test=allow_final_test)
    specs=validation_specs(count,scope) if explicit_specs is None else explicit_specs
    choices=[d.key for d,m in zip(DEPLOYMENTS,deployment_mask(protocol['config']['available_drones'],protocol['config']['minimum_search'])) if m]
    if paired:
        expanded=[s for s in specs for _ in choices]
        forced=choices*len(specs)
    else:expanded=specs;forced=None
    rows=collect(checkpoint,protocol,expanded,Path(folder)/'rollouts',forced,True,load_arrays=False)
    if not paired:
        result=summarize(read_episodes(rows))
    else:
        result=dict(scope=scope,by_deployment={},selector_vs_fixed={},oracle_information='Offline upper bound only, never an online input')
        for choice in choices:
            subset=[r for r in rows if r['deployment']==choice]
            result['by_deployment'][choice]=summarize(read_episodes(subset))
        model,_,_=load_checkpoint(checkpoint,protocol,'cpu');model.eval()
        from .contracts import entry_for,public_context
        selected=[];oracle=[];pairs=[]
        grouped=paired_comparison(rows)
        for spec,options in grouped.items():
            first=next(iter(options.values()))
            c=public_context(entry_for(first['region']),first['direction'],
                    protocol['config']['available_drones'],protocol['config']['minimum_search'])
            mask=deployment_mask(protocol['config']['available_drones'],protocol['config']['minimum_search'])
            with torch.no_grad():index=int(model.select(torch.tensor(c),torch.tensor(mask)).argmax())
            chosen=options[DEPLOYMENTS[index].key]
            selected.append(chosen)
            feasible=[r for r in options.values() if r['mission_completed'] and r['audit']['passed']]
            oracle.append(max(feasible or list(options.values()),key=lambda r:r['selector_return']))
        result['adaptive_selector']=summarize(read_episodes(selected))
        result['selected_case_metadata']=[str(Path(r['array_path']).with_suffix('.json')) for r in selected]
        result['offline_oracle']=summarize(read_episodes(oracle))
        for choice in choices:
            pairs=[(grouped[(r['region'],r['direction'],r['kind'],r['seed'])][choice],r) for r in selected]
            result['selector_vs_fixed'][choice]={k:paired_interval(pairs,k) for k in ['search_connected_mean','outage_total_s','duration_s','relay_travel_m']}
        if protocol['config']['available_drones']>=8 and protocol['config']['minimum_search']<=3:
            legacy=collect(checkpoint,protocol,specs,Path(folder)/'legacy_v5','legacy_v5',True,load_arrays=False)
            result['frozen_v5_best176']=summarize(read_episodes(legacy))
            result['selector_vs_frozen_v5']={k:paired_interval(list(zip(legacy,selected)),k)
                    for k in ['search_connected_mean','outage_total_s','duration_s','relay_travel_m']}
    result.update(checkpoint_sha256=sha(checkpoint),protocol_sha256=digest(protocol),development_scope=scope)
    if allow_final_test:
        sections=[result,*result.get('by_deployment',{}).values()]
        sections.extend(result[k] for k in ('adaptive_selector','offline_oracle','frozen_v5_best176') if k in result)
        for section in sections:section['scope']='Sealed test after model/alarm freeze; predetermined comparisons'
        for comparisons in list(result.get('selector_vs_fixed',{}).values())+[result.get('selector_vs_frozen_v5',{})]:
            for interval in comparisons.values():
                interval['warning']='Map-cluster bootstrap of a frozen predetermined comparison; no test-driven tuning'
    atomic_json(Path(folder)/'summary.json',result)
    return result


def train(name,protocol,updates=None,resume=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):raise ValueError('Invalid run name')
    assert_protocol(protocol)
    if not (ARTIFACTS/'readiness.json').exists():raise ValueError('Run preflight before long training')
    readiness=read_json(ARTIFACTS/'readiness.json')
    if not readiness.get('ready') or readiness['protocol_sha256']!=digest(protocol):
        raise ValueError('Readiness belongs to another source/configuration')
    c=dict(protocol['config']);c['updates']=updates if updates is not None else c['updates']
    from .experiment import validate_config
    validate_config(c)
    folder=ARTIFACTS/'runs'/name
    if folder.exists() and not resume:raise ValueError('Run exists; use --resume to continue it')
    folder.mkdir(parents=True,exist_ok=True)
    with lock(folder/'run.lock'):
        device=c['device']
        if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Configured CUDA is unavailable')
        if resume:
            pointer=read_json(folder/'latest.json');path=Path(pointer['path'])
            if sha(path)!=pointer['sha256']:raise ValueError('Committed checkpoint hash mismatch')
            model,opt,ck=load_checkpoint(path,protocol,device,True)
            if ck.get('purpose')!='training':raise ValueError('A preflight checkpoint is not a campaign resume point')
            if ck['config_sha256']!=config_hash(c):raise ValueError('Resume settings changed')
            done=ck['step'];history=ck['history']
            if c['updates']<done:raise ValueError('Target update budget is less than the committed step')
        else:
            seed_all(c['seed']);model=Controller().to(device);opt=optimizer(model,c);done=0;history=[]
            path=save_checkpoint(folder,model,opt,c,0,protocol,history)
        atomic_json(folder/'attempt.json',dict(token=os.environ.get('ADAPTIVE_RUN_TOKEN'),pid=os.getpid()))
        atomic_json(folder/'config.json',c)
        validation=read_json(folder/'validation_history.json') if (folder/'validation_history.json').exists() else []
        feasible_choices=[d.key for d,m in zip(DEPLOYMENTS,deployment_mask(c['available_drones'],c['minimum_search'])) if m]
        started=time.perf_counter()

        def validate(step,checkpoint):
            nonlocal validation
            result=evaluate(checkpoint,protocol,folder/'validation'/f'u{step:07d}',count=c['validation_cases'])
            means=result['means']
            rows=[read_json(p) for p in (folder/'validation'/f'u{step:07d}'/'rollouts').glob('*.json')]
            feasible=(result['completed']==result['cases'] and result['safety_passed'] and
                      all(r['pdr']>=c['minimum_pdr'] and (r['duration_ratio'] is None or r['duration_ratio']<=c['maximum_duration_ratio']) for r in rows))
            score=[int(feasible),means['search_connected_mean'],means['pdr'],-means['outage_total_s']]
            previous=[r for r in validation if r['feasible']]
            improved=feasible and (not previous or tuple(score)>max(tuple(r['score']) for r in previous))
            record=dict(step=step,score=score,feasible=feasible,improved=improved,checkpoint=str(checkpoint.resolve()),sha256=sha(checkpoint))
            validation=[r for r in validation if r['step']!=step]+[record]
            validation.sort(key=lambda r:r['step'])
            atomic_json(folder/'validation_history.json',validation)
            if improved:atomic_json(folder/'best.json',record)

        # Validation interrupted after a committed update is finished on resume.
        if done and c['validation_every'] and done%c['validation_every']==0 and not any(r['step']==done for r in validation):
            validate(done,path)
        state='completed'
        for step in range(done+1,c['updates']+1):
            assert_protocol(protocol)
            if validation and c['patience']:
                last=max((i for i,r in enumerate(validation) if r['improved']),default=-1)
                if len(validation)-last-1>=c['patience']:
                    state='early_stopped';break
            if shutil.disk_usage(ARTIFACTS).free<3*1024**3:raise OSError('Less than 3 GiB free for rollout/checkpoint transactions')
            balanced=c['mode']=='balanced' or step<=c['warmup_updates']
            if balanced:
                n=c['batch_episodes']//len(feasible_choices)
                cases=training_specs(c['seed'],(step-1)*n,n)
                specs=[s for s in cases for _ in feasible_choices]
                forced=feasible_choices*len(cases)
            else:
                warm_cases=c['warmup_updates']*(c['batch_episodes']//len(feasible_choices))
                offset=warm_cases+(step-c['warmup_updates']-1)*c['batch_episodes']
                specs=training_specs(c['seed'],offset,c['batch_episodes']);forced=None
            atomic_json(folder/'status.json',dict(state='running',stage='collect',update=step,target=c['updates']))
            episodes=collect(path,protocol,specs,folder/'rollouts'/f'u{step:07d}',forced)
            atomic_json(folder/'status.json',dict(state='running',stage='joint_ppo',update=step,target=c['updates']))
            model.to(device)
            stats=update(model,opt,episodes,c,step,device)
            stats.update(episodes=len(episodes),selected_deployments=[r['deployment'] for _,r in episodes],
                         upper_selector_on_policy=not balanced,wall_s=time.perf_counter()-started)
            history.append(stats)
            path=save_checkpoint(folder,model,opt,c,step,protocol,history)
            atomic_json(folder/'history.json',history)
            atomic_json(folder/'progress.json',dict(committed_updates=step,training_episodes=step*c['batch_episodes'],checkpoint=str(path),last=stats))
            print(json.dumps(dict(stage='committed',**stats)),flush=True)
            del episodes
            if c['validation_every'] and step%c['validation_every']==0:validate(step,path)
            # Delete only old raw arrays inside this exact study/run, never snapshots.
            if c['keep_rollouts']:
                for old in (folder/'rollouts').glob('u[0-9]*'):
                    if old.is_dir() and int(old.name[1:])<=step-c['keep_rollouts']:
                        root=folder.resolve()
                        for array in old.glob('*.npz'):
                            if root not in array.resolve().parents:raise ValueError('Unsafe pruning path')
                            array.unlink()
        atomic_json(folder/'status.json',dict(state=state,stage='finished',checkpoint=str(path),new_training_still_running=False))
    return path
