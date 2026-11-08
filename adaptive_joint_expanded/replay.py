"""Matched-observation prediction comparison; policy changes cannot change inputs."""
from pathlib import Path
import numpy as np
import torch
from solo.model import RiskTransformer
from solo.metrics import binary_metrics
from cooperative.fast_features import geometry_profile
from usable.collect import prepare_city_without_unused_radio_cache
from swarm_sim.config import Scenario
from . import LEGACY,ARTIFACTS
from .contracts import PILOT,DEPLOYMENTS,atomic_json,entry_for,read_json,sha,digest
from .experiment import assert_protocol,load_checkpoint
from .model import TimeForecaster
from .observation import Observer
from .targets import timing_labels,binary_targets,probabilities,TARGET_NAMES


def compare(checkpoint,protocol,folder,device='cuda',max_times=None):
    assert_protocol(protocol)
    joint,_,_=load_checkpoint(checkpoint,protocol,device)
    warm=TimeForecaster().to(device)
    warm.load_state_dict(torch.load(PILOT,map_location='cpu',weights_only=False)['model'])
    old=RiskTransformer().to(device)
    old_path=LEGACY/'results/transformer/train/transformer_main/checkpoints/s0000176.pt'
    ck=torch.load(old_path,map_location='cpu',weights_only=False)
    old.load_state_dict({k[5:]:v for k,v in ck['model'].items() if k.startswith('risk.')})
    joint.eval();warm.eval();old.eval()
    all_y=[];scores={k:[] for k in ('warm_1s5500','joint_v6','old176_scores_on_v6_targets')}
    cases=[]
    for meta in sorted(Path(folder).glob('*.json')):
        row=read_json(meta)
        if 'array_sha256' not in row:continue
        path=meta.with_suffix('.npz')
        if sha(path)!=row['array_sha256']:raise ValueError('Common-observation source changed')
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        d=next(d for d in DEPLOYMENTS if d.n==row['total_drones'] and d.ns==row['search_drones'])
        entry=entry_for(row['region'])
        scenario=Scenario(duration_s=7200,motion_dt_s=3,direction=row['direction']).validate()
        city=prepare_city_without_unused_radio_cache(LEGACY/entry['path'],scenario,ARTIFACTS/'cache/city')
        endpoints=np.arange(40,len(a['links'])-16,2)
        if max_times is not None:endpoints=endpoints[:max_times]
        y,valid=timing_labels(a['links'],endpoints,d.ns)
        observer=Observer();observer.bind(city,d)
        for t,end in enumerate(endpoints):
            observer.values=a['rf_values'][:end]
            observer.masks=a['rf_masks'][:end]
            observer.intents=a['rf_intents'][:end]
            b=observer.inputs()
            good=b['support']&valid[t]
            if not good.any():continue
            all_y.append(binary_targets(y[t])[good])
            v=torch.as_tensor(b['values'][good],device=device)
            m=torch.as_tensor(b['masks'][good],device=device)
            c=torch.as_tensor(b['intents'][good],device=device)
            g=torch.as_tensor(b['geometry'][good],device=device)
            with torch.no_grad():
                for name,model in [('warm_1s5500',warm),('joint_v6',joint.forecaster)]:
                    _,q=model.encode(v,m,c,g)
                    scores[name].append(probabilities(q).cpu().numpy())
                _,q=old.encode_links(v,m,c,g)
                scores['old176_scores_on_v6_targets'].append(q.cpu().numpy())
        cases.append(dict(region=row['region'],source_array_sha256=row['array_sha256'],
                          source_controller=row.get('deployment'),smoke_only=row['smoke_only']))
    if not all_y:raise ValueError('No eligible common observations in this folder')
    truth=np.concatenate(all_y)
    results={}
    for name,parts in scores.items():
        p=np.concatenate(parts)
        results[name]={target:binary_metrics(truth[:,h],p[:,h]) for h,target in enumerate(TARGET_NAMES)}
        for value in results[name].values():
            if value.get('tp',0)+value.get('fp',0)==0:value['precision']=None
    scope='SMOKE ONLY' if any(r['smoke_only'] for r in cases) or max_times is not None else 'Matched reconstructed 1s observations; source policy unchanged'
    result=dict(scope=scope,cases=cases,common_samples=len(truth),models=results,
                checkpoint_sha256=sha(checkpoint),warm_predictor_sha256=sha(PILOT),old176_sha256=sha(old_path),
                protocol_sha256=digest(protocol),
                caution='Old176 scores are evaluated against v6 next-onset truth, not their native any-event targets. This is not the old HTML AP comparison.',
                common_input_contract='Same RF history, causal masks, reconstructed geometry, support, timestamps and new targets for every predictor')
    return result
