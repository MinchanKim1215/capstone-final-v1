from pathlib import Path
import json
import numpy as np
import torch
from solo.contracts import ROOT,HEADS,sha,object_hash,atomic_json,atomic_torch,load_plan,sources,seed_all,rng_state,restore_rng
from solo.model import Controller,load_controller
from solo.data import read_episode,PRED_KEYS,link_batch
from solo.metrics import binary_metrics,fit_alarm,alert_metrics


def scenario_subset(specs,cases=0):
    if cases<0:raise ValueError('cases must be nonnegative')
    groups={}
    for s in specs:groups.setdefault(s['region'],[]).append(s)
    regions=sorted(groups,key=lambda k:object_hash(['validation-order',k]));ordered=[]
    for turn in range(max(len(v) for v in groups.values())):
        for i,region in enumerate(regions):ordered.append(groups[region][(turn+i)%len(groups[region])])
    assert len({(s['region'],s['direction'],s['kind'],s['seed']) for s in ordered})==len(specs)
    return ordered[:cases] if cases else ordered


def score_predictor(model,paths,device='cpu',limit_decisions=0):
    model.eval();ys=[[] for _ in HEADS];ps=[[] for _ in HEADS]
    with torch.no_grad():
        for prefix in paths:
            data,_=read_episode(prefix,'validation',PRED_KEYS)
            use=np.ones(len(data['geometry']),bool)
            if limit_decisions:
                use[:]=False;use[np.linspace(0,len(use)-1,min(limit_decisions,len(use))).astype(int)]=True
            rows=np.argwhere(data['prediction_supported']&use[:,None,None])
            for first in range(0,len(rows),256):
                part=rows[first:first+256];b=link_batch(data,part,device)
                _,p=model.encode_links(b['values'],b['masks'],b['intents'],b['geometry']);p=p.cpu().numpy()
                y=b['target'].cpu().numpy();valid=b['valid'].cpu().numpy()
                for h in range(4):ys[h].append(y[:,h][valid[:,h]]);ps[h].append(p[:,h][valid[:,h]])
    return {name:binary_metrics(np.concatenate(ys[h]) if ys[h] else [],np.concatenate(ps[h]) if ps[h] else []) for h,name in enumerate(HEADS)}


def reference_checkpoint():
    from solo.training import checkpoint,DEFAULTS
    plan=load_plan();path=ROOT/'results/transformer/reference/physics.pt'
    if not path.exists():
        state=rng_state();seed_all(2026102907);model=Controller('physics')
        atomic_torch(path,checkpoint(model,None,dict(DEFAULTS,mode='physics'),0,[],plan,device='cpu'));restore_rng(state)
    ck=torch.load(path,weights_only=False,map_location='cpu')
    if ck['source_sha256']!=sources() or ck['plan_sha256']!=object_hash(plan):raise ValueError('Physics reference changed')
    return path


def prediction_metrics(paths):
    ys=[[] for _ in HEADS];ps=[[] for _ in HEADS];episodes=[]
    for prefix in paths:
        with np.load(Path(prefix).with_suffix('.npz')) as z:
            p=z['prediction'];y=z['prediction_targets'];valid=z['prediction_valid']&z['prediction_supported'][...,None]
            episodes.append(dict(prediction=p.copy(),prediction_supported=z['prediction_supported'].copy(),links=z['links'].copy()))
            for h in range(4):ys[h].append(y[...,h][valid[...,h]]);ps[h].append(p[...,h][valid[...,h]])
    return {name:binary_metrics(np.concatenate(ys[h]),np.concatenate(ps[h])) for h,name in enumerate(HEADS)},episodes


def summarize(rows):
    fields=['search_connected_mean','pdr','timely_pdr','outage_total_s','outage_max_s','duration_ratio','decision_p95_ms','forecast_p95_ms','relay_travel_m']
    return dict(episodes=len(rows),regions=len({r['region'] for r in rows}),complete=sum(r['mission_completed'] for r in rows),
        safety_passed=all(r['audit']['passed'] for r in rows),minimum_separation_m=min(r['audit']['minimum_swept_separation_m'] for r in rows),
        means={k:float(np.mean([r[k] for r in rows])) for k in fields},
        worst_connectivity=min(r['search_connected_mean'] for r in rows),worst_pdr=min(r['pdr'] for r in rows),
        longest_outage_s=max(r['outage_max_s'] for r in rows),max_duration_ratio=max(r['duration_ratio'] for r in rows))


def analysis_sources():return {n:sha(ROOT/n) for n in ['solo/evaluate.py','solo/metrics.py','solo/alarms.py']}


def require_test(snapshot,freeze):
    if not freeze:raise ValueError('Final test requires a frozen full-validation selection')
    s=json.loads(Path(freeze).read_text());plan=load_plan()
    if s.get('format')!='standalone-transformer-selection-v5' or s['validation_cases']!=len(plan['validation']):raise ValueError('Final test requires full validation, not a reduced pilot')
    if s['source_sha256']!=sources() or s['analysis_sha256']!=analysis_sources() or s['plan_sha256']!=object_hash(plan):raise ValueError('Frozen source/data changed')
    for key in ['selected','reference']:
        if sha(ROOT/s[key+'_checkpoint'])!=s[key+'_sha256']:raise ValueError('Frozen checkpoint changed')
    if sha(snapshot) not in [s['selected_sha256'],s['reference_sha256']]:raise ValueError('Checkpoint was not selected before test')
    return s


def evaluate_checkpoint(snapshot,split='validation',workers=4,cases=0,freeze=None):
    from solo.training import collect
    if split not in ['validation','test']:raise ValueError('Use validation or test')
    plan=load_plan();snapshot=Path(snapshot);ck=torch.load(snapshot,weights_only=False,map_location='cpu')
    if ck['source_sha256']!=sources() or ck['plan_sha256']!=object_hash(plan):raise ValueError('Checkpoint contract changed')
    frozen=None
    if split=='test':
        frozen=require_test(snapshot,freeze)
        atomic_json(ROOT/'results/transformer/test_access.json',dict(freeze_sha256=sha(freeze),plan_sha256=object_hash(plan),checkpoint_sha256=sha(snapshot)))
    specs=scenario_subset(plan[split],cases);folder=ROOT/'results/transformer/evaluation'/split/sha(snapshot)[:16]
    paths,rows=collect(plan,snapshot,specs,folder,workers,True,'evaluate_'+split)
    metrics,episodes=prediction_metrics(paths);alarms={}
    if ck['mode']!='physics':
        if split=='validation':alarms={n:fit_alarm(episodes,h,limit) for n,h,limit in [('5s',1,5),('8s',2,8)]}
        elif sha(snapshot)==frozen['selected_sha256']:
            for name,rule in frozen['alarm_policy'].items():
                alarms[name]=dict(alert_metrics(episodes,rule['head'],rule['threshold'],rule['exit_threshold'],rule['horizon_s']),enabled=rule['enabled'],fit_source='frozen validation rule')
    result=dict(format='standalone-transformer-evaluation-v5',split=split,mode=ck['mode'],updates=ck['updates'],cases=len(rows),
        checkpoint=snapshot.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(snapshot),plan_sha256=object_hash(plan),source_sha256=sources(),
        analysis_sha256=analysis_sources(),communication=summarize(rows),prediction=metrics,alarms=alarms,
        lightgbm_used=False,gru_used=False,raw_paths=[p.relative_to(ROOT).as_posix() for p in paths],
        notes=['Physics reference has no learned outage predictor; its zero probability outputs are not a trained prediction model',
               'Raw physical/trajectory constraints retained; one learned Transformer forecaster in fixed/joint arms',
               'Classification uses correlated directed-link control-time samples; compare models on the same observation corpus'])
    atomic_json(folder/'summary.json',result)
    print(json.dumps(dict(stage='evaluation_complete',split=split,mode=ck['mode'],updates=ck['updates'],communication=result['communication'])),flush=True)
    return result


def select(names,workers=4,cases=0):
    from solo.training import best_path
    plan=load_plan();reference=reference_checkpoint();baseline=evaluate_checkpoint(reference,workers=workers,cases=cases)
    records=[]
    for name in names:
        path=best_path(ROOT/'results/transformer/train'/name);ck=torch.load(path,weights_only=False,map_location='cpu')
        if ck['mode']=='physics':continue
        result=evaluate_checkpoint(path,workers=workers,cases=cases);c=result['communication'];b=baseline['communication']
        feasible=c['complete']==c['episodes'] and c['safety_passed'] and c['max_duration_ratio']<=1.15 and c['means']['pdr']>=b['means']['pdr']-.0005
        score=[c['means']['search_connected_mean'],c['means']['pdr'],-c['longest_outage_s']]
        records.append(dict(run=name,checkpoint=path.relative_to(ROOT).as_posix(),sha256=sha(path),feasible=feasible,score=score,evaluation=result))
    if not records:raise ValueError('At least one Transformer candidate required')
    best=max([r for r in records if r['feasible']] or records,key=lambda r:tuple(r['score']))
    chosen=ROOT/best['checkpoint'];frozen=dict(format='standalone-transformer-selection-v5',plan_sha256=object_hash(plan),source_sha256=sources(),analysis_sha256=analysis_sources(),
        selected_run=best['run'],selected_checkpoint=best['checkpoint'],selected_sha256=sha(chosen),
        reference_checkpoint=reference.relative_to(ROOT).as_posix(),reference_sha256=sha(reference),validation_cases=baseline['cases'],
        candidates=records,alarm_policy=best['evaluation']['alarms'],meets_physics_constraints=best['feasible'],
        improves_physics_connectivity=best['score'][0]>baseline['communication']['means']['search_connected_mean'],
        test_used=False,real_flight_approved=False)
    file=ROOT/'results/transformer/frozen_selection.json'
    if file.exists() and (ROOT/'results/transformer/test_access.json').exists() and json.loads(file.read_text())!=frozen:raise ValueError('Cannot change selection after final test')
    atomic_json(file,frozen)
    # Score all forecasters on the identical physics-reference observations, independently of trajectory changes.
    common={}
    for r in records:
        model,_=load_controller(ROOT/r['checkpoint']);common[r['run']]=score_predictor(model.risk,[ROOT/p for p in baseline['raw_paths']])
    atomic_json(ROOT/'results/transformer/common_corpus_validation.json',dict(reference_sha256=sha(reference),cases=baseline['cases'],models=common,
        source_sha256=sources(),analysis_sha256=analysis_sources(),test_used=False))
    print(json.dumps(dict(stage='selection_complete',selected=best['run'],test_used=False)),flush=True)
    return file


def compare_lightgbm():
    """Optional benchmark only. The standalone actor/Transformer never calls this function."""
    from cooperative.prediction import ControlPredictor
    from solo.training import best_path
    from solo.model import RiskTransformer
    checkpoint=best_path(ROOT/'results/transformer/pretrain')
    neural=RiskTransformer();neural.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);neural.eval()
    manifest=json.loads((ROOT/'results/transformer/bootstrap/complete.json').read_text())
    ys=[[] for _ in range(3)];ps=[[] for _ in range(3)];nps=[[] for _ in range(3)];eligible=[0]*3
    plan=load_plan();entries={m['key']:m for m in plan['maps']}
    for prefix in manifest['validation_paths']:
        data,meta=read_episode(ROOT/prefix,'validation',PRED_KEYS)
        predictor=ControlPredictor(ROOT/'model/advance/selected/best.pt',ROOT/entries[meta['region']]['path'],observers=range(3,8))
        p=np.full((*data['prediction_supported'].shape,3),np.nan,np.float32)
        for t in range(len(data['rf_values'])):
            x=np.zeros((8,8,48),np.float32);m=np.zeros_like(x,bool);c=np.zeros((8,8,9),np.float32)
            x[3:8]=data['rf_values'][t];m[3:8]=data['rf_masks'][t];c[3:8]=data['rf_intents'][t]
            result=predictor.observe((t+1)*.5,x,m,c)
            decision=(t+1)//6
            if (t+1)%6==0 and decision<len(p) and result['status']=='ready':
                for row in result['links']:
                    i=row['source']-3;j=row['peer']
                    if 0<=i<5:
                        for h,key in enumerate(['risk_5s','risk_onset_3to4s','risk_onset_3to7s']):
                            if row.get(key) is not None:p[decision,i,j,h]=row[key]
        npred=np.full_like(p,np.nan);rows=np.argwhere(data['prediction_supported'])
        with torch.no_grad():
            for first in range(0,len(rows),256):
                part=rows[first:first+256];batch=link_batch(data,part)
                _,prediction=neural.encode_links(batch['values'],batch['masks'],batch['intents'],batch['geometry'])
                t,r,j=part.T;npred[t,r,j]=prediction.numpy()[...,:3]
        for h in range(3):
            eligible[h]+=int((data['prediction_supported']&data['prediction_valid'][...,h]).sum())
            keep=data['prediction_supported']&data['prediction_valid'][...,h]&np.isfinite(p[...,h])
            ys[h].append(data['prediction_targets'][...,h][keep]);ps[h].append(p[...,h][keep])
            nps[h].append(npred[...,h][keep])
        print(json.dumps(dict(stage='lightgbm_benchmark',region=meta['region'])),flush=True)
    result=dict(benchmark_only=True,used_by_standalone_policy=False,cases=len(manifest['validation_paths']),
                neural_checkpoint=checkpoint.relative_to(ROOT).as_posix(),neural_checkpoint_sha256=sha(checkpoint),
                cohort='Identical public observations, timestamps, labels and finite baseline eligibility for both predictors',
                eligible_samples=dict(zip(HEADS[:3],eligible)),
                metrics={HEADS[h]:binary_metrics(np.concatenate(ys[h]),np.concatenate(ps[h])) for h in range(3)},
                standalone_transformer={HEADS[h]:binary_metrics(np.concatenate(ys[h]),np.concatenate(nps[h])) for h in range(3)})
    atomic_json(ROOT/'results/transformer/lightgbm_reference.json',result);return result
