"""Matched natural-row LightGBM ablations, actual event-recall checkpoints."""
from pathlib import Path
import time
import gc
import numpy as np
import lightgbm as lgb
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,status
from .inference import columns
from .metrics import evaluate_rows
from forecast_recall.episodes import cohort
from forecast_extension.metrics import binary

def load(records):
    parts={k:[] for k in ('base','network','physics','y','up','support','uniform')};cases=[];offset=0
    for item in records:
        with np.load(item['path'],allow_pickle=False) as a:
            for k in parts:parts[k].append(a[k])
            cases.append((cohort(item),slice(offset,offset+len(a['ends']))));offset+=len(a['ends'])
    out={k:np.concatenate(v) for k,v in parts.items()};out['cases']=cases;return out

def assess(p,data):
    a=np.zeros((len(p),1,1,6),np.float32);a[:,0,0,5]=p
    return evaluate_rows([(ep,a[s]) for ep,s in data['cases']])

def fit(variant,seed,tr,val):
    plan=read(ARTIFACTS/'experiment_plan.json');folder=ARTIFACTS/'models'/f'v16_{variant}_s{seed}';folder.mkdir(parents=True,exist_ok=True);meta=folder/'model.json'
    pins={k:sha(ROOT/k) for k in ('tree_train.py','inference.py','metrics.py','features.py','physics.py','data.py')}
    if meta.exists():
        if read(meta)['source_pins']!=pins:raise ValueError('Completed treeimplementation changed')
        return meta
    x=columns(tr,variant);vx=columns(val,variant);heads={};history={};params=dict(plan['tree_params'],seed=seed,metric='None')
    for head in (2,5):
        record=folder/f'head{head}.json'
        if record.exists():
            old=read(record)
            if old['source_pins']!=pins or sha(old['path'])!=old['sha256']:raise ValueError('Partial tree checkpoint changed')
            heads[str(head)]={k:old[k] for k in ('path','sha256','selected_iteration')};history[str(head)]=old['history'];continue
        train_good=tr['uniform']&tr['support'];valid_good=val['support'].copy()
        if head==2:train_good &= tr['up'];valid_good &= val['up']
        dataset=lgb.Dataset(x[train_good],label=tr['y'][train_good,head],free_raw_data=True)
        target=folder/f'head{head}.txt';trace=[];best=None;best_iteration=None;began=time.monotonic()
        def callback(env):
            nonlocal best,best_iteration
            step=env.iteration+1
            if step%plan['tree_validate_every'] and step!=plan['tree_rounds']:return
            p=env.model.predict(vx,num_iteration=step,num_threads=2);ranking=binary(val['y'][valid_good,head],p[valid_good])
            score=assess(p,val) if head==5 else None;key=tuple(score['score']) if score else (ranking['ap'] or 0,)
            row=dict(iteration=step,selection_ranking=ranking,selection_event=score,seconds=time.monotonic()-began);trace.append(row)
            if best is None or key>best:best=key;best_iteration=step;target.write_text(env.model.model_to_string(num_iteration=step),encoding='utf-8')
            write(folder/f'head{head}_history.json',dict(history=trace,selection_only=True));status('stage3_tree_training',variant=variant,seed=seed,head=head,step=step,total=plan['tree_rounds'],best_step=best_iteration)
            print(variant,seed,head,step,'best',best_iteration,'AP',ranking['ap'],'R',None if score is None else score['point']['event_recall_3s'],flush=True)
        model=lgb.train(params,dataset,num_boost_round=plan['tree_rounds'],callbacks=[callback]);del model,dataset;gc.collect()
        value=dict(path=str(target),sha256=sha(target),selected_iteration=best_iteration,history=trace,source_pins=pins,created_at=now());write(record,value)
        heads[str(head)]={k:value[k] for k in ('path','sha256','selected_iteration')};history[str(head)]=trace
    if pins!={k:sha(ROOT/k) for k in pins}:raise ValueError('Tree source changed mid-training')
    write(folder/'history.json',dict(heads=history,selection_only=True));write(meta,dict(family='v16_tree',variant=variant,tag=folder.name,dimension=x.shape[1],seed=seed,heads=heads,source_pins=pins,
        params=params,iterations=plan['tree_rounds'],natural_training_rows=int((tr['uniform']&tr['support']).sum()),created_at=now(),confirmation_used=False))
    return meta

def run(records):
    plan=read(ARTIFACTS/'experiment_plan.json');tr=load([r for r in records if r['role']=='train']);val=load([r for r in records if r['role']=='selection']);out=[]
    for variant in plan['tree_variants']:
        paths=[fit(variant,seed,tr,val) for seed in plan['tree_seeds']];folder=ARTIFACTS/'models'/('v16_'+variant);folder.mkdir(exist_ok=True);meta=folder/'model.json'
        if not meta.exists():write(meta,dict(family='v16_ensemble',variant=variant,tag=folder.name,members=[dict(path=str(p),sha256=sha(p),weight=.5) for p in paths],created_at=now()))
        out.append(str(meta))
    del tr,val;gc.collect();return out
