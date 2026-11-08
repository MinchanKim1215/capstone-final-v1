"""Matched-seed, one-change-at-a-time tree experiments on two distinct tasks."""
from pathlib import Path
import time
import numpy as np
import lightgbm as lgb
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now,verify,status,V9
from .features import BASE_DIM
from forecast_rebuild.dataset import Samples
from forecast_rebuild.metrics import binary


def samples(task,split):
    if task=='link':
        data=Samples(V9/'samples/merged_trees',split)
        return dict(x=data.x,y=data.y,up=data.up,support=data.legacy)
    folder=ARTIFACTS/'samples'
    return {name:np.load(folder/(split+'_'+name+'.npy'),mmap_mode='r',allow_pickle=False) for name in ('x','y','up','support')}


def seed_tag(task,variant,seed):return f'v10_{task}_{variant}_s{seed}'


def train_one(task,variant,seed,weights=None):
    verify();folder=ARTIFACTS/'models'/seed_tag(task,variant,seed);manifest=folder/'model.json';folder.mkdir(parents=True,exist_ok=True)
    sources={name:sha(ROOT/name) for name in ('train.py','features.py','data.py')}
    if manifest.exists():
        previous=read(manifest)
        if previous['source_sha256']!=sources:raise ValueError('Cannot resume a trained model with altered code')
        return manifest
    data=samples(task,'train');validation=samples(task,'selection')
    dimension=data['x'].shape[1] if task=='link' or variant=='targeted_topology' else BASE_DIM
    x=data['x'][:,:dimension];vx=validation['x'][:,:dimension]
    params=dict(objective='binary',metric='average_precision',learning_rate=.04,num_leaves=31,max_bin=127,min_data_in_leaf=100,
        lambda_l2=5.,feature_fraction=.9,bagging_fraction=.85,bagging_freq=1,num_threads=4,verbosity=-1,
        deterministic=True,force_col_wise=True,seed=seed,is_unbalance=False)
    heads={};scores={};started=time.monotonic()
    for head in (2,5):
        good=validation['support']&validation['up'] if head==2 else validation['support']
        if np.asarray(data['y'][:,head]).sum()==0 or np.asarray(validation['y'][good,head]).sum()==0:
            raise ValueError('Insufficient positive examples; do not report a trained predictor for this target')
        weight=None if weights is None else np.load(weights[str(head)],mmap_mode='r')
        trainset=lgb.Dataset(x,label=np.asarray(data['y'][:,head]),weight=weight,free_raw_data=True)
        valset=lgb.Dataset(vx[good],label=np.asarray(validation['y'][good,head]),reference=trainset,free_raw_data=True)
        model=lgb.train(params,trainset,num_boost_round=800,valid_sets=[valset],callbacks=[lgb.early_stopping(60,first_metric_only=True,verbose=False)])
        path=folder/f'head{head}.txt';path.write_text(model.model_to_string(),encoding='utf-8')
        p=model.predict(vx,num_threads=4);scores[str(head)]=binary(validation['y'][good,head],p[good])
        heads[str(head)]=dict(path=str(path),sha256=sha(path),iterations=model.best_iteration)
        write(folder/'status.json',dict(state='training',head=head,updated_at=now()));print(folder.name,head,scores[str(head)]['ap'],flush=True)
        del model,trainset,valset
    if sources!={name:sha(ROOT/name) for name in sources}:raise ValueError('Trainer implementation changed during fitting')
    result=dict(family='lightgbm' if task=='link' else 'service_lightgbm',task=task,variant=variant,tag=folder.name,dimension=dimension,
        heads=heads,use_energy=task=='link',use_plan=task=='link',optional_energy_in_service_features=task=='service',
        selection=scores,seed=seed,source_sha256=sources,params=params,weights=weights,created_at=now(),wall_seconds=time.monotonic()-started,
        experiment_plan_sha256=sha(ARTIFACTS/'experiment_plan.json'),calibration_used=False,confirmation_used=False)
    write(manifest,result);write(folder/'status.json',dict(state='completed',wall_seconds=result['wall_seconds']));return manifest


def ensemble(task,variant,members):
    path=ARTIFACTS/'models'/f'v10_{task}_{variant}'/'model.json'
    value=dict(family='ensemble' if task=='link' else 'service_ensemble',task=task,variant=variant,tag=path.parent.name,
        members=[dict(path=str(m),model_sha256=sha(m),weight=1/len(members)) for m in members],
        use_energy=task=='link',use_plan=task=='link',dimension=read(members[0])['dimension'],created_at=now())
    if path.exists():
        if read(path)['members']!=value['members']:raise ValueError('Seed ensemble changed')
    else:write(path,value)
    return path


def weights_for(task,uniform_members):
    folder=ARTIFACTS/'weights';folder.mkdir(exist_ok=True);manifest=folder/(task+'.json')
    if manifest.exists():return read(manifest)['paths']
    plan=read(ARTIFACTS/'experiment_plan.json')['targeted_weights'];data=samples(task,'train')
    dimension=data['x'].shape[1] if task=='link' else BASE_DIM;x=data['x'][:,:dimension];paths={};diagnostic={}
    for head in (2,5):
        scores=np.zeros(len(x),np.float32)
        for member in uniform_members:
            item=read(member)['heads'][str(head)];booster=lgb.Booster(model_str=Path(item['path']).read_text(encoding='utf-8'))
            for start in range(0,len(x),32768):scores[start:start+32768]+=booster.predict(x[start:start+32768],num_threads=4)/len(uniform_members)
        y=np.asarray(data['y'][:,head],bool);prior=float(y.mean());target=plan['positive_target_fraction']
        multiplier=min(plan['maximum_positive_multiplier'],max(1,target*(1-prior)/max(prior*(1-target),1e-12)))
        negative_threshold=float(np.quantile(scores[~y],.9));hard=(~y)&(scores>=negative_threshold)
        weight=np.where(y,multiplier,np.where(hard,plan['hard_negative_multiplier'],1.)).astype(np.float32);weight/=weight.mean()
        path=folder/(task+'_head'+str(head)+'.npy');np.save(path,weight,allow_pickle=False);paths[str(head)]=str(path)
        diagnostic[str(head)]=dict(positive_prior=prior,positive_multiplier=multiplier,hard_negative_threshold=negative_threshold,hard_negative_count=int(hard.sum()),
            train_rows=len(y),weight_sha256=sha(path),validation_mined=False)
    write(manifest,dict(paths=paths,diagnostic=diagnostic,uniform_models=[dict(path=str(m),sha256=sha(m)) for m in uniform_members],created_at=now()))
    return paths


def main():
    verify();plan=read(ARTIFACTS/'experiment_plan.json');catalog=[]
    for task in ('link','service'):
        status('stage3_'+task+'_uniform_training')
        uniform=[train_one(task,'uniform',seed) for seed in plan['seeds']]
        catalog.append(dict(task=task,model=str(ensemble(task,'uniform',uniform))))
        weights=weights_for(task,uniform)
        status('stage3_'+task+'_targeted_training')
        targeted=[train_one(task,'targeted',seed,weights) for seed in plan['seeds']]
        catalog.append(dict(task=task,model=str(ensemble(task,'targeted',targeted))))
        if task=='service':
            status('stage3_service_topology_training')
            topology=[train_one(task,'targeted_topology',seed,weights) for seed in plan['seeds']]
            catalog.append(dict(task=task,model=str(ensemble(task,'targeted_topology',topology))))
    write(ARTIFACTS/'trained_models.json',dict(candidates=catalog,created_at=now(),seeds=plan['seeds'],source_sha256=sha(__file__)))


if __name__=='__main__':main()
