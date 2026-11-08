"""Strong tree controls retain old inputs and add received graph/physics inputs."""
from pathlib import Path
import numpy as np
import lightgbm as lgb
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status
from .codec import MAX_N
from forecast_rebuild.dataset import Episode
from forecast_rebuild.planning import public_context


def extra(arrays,task):
    node=arrays['node_shared'].astype(np.float32);edge=arrays['edge_shared'].astype(np.float32)
    ph=arrays['physics'].astype(np.float32);g=arrays['global'].astype(np.float32)
    count=len(node);idx=np.arange(count);observer=arrays['observer'];n=arrays['n'];ns=arrays['ns']
    if task=='link':
        own=np.broadcast_to(node[idx,-1,observer,None],(count,MAX_N,node.shape[-1]))
        peer=node[:,-1];eh=edge[idx,:,observer].transpose(0,2,1,3).reshape(count,MAX_N,-1)
        pf=ph[idx,:,observer].transpose(0,2,1,3).reshape(count,MAX_N,-1)
        return np.concatenate([own,peer,eh,pf,np.broadcast_to(g[:,None],(count,MAX_N,g.shape[-1]))],-1)
    valid=np.arange(MAX_N)[None,:]<n[:,None];search=np.arange(MAX_N)[None,:]<ns[:,None]
    pieces=[]
    for mask in (valid,search,valid&~search):
        for x in (node[:,-1],node.mean(1),node[:,-1]-node[:,0]):
            mean=(x*mask[...,None]).sum(1)/np.maximum(mask.sum(1),1)[:,None]
            low=np.where(mask[...,None],x,np.inf).min(1);high=np.where(mask[...,None],x,-np.inf).max(1)
            pieces.extend([mean,low,high])
    pm=valid[:,:,None]&valid[:,None,:]&~np.eye(MAX_N,dtype=bool)[None]
    for x in (edge[:,-1],edge.mean(1)):
        pieces.extend([(x*pm[...,None]).sum((1,2))/np.maximum(pm.sum((1,2)),1)[:,None],
            np.where(pm[...,None],x,np.inf).min((1,2)),np.where(pm[...,None],x,-np.inf).max((1,2))])
    pieces.append(g)
    return np.concatenate(pieces,-1)


def features(record,arrays,ep=None,city=None):
    ep=Episode(record) if ep is None else ep
    if city is None:city,_=public_context(record)
    q=arrays['query'];t,r=q.T;n=ep.values.shape[2];count=len(q)
    grids=np.stack([np.repeat(t,n),np.repeat(r,n),np.tile(np.arange(n),count)],-1)
    static=[]
    for start in range(0,len(grids),512):static.append(ep.features(grids[start:start+512],use_energy=True,planning=True))
    old=np.concatenate(static).reshape(count,n,-1)
    lx=np.concatenate([old,extra(arrays,'link')[:,:n]],-1)
    from forecast_service.features import build as service_features
    sx=[]
    for start in range(0,len(q),256):sx.append(service_features(ep,q[start:start+256],record['n_search'],city))
    sx=np.concatenate([np.concatenate(sx),extra(arrays,'service')],-1)
    return lx,sx


def tables(records):
    parts={key:[] for key in ('lx','ly','sx','sy','lup','sup')}
    for i,item in enumerate(records):
        record=item['source_record']
        with np.load(item['path'],allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
        lx,sx=features(record,arrays);n=record['n_search']+arrays['n'][0]-record['n_search']
        good=arrays['link_support'][:,:n]
        parts['lx'].append(lx[good]);parts['ly'].append(arrays['link_y'][:,:n][good]);parts['lup'].append(arrays['link_up'][:,:n][good])
        sg=arrays['service_support'];parts['sx'].append(sx[sg]);parts['sy'].append(arrays['service_y'][sg]);parts['sup'].append(arrays['service_up'][sg])
        if i%20==0:print('tree features',i+1,'/',len(records),flush=True)
    return {k:np.concatenate(v) for k,v in parts.items()}


def main():
    p=verify();catalog=read(ARTIFACTS/'data_manifest.json')['records']
    tr=tables([r for r in catalog if r['role']=='train']);val=tables([r for r in catalog if r['role']=='selection'])
    pins={name:sha(ROOT/name) for name in ('tree.py','features.py','data.py')}
    write(ARTIFACTS/'tree_control_plan.json',dict(created_at=now(),source_pins=pins,
        variants=['existing_inputs','shared'],same_queries=True,same_seeds=p['seeds'],same_hyperparameters=True,
        old_dimensions=dict(link=728,service=432),purpose='Isolate the added information from model family and training-query changes'))
    candidates=[]
    for variant in ('existing_inputs','shared'):
      members=[]
      for seed in p['seeds']:
        folder=ARTIFACTS/'models'/f'v11_tree_{variant}_s{seed}';folder.mkdir(parents=True,exist_ok=True);manifest=folder/'model.json'
        if manifest.exists():
            if read(manifest)['source_pins']!=pins:raise ValueError('Tree control changed')
            members.append(str(manifest));continue
        heads={};dimensions={}
        for task,prefix in [('link','l'),('service','s')]:
            dimension=(728 if task=='link' else 432) if variant=='existing_inputs' else tr[prefix+'x'].shape[1]
            dimensions[task]=dimension
            heads[task]={}
            for head in (2,5):
                good=val[prefix+'up'] if head==2 else np.ones(len(val[prefix+'x']),bool)
                params=dict(objective='binary',metric='average_precision',learning_rate=.04,num_leaves=31,max_bin=127,
                    min_data_in_leaf=100,lambda_l2=5.,feature_fraction=.9,bagging_fraction=.85,bagging_freq=1,
                    num_threads=4,verbosity=-1,deterministic=True,force_col_wise=True,seed=seed)
                a=lgb.Dataset(tr[prefix+'x'][:,:dimension],label=tr[prefix+'y'][:,head],free_raw_data=True)
                b=lgb.Dataset(val[prefix+'x'][good,:dimension],label=val[prefix+'y'][good,head],reference=a,free_raw_data=True)
                booster=lgb.train(params,a,num_boost_round=800,valid_sets=[b],callbacks=[lgb.early_stopping(60,verbose=False)])
                path=folder/f'{task}_head{head}.txt';path.write_text(booster.model_to_string(),encoding='utf-8')
                heads[task][str(head)]=dict(path=str(path),sha256=sha(path),iterations=booster.best_iteration)
                print('tree',variant,seed,task,head,booster.best_score,flush=True);del a,b,booster
        write(manifest,dict(family='v11_tree',tag=folder.name,heads=heads,source_pins=pins,seed=seed,created_at=now(),
            variant=variant,input_dimensions=dimensions,old_inputs_retained=True,calibration_used=False,confirmation_used=False))
        members.append(str(manifest))
      folder=ARTIFACTS/'models'/('v11_tree_'+variant);folder.mkdir(exist_ok=True);path=folder/'model.json'
      if not path.exists():write(path,dict(family='v11_tree_ensemble',tag=folder.name,members=[dict(path=m,sha256=sha(m),weight=.5) for m in members],created_at=now()))
      candidates.append(str(path))
    write(ARTIFACTS/'tree_status.json',dict(state='completed',models=candidates,created_at=now(),new_tree_models=4))


if __name__=='__main__':main()

