from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
import torch
from cooperative.train import gae
from solo.contracts import ROOT,SCHEMA,sha,HISTORY
from solo.world import history

POLICY_KEYS=['context','candidates','mask','teacher','gain','geometry','state','chosen','logp','value','old_logits',
 'prediction_targets','prediction_valid','prediction_supported','rewards','rf_values','rf_masks','rf_intents']


def read_episode(prefix,purpose='train',keys=POLICY_KEYS):
    prefix=Path(prefix);meta=json.loads(prefix.with_suffix('.json').read_text())
    if meta.get('schema')!=SCHEMA or sha(prefix.with_suffix('.npz'))!=meta['array_sha256']:raise ValueError('Corrupt/wrong-schema episode '+str(prefix))
    plan=json.loads((ROOT/'configs/transformer_plan.json').read_text());roles={m['key']:m['joint_split'] for m in plan['maps']}
    if purpose and roles[meta['region']]!=purpose:raise ValueError('Data split violation: '+meta['region'])
    if meta.get('lightgbm_used') is not False or meta.get('gru_used') is not False:raise ValueError('Standalone data may not use GRU/LightGBM')
    with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as z:data={k:z[k] for k in keys}
    if len(data['rf_values'])!=len(data['geometry'])*6:raise ValueError('RF/control alignment mismatch')
    return data,meta


class EpisodeBatch:
    def __init__(self,prefixes,purpose='train'):
        self.data=[];self.index=[];self.metadata=[];advantages=[];returns=[]
        for file,prefix in enumerate(prefixes):
            data,meta=read_episode(prefix,purpose);n=len(data['rewards'])
            if n<2:raise ValueError('Complete episodes are required')
            a,r=gae(data['rewards'],data['value']);data.update(advantage=a,returns=r)
            self.data.append(data);self.metadata.append(meta);self.index.extend((file,t) for t in range(n));advantages.append(a);returns.append(r)
        self.advantages=np.concatenate(advantages).astype(np.float32);self.returns=np.concatenate(returns).astype(np.float32)
        self.advantages=(self.advantages-self.advantages.mean())/(self.advantages.std()+1e-6)
        self.value_scale=max(float(self.returns.std()),1.)
    def __len__(self):return len(self.index)
    def batch(self,indices,device='cpu'):
        indices=np.asarray(indices,dtype=int);rows=[self.index[int(i)] for i in indices]
        keys=['context','candidates','mask','teacher','gain','geometry','state','chosen','logp','value','old_logits',
              'prediction_targets','prediction_valid','prediction_supported']
        out={k:np.stack([self.data[f][k][t] for f,t in rows]) for k in keys}
        for dest,src in [('obs_values','rf_values'),('obs_masks','rf_masks'),('obs_intents','rf_intents')]:
            out[dest]=np.stack([history(self.data[f][src],t) for f,t in rows])
        out['advantage']=self.advantages[indices];out['returns']=self.returns[indices]
        return {k:torch.as_tensor(v,device=device) for k,v in out.items()}


def actor_inputs(batch):
    return [batch[k] for k in ['context','candidates','mask','teacher','gain','obs_values','obs_masks','obs_intents','geometry','prediction_supported']]


def prediction_loss(probabilities,batch):
    valid=batch['prediction_valid'].bool()&batch['prediction_supported'].bool()[...,None]
    weights=torch.as_tensor([1.,2.,2.,1.],device=probabilities.device)*valid
    bce=torch.nn.functional.binary_cross_entropy(probabilities.clamp(1e-7,1-1e-7),batch['prediction_targets'].float(),reduction='none')
    return (bce*weights).sum()/weights.sum().clamp_min(1),int(valid.sum().item())


PRED_KEYS=['rf_values','rf_masks','rf_intents','geometry','prediction_targets','prediction_valid','prediction_supported']


class PredictorPool:
    def __init__(self,paths,purpose='train',capacity=2):self.paths=list(paths);self.purpose=purpose;self.capacity=capacity;self.cache=OrderedDict()
    def episode(self,index):
        if index not in self.cache:
            data,meta=read_episode(self.paths[index],self.purpose,PRED_KEYS)
            eligible=data['prediction_valid'][...,0]&data['prediction_supported']
            pos=np.argwhere(eligible&(data['prediction_targets'][...,0]>.5));neg=np.argwhere(eligible&(data['prediction_targets'][...,0]<.5))
            self.cache[index]=(data,pos,neg)
            while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        self.cache.move_to_end(index);return self.cache[index]
    def sample(self,rng,batch_size=256,device='cpu'):
        data,pos,neg=self.episode(int(rng.integers(len(self.paths))))
        if not len(pos) and not len(neg):raise ValueError('No eligible supervised link samples')
        if len(pos) and len(neg):
            npos=batch_size//2;nneg=batch_size-npos
            rows=np.concatenate([pos[rng.integers(len(pos),size=npos)],neg[rng.integers(len(neg),size=nneg)]])
            prevalence=len(pos)/(len(pos)+len(neg))
            # Correct the deliberate positive oversampling; retain population probability calibration.
            importance=np.r_[np.full(npos,prevalence/(npos/batch_size)),np.full(nneg,(1-prevalence)/(nneg/batch_size))]
        else:
            source=pos if len(pos) else neg;rows=source[rng.integers(len(source),size=batch_size)];importance=np.ones(batch_size)
        return link_batch(data,rows,device),torch.as_tensor(importance,dtype=torch.float32,device=device)


def link_batch(data,rows,device='cpu'):
    out={}
    for dest,src in [('values','rf_values'),('masks','rf_masks'),('intents','rf_intents')]:
        values=[]
        for t,relay,peer in rows:
            end=int(t)*6;part=data[src][max(0,end-HISTORY):end,relay,peer]
            if len(part)<HISTORY:part=np.concatenate([np.zeros((HISTORY-len(part),part.shape[-1]),dtype=part.dtype),part])
            values.append(part)
        out[dest]=np.stack(values)
    t,r,p=rows.T
    for dest,src in [('geometry','geometry'),('target','prediction_targets'),('valid','prediction_valid')]:out[dest]=data[src][t,r,p]
    return {k:torch.as_tensor(v,device=device) for k,v in out.items()}


def link_loss(probability,batch,importance):
    valid=batch['valid'].bool();weights=torch.as_tensor([1.,2.,2.,1.],device=probability.device)*valid*importance[:,None]
    bce=torch.nn.functional.binary_cross_entropy(probability.clamp(1e-7,1-1e-7),batch['target'].float(),reduction='none')
    return (bce*weights).sum()/weights.sum().clamp_min(1)
