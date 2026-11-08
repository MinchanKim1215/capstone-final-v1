"""Padded variable-agent batches; fake agents never enter any loss or metric."""
import numpy as np
import torch
from .rollout import observation_at


class Episodes:
    def __init__(self, episodes, config):
        self.episodes=episodes
        self.indices=[];self.advantages=[];self.returns=[];self.aux=[]
        for e,(a,row) in enumerate(episodes):
            rewards=a['rewards'].astype(float)
            values=a['value'].astype(float)
            advantages=np.zeros(len(rewards));gae=0.
            for t in range(len(rewards)-1,-1,-1):
                following=values[t+1] if t+1<len(values) else 0.
                delta=rewards[t]+config['gamma']*following-values[t]
                gae=delta+config['gamma']*config['gae_lambda']*gae
                advantages[t]=gae
            self.indices.extend((e,t) for t in range(len(rewards)))
            self.advantages.extend(advantages)
            self.returns.extend(advantages+values)
            good=a['aux_support']&a['aux_valid'] if len(a['aux_endpoints']) else np.zeros((0,))
            self.aux.extend((e,*r) for r in np.argwhere(good))
        self.advantages=np.asarray(self.advantages,np.float32)
        self.returns=np.asarray(self.returns,np.float32)
        self.advantages=(self.advantages-self.advantages.mean())/max(float(self.advantages.std()),1e-6)
        self.value_scale=max(float(self.returns.std()),.1)
        self.aux=np.asarray(self.aux,np.int64).reshape(-1,4)

    def __len__(self):
        return len(self.indices)

    def batch(self, indices, device='cpu'):
        batch=len(indices);nr,n=5,10
        shapes=dict(context=(nr,34),candidates=(nr,25,12),action_mask=(nr,25),
                    teacher=(nr,25),gain=(nr,25,n),geometry=(nr,n,46),support=(nr,n),
                    values=(nr,n,40,48),masks=(nr,n,40,48),intents=(nr,n,40,9),design=(5,))
        obs={k:np.zeros((batch,*shape),bool if k in ('action_mask','support','masks') else np.float32)
             for k,shape in shapes.items()}
        obs['action_mask'][:,:,0]=True
        meta=dict(agent_mask=np.zeros((batch,nr),bool),labels=np.full((batch,nr,n),16,np.int64),
                  label_valid=np.zeros((batch,nr,n),bool),chosen=np.zeros((batch,nr),np.int64),
                  logp=np.zeros((batch,nr),np.float32),old_logits=np.full((batch,nr,25),-1e9,np.float32),
                  state=np.zeros((batch,32),np.float32))
        meta['old_logits'][:,:,0]=0
        for j,index in enumerate(indices):
            e,t=self.indices[int(index)];a,_=self.episodes[e]
            b=observation_at(a,t);r,peers=b['support'].shape
            for key,value in b.items():
                slices=(j,)+tuple(slice(0,q) for q in value.shape)
                obs[key][slices]=value
            meta['agent_mask'][j,:r]=True
            for key in ('chosen','logp','old_logits'):
                meta[key][j,:r]=a[key][t]
            meta['labels'][j,:r,:peers]=a['labels'][t]
            meta['label_valid'][j,:r,:peers]=a['label_valid'][t]
            meta['state'][j]=a['state'][t]
        meta['advantage']=self.advantages[indices]
        meta['returns']=self.returns[indices]
        return ({k:torch.as_tensor(v,device=device) for k,v in obs.items()},
                {k:torch.as_tensor(v,device=device) for k,v in meta.items()})

    def aux_batch(self, rng, count=128, device='cpu'):
        if not len(self.aux):
            return None
        take=self.aux[rng.integers(len(self.aux),size=min(count,len(self.aux)))]
        out={k:[] for k in ('values','masks','intents','geometry','labels')}
        for e,t,r,p in take:
            a,_=self.episodes[e];end=int(a['aux_endpoints'][t])
            for key in ('values','masks','intents'):
                out[key].append(a['rf_'+key][end-40:end,r,p])
            out['geometry'].append(a['aux_geometry'][t,r,p])
            out['labels'].append(a['aux_labels'][t,r,p])
        return {k:torch.as_tensor(np.asarray(v),device=device).to(
                torch.bool if k=='masks' else torch.long if k=='labels' else torch.float32)
                for k,v in out.items()}
