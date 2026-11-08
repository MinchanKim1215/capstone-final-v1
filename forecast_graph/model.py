"""Temporal message passing, causal edge memory and optional physics correction."""
import torch
from torch import nn
from .features import NODE,EDGE,PHYS,GLOBAL
from .codec import MAX_N


def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.SiLU(),nn.Linear(b,c))


class ForecastGraph(nn.Module):
    def __init__(self,variant,hidden=64):
        super().__init__();self.variant=variant;self.hybrid=variant=='hybrid_shared';h=hidden
        self.node=mlp(NODE,h,h);self.edge=mlp(EDGE,32,32)
        self.message=mlp(2*h+32,h,h);self.gate=nn.Linear(h,1)
        self.update=nn.GRUCell(h,h);self.temporal=nn.GRU(h,h,batch_first=True)
        self.edge_time=nn.GRU(EDGE,32,batch_first=True)
        self.pair=mlp(2*h+32+9*PHYS,96,96)
        self.link=nn.Linear(96+9,6);self.aux=nn.Linear(96,9)
        self.service=mlp(4*h+2*96+GLOBAL+9,128,6)

    def forward(self,node,edge,physics,global_features,n,ns,observer):
        b,t,k,_=node.shape;device=node.device
        valid=torch.arange(k,device=device)[None,:]<n[:,None]
        pair_valid=valid[:,:,None]&valid[:,None,:]
        h=self.node(node);e=self.edge(edge)
        for step in range(2):
            hi=h[:,:,:,None,:].expand(-1,-1,-1,k,-1);hj=h[:,:,None,:,:].expand(-1,-1,k,-1,-1)
            message=self.message(torch.cat([hi,hj,e],-1))
            mask=pair_valid[:,None,:,:,None]
            weighted=message*torch.sigmoid(self.gate(message))*mask
            aggregate=weighted.sum(3)/n[:,None,None,None].clamp_min(1)
            h=self.update(aggregate.reshape(-1,h.shape[-1]),h.reshape(-1,h.shape[-1])).reshape_as(h)
            h=h*valid[:,None,:,None]
        temporal=h.transpose(1,2).reshape(b*k,t,-1)
        state=self.temporal(temporal)[0][:,-1].reshape(b,k,-1)
        et=edge.permute(0,2,3,1,4).reshape(b*k*k,t,EDGE)
        estate=self.edge_time(et)[0][:,-1].reshape(b,k,k,32)
        hi=state[:,:,None].expand(-1,-1,k,-1);hj=state[:,None].expand(-1,k,-1,-1)
        ph=physics.permute(0,2,3,1,4).reshape(b,k,k,9*PHYS) if self.hybrid else torch.zeros((b,k,k,9*PHYS),device=device,dtype=node.dtype)
        pairs=self.pair(torch.cat([hi,hj,estate,ph],-1))
        residual=self.aux(pairs).permute(0,3,1,2)
        reference=physics[...,0] if self.hybrid else torch.zeros_like(residual)
        aux=reference+residual
        correction=aux.permute(0,2,3,1) if self.hybrid else torch.zeros_like(residual).permute(0,2,3,1)
        logits=self.link(torch.cat([pairs,correction],-1))
        ids=torch.arange(k,device=device)[None,:];search=(ids<ns[:,None])&valid;relay=(ids>=ns[:,None])&valid
        def pool(x,mask):
            mean=(x*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            maximum=x.masked_fill(~mask[...,None],-1e4).max(1).values
            return torch.cat([mean,maximum],-1)
        pm=pair_valid&~torch.eye(k,device=device,dtype=torch.bool)[None]
        flat=pairs.reshape(b,k*k,-1);pmean=(flat*pm.reshape(b,-1,1)).sum(1)/pm.sum((1,2)).clamp_min(1)[:,None]
        pmax=flat.masked_fill(~pm.reshape(b,-1,1),-1e4).max(1).values
        gf=global_features if self.hybrid else torch.zeros_like(global_features)
        connected=torch.zeros((b,9),device=device,dtype=node.dtype)
        if self.hybrid:
            #A differentiable widest-path proxy connects corrected channel
            #forecasts to the service head. It is not an exact packetPDR model.
            availability=torch.sigmoid(aux*6)
            reach=torch.minimum(availability,availability.transpose(-1,-2))*pair_valid[:,None]
            eye=torch.eye(k,device=device,dtype=torch.bool)[None,None]
            reach=torch.where(eye,torch.ones_like(reach),reach)
            for middle in range(k):
                via=torch.minimum(reach[:,:,:,middle,None],reach[:,:,None,middle,:])
                reach=torch.maximum(reach,via)
            search_pair=search[:,:,None]&search[:,None,:]
            connected=torch.where(search_pair[:,None],reach,torch.ones_like(reach)).amin((-1,-2))
        service=self.service(torch.cat([pool(state,search),pool(state,relay),pmean,pmax,gf,connected],-1))
        chosen=logits[torch.arange(b,device=device),observer]
        return dict(link=chosen,service=service,aux=aux)


def tensors(arrays,positions,variant,device='cuda'):
    import numpy as np
    suffix='local' if variant=='graph_local' else 'shared'
    names={'node':'node_'+suffix,'edge':'edge_'+suffix,'physics':'physics','global_features':'global','n':'n','ns':'ns','observer':'observer'}
    result={}
    for key,name in names.items():
        a=np.asarray(arrays[name][positions])
        result[key]=torch.as_tensor(a,device=device,dtype=torch.long if key in ('n','ns','observer') else torch.float32)
    return result
