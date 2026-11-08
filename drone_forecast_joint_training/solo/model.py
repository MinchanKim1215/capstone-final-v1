"""Single learned Transformer forecaster. No tree probabilities or recurrent network inputs."""
from pathlib import Path
import math,time
import numpy as np
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel,SDPBackend
from cooperative.policy import load_policy
from solo.contracts import ROOT,HISTORY

torch.backends.mha.set_fastpath_enabled(False)
SCALE=np.asarray([5,5,50,1,1,16,16,1,100,30,20,64,200,200,16,20,400,400,10,10,400,10,150,150,1,1,50,400,
 100,30,20,30,20,16,400,400,10,10,400,10,30,30,10,10,16,32,1,10],np.float32)
INTENT_SCALE=np.asarray([400,400,400,400,30,3,400,400,1],np.float32)
GEO_SCALE=np.asarray([30]*13+[1]*13+[30]*19+[1],np.float32)


class RiskTransformer(nn.Module):
    def __init__(self):
        super().__init__();self.hidden=32
        self.register_buffer('scale',torch.from_numpy(SCALE));self.register_buffer('intent_scale',torch.from_numpy(INTENT_SCALE))
        self.register_buffer('geo_scale',torch.from_numpy(GEO_SCALE))
        self.project=nn.Sequential(nn.Linear(105,32),nn.LayerNorm(32))
        pos=torch.arange(HISTORY)[:,None];freq=torch.exp(torch.arange(0,32,2)*(-math.log(10000.)/32))
        pe=torch.zeros(HISTORY,32);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('position',pe)
        # Independent initialization per layer; no stochastic dropout between behavior collection and PPO.
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(32,4,64,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.geometry=nn.Sequential(nn.Linear(46,32),nn.Tanh())
        self.norm=nn.LayerNorm(32);self.pred_head=nn.Linear(32,4)
        nn.init.normal_(self.pred_head.weight,std=.02)
        with torch.no_grad():self.pred_head.bias.copy_(torch.logit(torch.tensor([.65,.15,.4,.04])))

    def encode_links(self,values,masks,intents,geometry):
        """[links,40,48] and [links,46]; label arrays are not accepted by this API."""
        values=values.float();intents=intents.float();geometry=geometry.float();known=masks.bool()
        if values.shape[-2:]!=(HISTORY,48) or masks.shape!=values.shape:raise ValueError('Expected [links,40,48]')
        if intents.shape!=(*values.shape[:-1],9) or geometry.shape!=(*values.shape[:-2],46):raise ValueError('History/geometry shapes differ')
        scaled=torch.where(known,values,torch.zeros_like(values))/self.scale
        tokens=torch.cat([scaled.clamp(-8,8),known.float(),(intents/self.intent_scale).clamp(-8,8)],-1)
        valid=known.any(-1);nonempty=valid.any(-1)
        padding=~valid;safe=padding.clone();safe[~nonempty,-1]=False
        z=self.project(tokens)+self.position
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.layers:z=layer(z,src_key_padding_mask=safe)
        z=z[:,-1]*nonempty[:,None]
        g=(geometry/self.geo_scale).clamp(-8,8)*geometry[:,-1:].clamp(0,1)
        z=self.norm(z+self.geometry(g))*nonempty[:,None]
        q=torch.sigmoid(self.pred_head(z))
        p07=q[...,3];p5=p07*q[...,0];p37=p07*q[...,2];p34=torch.minimum(p5,p37)*q[...,1]
        return z,torch.stack([p5,p34,p37,p07],-1)

    def encoder_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith('pred_head.')]

    def forward(self,values,masks,intents,geometry):
        if values.shape[-3:]!=(HISTORY,8,48):raise ValueError('Expected histories ending in [40,8,48]')
        outer=values.shape[:-3]
        v=values.transpose(-3,-2).reshape(-1,HISTORY,48)
        m=masks.transpose(-3,-2).reshape(-1,HISTORY,48)
        c=intents.transpose(-3,-2).reshape(-1,HISTORY,9)
        z,p=self.encode_links(v,m,c,geometry.reshape(-1,46))
        return z.reshape(*outer,8,32),p.reshape(*outer,8,4)


class Controller(nn.Module):
    def __init__(self,mode='joint'):
        super().__init__()
        if mode not in ['physics','fixed','joint']:raise ValueError('Use physics, fixed or joint')
        self.mode=mode;self.base,_=load_policy(ROOT/'model/continuity/ensemble.pt')
        for name,p in self.base.named_parameters():
            if '.critic.' in name or name.startswith('critic.'):p.requires_grad_(False)
        self.risk=RiskTransformer()
        self.adapter=nn.Sequential(nn.Linear(167,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
        self.critic=nn.Sequential(nn.Linear(224,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        if mode!='joint':
            for p in self.risk.parameters():p.requires_grad_(False)
        if mode=='physics':
            for p in self.adapter.parameters():p.requires_grad_(False)
        self.last_prediction_ms=0.

    def policy(self,context,candidates,mask,teacher,gain,values,masks,intents,geometry,supported):
        if not mask.bool().any(-1).all():raise ValueError('No feasible action')
        if self.mode=='physics':
            p=torch.zeros((*context.shape[:-1],8,4),device=context.device)
            return self.base.actor(context,candidates,mask,teacher),p
        tick=time.perf_counter();z,p=self.risk(values,masks,intents,geometry)
        self.last_prediction_ms=(time.perf_counter()-tick)*1000
        structural=torch.ones((5,8),dtype=torch.bool,device=p.device)
        structural[torch.arange(5,device=p.device),torch.arange(3,8,device=p.device)]=False
        supported=supported.bool()&structural; risk=p*supported[...,None]
        # Every neural-dependent feature is rebuilt from raw observations here, including PPO updates.
        c=context.clone()
        for relay in range(5):
            peers=[j for j in range(8) if j!=relay+3]
            for slot,peer in enumerate(peers):
                c[...,relay,14+slot*11+6]=risk[...,relay,peer,0]
                c[...,relay,14+slot*11+7]=risk[...,relay,peer,2]
                c[...,relay,14+slot*11+8]=supported[...,relay,peer].float()
        f=candidates.clone()
        delta=(gain*(risk[...,0]+risk[...,2]).unsqueeze(-2)).sum(-1)/7.
        old_raw=.15*gain.sum(-1)/7.*10
        new_raw=old_raw+delta*10
        f[...,5]=new_raw.clamp(-30,30)/10
        old_weight=candidates[...,5]*10;new_weight=f[...,5]*10
        extra=torch.where(candidates[...,0:1,9]<.999,.15*(new_raw-old_raw),
                          .15*(torch.tanh(new_weight/3)-torch.tanh(old_weight/3)))
        t=teacher+extra
        logits=self.base.actor(c,f,mask,t)
        present=masks[...,-1,:,30].bool()|masks[...,-1,:,6].bool()
        pooled=(z*present[...,None]).sum(-2)/present.sum(-1,keepdim=True).clamp_min(1)
        feature=torch.cat([c,pooled,risk.flatten(-2)],-1)
        expanded=feature.unsqueeze(-2).expand(*f.shape[:-1],feature.shape[-1])
        logits=logits+self.adapter(torch.cat([expanded,f],-1)).squeeze(-1)
        return logits.masked_fill(~mask.bool(),-1e9),p

    def value(self,state):return self.critic(state).squeeze(-1)


def load_controller(path,device='cpu'):
    ck=torch.load(Path(path),map_location='cpu',weights_only=False)
    if ck.get('format')!='standalone-transformer-ppo-v5':raise ValueError('This is not a standalone Transformer checkpoint; GRU checkpoints cannot be resumed here')
    model=Controller(ck['mode']);model.load_state_dict(ck['model']);model.to(device);model.eval()
    return model,ck
