"""Prediction-only causal-window convolutional model, independent of PPO."""
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.dilation=dilation
        self.a=nn.Conv1d(width,width,3,dilation=dilation)
        self.b=nn.Conv1d(width,width,3,dilation=dilation)
        self.norm=nn.GroupNorm(8,width);self.drop=nn.Dropout(.05)

    def forward(self,x):
        y=F.gelu(self.a(F.pad(x,(2*self.dilation,0))))
        y=self.drop(y);y=self.b(F.pad(y,(2*self.dilation,0)))
        return F.gelu(self.norm(x+y))


class TemporalPredictor(nn.Module):
    def __init__(self,static_dim=646,width=64):
        super().__init__();self.static_dim=static_dim;self.width=width
        self.project=nn.Conv1d(111,width,1)
        self.blocks=nn.Sequential(*[Residual(width,d) for d in (1,2,4,8)])
        self.static=nn.Sequential(nn.Linear(static_dim,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.05))
        self.joint=nn.Sequential(nn.Linear(width*2+128,128),nn.LayerNorm(128),nn.GELU())
        self.any_head=nn.Linear(128,6);self.next_head=nn.Linear(128,17)

    def forward(self,history,static):
        x=self.blocks(self.project(history.transpose(1,2)))
        z=torch.cat([x[:,:,-1],x.mean(-1),self.static(static.clamp(-8,8))],-1)
        z=self.joint(z)
        return self.any_head(z),self.next_head(z)


class ResidualTransformer(nn.Module):
    """Start exactly at the old predictor, then learn supervised corrections."""
    def __init__(self,static_dim=646,width=64,warm_start=True):
        super().__init__();self.static_dim=static_dim;self.width=width
        from solo.model import RiskTransformer
        self.backbone=RiskTransformer();self.timing=nn.Linear(32,17)
        for parameter in self.backbone.pred_head.parameters():parameter.requires_grad_(False)
        self.static=nn.Sequential(nn.Linear(static_dim,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.05))
        self.delta=nn.Sequential(nn.Linear(166,128),nn.GELU(),nn.Linear(128,6))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
        if warm_start:
            from .contracts import BASELINE
            value=torch.load(BASELINE,map_location='cpu',weights_only=False)['model']
            self.backbone.load_state_dict({k[len('forecaster.backbone.'):]:v for k,v in value.items() if k.startswith('forecaster.backbone.')})
            self.timing.load_state_dict({k[len('forecaster.timing.'):]:v for k,v in value.items() if k.startswith('forecaster.timing.')})

    def forward(self,history,static):
        values=history[...,:48]*self.backbone.scale
        masks=history[...,48:96]>.5
        intents=history[...,96:105]*self.backbone.intent_scale
        geometry=static[...,537:583]*self.backbone.geo_scale
        z,_=self.backbone.encode_links(values,masks,intents,geometry)
        timing=self.timing(z);p=torch.softmax(timing.float(),-1)
        base=torch.stack([p[:,:2].sum(-1),p[:,:6].sum(-1),p[:,:10].sum(-1),p[:,:16].sum(-1),
                          p[:,6:8].sum(-1),p[:,6:14].sum(-1)],-1)
        features=torch.cat([z,self.static(static.clamp(-8,8)),base],-1)
        return torch.logit(base.clamp(1e-6,1-1e-6))+self.delta(features),timing
