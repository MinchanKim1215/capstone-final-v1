"""Rare-event classification plus dense future channel quantile supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from .models import ResidualTransformer


class DenseResidual(ResidualTransformer):
    def __init__(self,static_dim=728,width=64,warm_start=True):
        super().__init__(static_dim,width,warm_start)
        self.margin_head=nn.Linear(128,24)

    def forward(self,history,static,return_margin=False):
        values=history[...,:48]*self.backbone.scale;masks=history[...,48:96]>.5
        intents=history[...,96:105]*self.backbone.intent_scale
        geometry=static[...,537:583]*self.backbone.geo_scale
        z,_=self.backbone.encode_links(values,masks,intents,geometry)
        timing=self.timing(z);p=torch.softmax(timing.float(),-1)
        base=torch.stack([p[:,:2].sum(-1),p[:,:6].sum(-1),p[:,:10].sum(-1),p[:,:16].sum(-1),
                          p[:,6:8].sum(-1),p[:,6:14].sum(-1)],-1)
        features=torch.cat([z,self.static(static.clamp(-8,8)),base],-1)
        latent=self.delta[1](self.delta[0](features))
        logits=torch.logit(base.clamp(1e-6,1-1e-6))+self.delta[2](latent)
        if not return_margin:return logits,timing
        raw=self.margin_head(latent).reshape(-1,8,3)
        median=raw[...,1]
        quantiles=torch.stack([median-F.softplus(raw[...,0]),median,median+F.softplus(raw[...,2])],-1)
        return logits,timing,quantiles
