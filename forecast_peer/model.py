"""Per-peer temporal memory retains individual RF details before attention."""
import math
import torch
from torch import nn
from .features import DIM

class PeerAdapter(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.global_project=nn.Sequential(nn.Linear(480,96),nn.SiLU(),nn.LayerNorm(96))
        self.peer_project=nn.Sequential(nn.Linear(DIM,96),nn.SiLU(),nn.LayerNorm(96))
        self.memory=nn.GRU(96,64,batch_first=True);self.query=nn.Linear(481,64)
        self.head=nn.Sequential(nn.Linear(545,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,embedding,logits,global_history,peer_history,n,ns):
        if self.variant=='global_event':context=self.memory(self.global_project(global_history))[0][:,-1]
        else:
            b,t,k,d=peer_history.shape;x=self.peer_project(peer_history).transpose(1,2).reshape(b*k,t,96)
            state=self.memory(x)[0][:,-1].reshape(b,k,64)
            ids=torch.arange(k,device=state.device)[None];valid=(ids<n[:,None])&(ids!=ns[:,None])
            weight=(state*self.query(embedding)[:,None]).sum(-1)/math.sqrt(64);weight=torch.softmax(weight.masked_fill(~valid,-1e4),-1)
            context=(state*weight[...,None]).sum(1)
        residual=self.head(torch.cat([embedding,context],-1));return logits+residual,residual
