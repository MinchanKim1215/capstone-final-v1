"""Frozen graph context plus optional20-second local memory, six residual risks."""
import torch
from torch import nn
from .embedding import DIM
from .features import DIM as LOCAL


class EventAdapter(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.project=nn.Sequential(nn.Linear(LOCAL,96),nn.SiLU(),nn.LayerNorm(96))
        self.memory=nn.GRU(96,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(DIM+64,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,embedding,logits,history=None):
        context=torch.zeros((len(embedding),64),device=embedding.device,dtype=embedding.dtype)
        if self.variant=='memory_event':
            if history is None:raise ValueError('The long-memory model requires causal history')
            context=self.memory(self.project(history))[0][:,-1]
        residual=self.head(torch.cat([embedding,context],-1))
        return logits+residual,residual
