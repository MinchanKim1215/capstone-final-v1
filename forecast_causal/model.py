"""Frozen physical/RF expert plus matched RF and measured-network GRUs."""
import torch
from torch import nn
class CausalAdapter(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.rf_project=nn.Sequential(nn.Linear(480,96),nn.SiLU(),nn.LayerNorm(96));self.rf_memory=nn.GRU(96,64,batch_first=True)
        self.network_project=nn.Sequential(nn.Linear(186,96),nn.SiLU(),nn.LayerNorm(96));self.network_memory=nn.GRU(96,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(609,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,6));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,embedding,baseline_logits,rf_history,network_history):
        r=self.rf_memory(self.rf_project(rf_history))[0][:,-1];n=torch.zeros_like(r)
        if self.variant=='gru_network':n=self.network_memory(self.network_project(network_history))[0][:,-1]
        residual=self.head(torch.cat([embedding,r,n],-1));return baseline_logits+residual,residual
