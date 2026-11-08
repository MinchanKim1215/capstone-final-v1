"""단일 Transformer 계열의 단일/다중 정답 지도학습 비교."""
import torch
from torch import nn
from upgrade.model import LinkTransformer


class RiskTransformer(LinkTransformer):
    def __init__(self,features=43,multitask=False,**kwargs):
        super().__init__(features=features,**kwargs)
        # 사용하지 않는 복원용 머리는 제거합니다. 앙상블은 없습니다.
        del self.reconstruction
        self.multitask=multitask
        if multitask:self.forecast[-1]=nn.Linear(32,3)
        self.spec.update(multitask=multitask)
    def forward(self,values,observed):
        hidden=torch.zeros_like(observed)
        packed=torch.cat((values.masked_fill(~observed,0),observed.float(),hidden.float()),-1)
        e=self.encoder(self.projection(packed)+self.positional[None,:values.shape[1]])
        out=self.forecast(torch.cat((e[:,-1],e.mean(1)),-1))
        return out if self.multitask else out[:,0:1]
