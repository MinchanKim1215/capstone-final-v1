"""한 종류의 시계열 Transformer: 마스크 복원 사전학습과 장애 예측을 공유합니다.

TST의 '과거 관측 일부를 가리고 복원한 후 미세조정' 원리를 적용한 변형입니다.
원 논문의 정확한 구조 재현 또는 UAV 분야에서 성능 보장을 주장하지 않습니다.
"""
import math

import numpy as np
import torch
from torch import nn


class LinkTransformer(nn.Module):
    def __init__(self, features=28, d_model=64, nhead=4, layers=2, feedforward=128, dropout=.1, length=40):
        super().__init__()
        self.spec = dict(features=features, d_model=d_model, nhead=nhead, layers=layers,
                         feedforward=feedforward, dropout=dropout, length=length)
        # 실제 관측 여부와 사전학습에서 인위적으로 가린 여부를 구분합니다.
        self.projection = nn.Linear(3*features, d_model)
        t = torch.arange(length).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.)/d_model))
        positional = torch.zeros(length, d_model)
        positional[:, 0::2] = torch.sin(t*frequency)
        positional[:, 1::2] = torch.cos(t*frequency)
        self.register_buffer('positional', positional)
        block = nn.TransformerEncoderLayer(d_model, nhead, feedforward, dropout,
                                            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.forecast = nn.Sequential(nn.Linear(2*d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        self.reconstruction = nn.Linear(d_model, features)

    def forward(self, values, observed, hidden=None, reconstruct=False):
        if hidden is None:
            hidden = torch.zeros_like(observed)
        packed = torch.cat((values.masked_fill(hidden | ~observed, 0), (observed & ~hidden).float(), hidden.float()), -1)
        encoded = self.encoder(self.projection(packed)+self.positional[None, :values.shape[1]])
        if reconstruct:
            return self.reconstruction(encoded)
        return self.forecast(torch.cat((encoded[:, -1], encoded.mean(dim=1)), -1)).squeeze(-1)


def contiguous_hidden(observed, rng, ratio=.15, mean_length=3):
    """연속 가림의 시작값: TST §3.3의 15%, 평균 3구간. 보편 최적값은 아닙니다."""
    mask = np.empty(tuple(observed.shape), bool)
    mask[:, 0] = rng.random(mask[:, 0].shape) < ratio
    enter = ratio/(mean_length*(1-ratio))
    leave = 1/mean_length
    for k in range(1, mask.shape[1]):
        r = rng.random(mask[:, k].shape)
        mask[:, k] = np.where(mask[:, k-1], r >= leave, r < enter)
    return torch.from_numpy(mask) & observed


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
