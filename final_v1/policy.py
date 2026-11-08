"""Small trainable PPO residual with a continuous, causal forecast interface."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from forecast_learning.features import COLUMNS

N_FEATURES = len(COLUMNS)
EDGES = (.5, 3., 5., 7.)
SEED = 2026110501


def tensor(array, device="cpu"):
    return torch.as_tensor(array, device=device)


class Policy(nn.Module):
    def __init__(self, continuous):
        super().__init__()
        self.continuous = continuous
        self.register_buffer("center", torch.zeros(N_FEATURES))
        self.register_buffer("scale", torch.ones(N_FEATURES))
        self.encoder = nn.Sequential(nn.Linear(N_FEATURES, 48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU())
        self.timing = nn.Linear(24, 4)
        self.actor = nn.Sequential(nn.Linear(34+12+24+4+1, 40), nn.SiLU(), nn.Linear(40, 1))
        self.value_delta = nn.Sequential(nn.Linear(32+24+4+1, 32), nn.SiLU(), nn.Linear(32, 1))
        for layer in (self.actor[-1], self.value_delta[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forecast(self, raw):
        value = (raw-self.center)/self.scale
        features = torch.where(torch.isfinite(value), value.clamp(-8., 8.), 0.)
        latent = self.encoder(features)
        return latent, self.timing(latent)

    def signal(self, raw):
        if not self.continuous:
            return raw.new_zeros((*raw.shape[:-1], 29))
        known = torch.isfinite(raw).any(-1, keepdim=True)
        latent, timing = self.forecast(raw)
        return torch.cat((latent, timing.softmax(-1), known.to(raw.dtype)), -1)*known

    def forward(self, raw, context, candidates, base, critic, value):
        signal = self.signal(raw)
        c = context.unsqueeze(-2).expand(*candidates.shape[:-1], context.shape[-1])
        s = signal[..., None, None, :].expand(*candidates.shape[:-1], signal.shape[-1])
        delta = 2.*torch.tanh(self.actor(torch.cat((c, candidates, s), -1)).squeeze(-1))
        v = self.value_delta(torch.cat((critic, signal), -1)).squeeze(-1)
        return base + delta, value + v


def timing_labels(times, onsets, network):
    idx = np.searchsorted(onsets, times+EDGES[0])
    next_onset = np.r_[onsets, np.inf][idx]
    lead = next_onset-times
    healthy = network[(times*2).astype(int)-1]
    return np.where(healthy, np.searchsorted(EDGES[1:], lead, side="right")+1, 0) % 4


def supervised_loss(model, raw, labels, weights):
    return F.cross_entropy(model.forecast(raw)[1], labels, weight=weights)


def scores(model, raw):
    out = np.full((len(raw), 4), np.nan)
    keep = np.isfinite(raw).any(1)
    if keep.any():
        device = next(model.parameters()).device
        with torch.no_grad():
            out[keep] = model.forecast(tensor(raw[keep], device))[1].softmax(-1).cpu().numpy()
    return out
