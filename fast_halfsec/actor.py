"""Trainable residual policy over the physics-checked action set."""
import numpy as np
import torch
from torch import nn

from .features import MAX_OPTIONS, N_CANDIDATE, N_FEATURES

STATE_DIM = N_FEATURES + 2


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(STATE_DIM, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(32+N_CANDIDATE, 32), nn.Tanh(), nn.Linear(32, 1))
        self.critic = nn.Linear(32, 1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, state, options, priors, valid):
        hidden = self.encoder(state)
        paired = torch.cat((hidden.unsqueeze(-2).expand(*options.shape[:-1], 32), options), dim=-1)
        residual = self.actor(paired).squeeze(-1)
        return (priors + residual).masked_fill(~valid, -1e9), self.critic(hidden).squeeze(-1)


def inputs(raw, early, short, options):
    state = np.r_[np.clip(raw / 30., -6., 6.), early, short].astype(np.float32)
    candidates = np.zeros((MAX_OPTIONS, N_CANDIDATE), dtype=np.float32)
    priors = np.zeros(MAX_OPTIONS, dtype=np.float32)
    valid = np.arange(MAX_OPTIONS) < len(options)
    for i, option in enumerate(options):
        candidates[i] = option.values
        priors[i] = np.clip(option.physical_value / 7., -4., 4.)
    return state, candidates, priors, valid


def select(model, state, explore, rng):
    with torch.no_grad():
        logits, value = model(*(torch.from_numpy(part[None]) for part in state))
        probabilities = torch.softmax(logits[0], dim=0).numpy()
        action = int(rng.choice(len(probabilities), p=probabilities.astype(float)/probabilities.sum(dtype=float))) if explore else int(logits[0].argmax())
        logp = float(torch.log_softmax(logits[0], dim=0)[action])
    return action, float(value[0]), logp
