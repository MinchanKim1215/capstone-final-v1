"""Variable-fleet shared relay policy, timing predictor and pre-mission selector."""
import numpy as np
import torch
from torch import nn
from solo.model import RiskTransformer
from .contracts import PILOT, DEPLOYMENTS,PUBLIC_CONTEXT_DIM
from .targets import probabilities


class TimeForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RiskTransformer()
        self.timing = nn.Linear(32,17)

    def encode(self, values, masks, intents, geometry):
        shape = values.shape[:-2]
        z,_ = self.backbone.encode_links(values.reshape(-1,40,48), masks.reshape(-1,40,48),
                                         intents.reshape(-1,40,9), geometry.reshape(-1,46))
        return z.reshape(*shape,32), self.timing(z).reshape(*shape,17)


class Controller(nn.Module):
    def __init__(self, warm_start=True):
        super().__init__()
        self.forecaster = TimeForecaster()
        if warm_start:
            ck = torch.load(PILOT, map_location='cpu', weights_only=False)
            self.forecaster.load_state_dict(ck['model'], strict=True)
        # The old four-head layer is retained only for exact encoder transfer.
        # It is neither used nor optimized in this new next-event experiment.
        for p in self.forecaster.backbone.pred_head.parameters():
            p.requires_grad_(False)
        self.actor = nn.Sequential(nn.Linear(92,64), nn.Tanh(), nn.Linear(64,32),
                                   nn.Tanh(), nn.Linear(32,1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.critic = nn.Sequential(nn.Linear(32,64), nn.Tanh(), nn.Linear(64,32), nn.Tanh(), nn.Linear(32,1))
        self.selector = nn.Sequential(nn.Linear(PUBLIC_CONTEXT_DIM,32), nn.Tanh(), nn.Linear(32,16), nn.Tanh(), nn.Linear(16,4))
        nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)
        self.selector_value = nn.Sequential(nn.Linear(PUBLIC_CONTEXT_DIM,32), nn.Tanh(), nn.Linear(32,1))

    def select(self, public_context, feasible):
        if not feasible.bool().any(-1).all():
            raise ValueError('No deployment satisfies the public budget')
        return self.selector(public_context).masked_fill(~feasible.bool(), -1e9)

    def policy(self, observation, prediction_mode='joint'):
        if prediction_mode not in ('joint','fixed','none'):
            raise ValueError('Unknown predictor ablation')
        b = observation
        mask = b['action_mask'].bool()
        if not mask.any(-1).all():
            raise ValueError('An active relay has no feasible move')
        if prediction_mode == 'none':
            # Same actor architecture, but no forecast/embedding information.
            shape = b['support'].shape
            z = torch.zeros((*shape,32), device=b['context'].device)
            timing = torch.zeros((*shape,17), device=z.device)
            risk = torch.zeros((*shape,4), device=z.device)
        else:
            z,timing = self.forecaster.encode(b['values'], b['masks'], b['intents'], b['geometry'])
            risk = probabilities(timing)
            if prediction_mode == 'fixed':
                z,timing,risk = z.detach(),timing.detach(),risk.detach()
        support = b['support'].to(z.dtype)
        risk = risk*support[...,None]
        embedding = (z*support[...,None]).sum(-2)/support.sum(-1,keepdim=True).clamp_min(1)
        mean = risk.sum(-2)/support.sum(-1,keepdim=True).clamp_min(1)
        maximum = risk.max(-2).values
        design = b['design'].unsqueeze(-2).expand(*b['context'].shape[:-1],5)
        count=(b['design'][...,0]*10-1).clamp_min(1)
        while count.ndim<support.ndim:count=count.unsqueeze(-1)
        coverage=support.sum(-1,keepdim=True)/count
        context = torch.cat([b['context'],design,embedding,mean,maximum,coverage],-1)
        candidates = b['candidates']
        expanded = context.unsqueeze(-2).expand(*candidates.shape[:-1],context.shape[-1])
        residual = self.actor(torch.cat([expanded,candidates],-1)).squeeze(-1)
        # Raw physical gains are stored; forecast-dependent weights are rebuilt
        # on every forward, including PPO. No cached neural features enter PPO.
        delta = (b['gain']*(risk[...,0]+risk[...,2]).unsqueeze(-2)).sum(-1)
        denominator = (b['design'][...,0]*10-1).clamp_min(1)
        while denominator.ndim < delta.ndim:
            denominator = denominator.unsqueeze(-1)
        delta = delta/denominator
        old_raw=.15*b['gain'].sum(-1)/denominator*10
        new_weight=(old_raw+delta*10).clamp(-30,30)
        old_weight=b['candidates'][...,5]*10
        extra=torch.where(b['candidates'][...,0:1,9]<.999,
                          1.5*delta,.15*(torch.tanh(new_weight/3)-torch.tanh(old_weight/3)))
        logits = b['teacher'] + extra + residual
        return logits.masked_fill(~mask,-1e9), timing

    def value(self, state):
        return self.critic(state).squeeze(-1)


def tensor_observation(observation, device='cpu'):
    return {key:torch.as_tensor(value, device=device) for key,value in observation.items()}


def optimizer(model, config):
    groups = [dict(params=model.actor.parameters(), lr=config['actor_lr']),
              dict(params=model.critic.parameters(), lr=config['critic_lr']),
              dict(params=list(model.selector.parameters())+list(model.selector_value.parameters()), lr=config['selector_lr'])]
    if config.get('prediction_mode','joint') == 'joint':
        groups.extend([dict(params=model.forecaster.backbone.encoder_parameters(), lr=config['encoder_lr']),
                       dict(params=model.forecaster.timing.parameters(), lr=config['timing_lr'])])
    return torch.optim.Adam(groups, eps=1e-5)
