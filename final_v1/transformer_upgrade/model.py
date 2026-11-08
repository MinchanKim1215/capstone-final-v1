"""Fleet risk derived from causal V7 link states and projected route geometry."""
import torch
from torch import nn

from adaptive_joint_expanded.targets import probabilities

from .capture import N_GEOMETRY

N_INPUT = N_GEOMETRY + 2 * (2 * (32 + 4) + 1)


def fleet_inputs(embedding, support, route, timing):
    z = torch.as_tensor(embedding, dtype=torch.float32)
    known = torch.as_tensor(support, dtype=torch.bool)
    geometry = torch.as_tensor(route, dtype=torch.float32)
    values = torch.cat((z, probabilities(timing(z))), dim=-1)
    ns = z.shape[-2] - z.shape[-3]
    parts = [geometry]
    for indices in (slice(None, ns), slice(ns, None)):
        mask = known[..., indices]
        selected = values[..., indices, :]
        count = mask.sum(dim=(-2, -1), keepdim=False)[..., None]
        mean = (selected * mask[..., None]).sum(dim=(-3, -2)) / count.clamp_min(1)
        peak = selected.masked_fill(~mask[..., None], -torch.inf).amax(dim=(-3, -2))
        peak = torch.where(count > 0, peak, 0.)
        parts.extend((mean, peak, count / (mask.shape[-1] * mask.shape[-2])))
    return torch.cat(parts, dim=-1)


class FleetForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('center', torch.zeros(N_INPUT))
        self.register_buffer('scale', torch.ones(N_INPUT))
        self.head = nn.Sequential(nn.Linear(N_INPUT, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(.10),
                                  nn.Linear(96, 48), nn.SiLU(), nn.Dropout(.10), nn.Linear(48, 4))

    def forward(self, inputs):
        filled = torch.where(torch.isfinite(inputs), inputs, self.center)
        return self.head(((filled - self.center) / self.scale).clamp(-8., 8.))

    def normalize_from(self, train):
        valid = torch.isfinite(train)
        denominator = valid.sum(dim=0).clamp_min(1)
        center = torch.where(valid, train, 0.).sum(dim=0) / denominator
        variance = torch.where(valid, (train - center).square(), 0.).sum(dim=0) / denominator
        self.center.copy_(center)
        self.scale.copy_(variance.sqrt().clamp_min(.05))

    def early_score(self, inputs):
        return self(inputs).softmax(dim=-1)[..., 2:].sum(dim=-1)


def load_heads(payload):
    timing = nn.Linear(32, 17)
    timing.load_state_dict(payload['timing'], strict=True)
    network = FleetForecaster()
    network.load_state_dict(payload['network'], strict=True)
    return timing.eval(), network.eval()
