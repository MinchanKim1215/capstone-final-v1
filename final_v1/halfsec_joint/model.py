"""Warm-start the installed forecasts and rank half-second safe moves."""

import torch
from torch import nn

from adaptive_joint_expanded.model import TimeForecaster
from fast_halfsec.actor import ActorCritic, STATE_DIM
from fast_halfsec.features import N_FEATURES
from final_v1.transformer_upgrade.capture import N_GEOMETRY
from final_v1.transformer_upgrade.model import FleetForecaster, N_INPUT, fleet_inputs


OBSERVATIONS = ('values', 'masks', 'intents', 'geometry', 'support', 'route', 'native')


class JointPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecaster = TimeForecaster()
        self.forecaster.backbone.pred_head.requires_grad_(False)
        self.fleet = FleetForecaster()
        self.moves = ActorCritic()
        self.moves.encoder[0] = nn.Linear(STATE_DIM + N_INPUT - N_GEOMETRY, 32)
        self.register_buffer('fleet_weight', torch.tensor(.75))

    def forecast(self, values, masks, intents, geometry, support, route, native):
        z = torch.zeros((*support.shape, 32), device=values.device)
        timing = torch.zeros((*support.shape, 17), device=values.device)
        if support.any():
            z[support], timing[support] = self.forecaster.encode(
                values[support], masks[support], intents[support], geometry[support])
        features = fleet_inputs(z, support, route, self.forecaster.timing)
        fleet_logits = self.fleet(features)
        available = support.any(dim=(-2, -1))
        fleet_risk = fleet_logits.softmax(dim=-1)[..., 2:].sum(dim=-1)
        early = torch.where(available, self.fleet_weight * fleet_risk
                            + (1. - self.fleet_weight) * native[..., 0], native[..., 0])
        context = torch.cat((early[..., None], native[..., 1:2],
                             features[..., N_GEOMETRY:].clamp(-6., 6.)), dim=-1)
        return timing, fleet_logits, context

    def policy(self, state, options, priors, valid, context):
        return self.moves(torch.cat((state[..., :N_FEATURES], context), dim=-1),
                          options, priors, valid)


def initial_model():
    from fast_halfsec.context import initialize
    from final_v1 import runner
    from final_v1.transformer_upgrade.release import installed

    initialize()
    payload, freeze = installed()
    model = JointPolicy()
    model.forecaster.load_state_dict(runner.parent.MODEL.forecaster.state_dict())
    model.forecaster.timing.load_state_dict(payload['timing'])
    model.fleet.load_state_dict(payload['network'])
    profile = freeze['thresholds']['fleet']
    model.fleet_weight.fill_(profile['network_weight'])
    return model.eval(), float(profile['threshold'])
