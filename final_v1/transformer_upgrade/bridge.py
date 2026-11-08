"""Connect the learned heads to Final V1 while retaining the original guards."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from fast_halfsec.context import initialize
from final_v1 import control, runner
from forecast_learning.control import load_profile
from forecast_learning.features import extract
from forecast_learning.training import make_score

from .capture import route_features, watch
from .model import load_heads, fleet_inputs

MODES = ('v21', 'v21_timing', 'v21_transformer')


@contextmanager
def updated_timing(model, timing):
    layer = model.forecaster.timing
    original = {name: value.detach().clone() for name, value in layer.state_dict().items()}
    layer.load_state_dict(timing.state_dict(), strict=True)
    try:
        yield layer
    finally:
        layer.load_state_dict(original, strict=True)


class FleetGate:
    def __init__(self, capture, timing, network, threshold, network_weight):
        self.capture, self.timing = capture, timing
        self.network, self.threshold = network, threshold
        self.network_weight = network_weight
        self.scored = self.qualified = 0
        self.native_choose = runner.choose

    def choose(self, arm, planner, actions, logits, valid, local, phase, now,
               base_data=None, allow_recovery=True):
        received = base_data
        details = {}
        mode, config, early, short = load_profile()
        override = (mode, dict(config, threshold=self.threshold), early, short)

        def remember(*args):
            nonlocal received
            data = extract(*args)
            if received is None:
                received = data
            return data

        def blended(model, profile, x, physics_score):
            if received is None:
                raise ValueError('Native control reached its score gate without inputs')
            route = route_features(planner, received)
            with torch.no_grad():
                inputs = fleet_inputs(self.capture.embedding, self.capture.support, route, self.timing)
                fleet_risk = float(self.network.early_score(inputs).item())
            native_score = float(make_score(model, config, x, physics_score)[0])
            score = self.network_weight * fleet_risk + (1. - self.network_weight) * native_score
            self.scored += 1
            self.qualified += score >= self.threshold
            details.update(forecast_source='fleet_transformer', fleet_risk=fleet_risk,
                           native_v21_risk=native_score, enabled_by_fleet_score=native_score < config['threshold'])
            return np.asarray([score])

        with (patch.object(control, 'extract', remember),
              patch.object(control, 'load_profile', return_value=override),
              patch.object(control, 'make_score', blended)):
            new, detail, reason = self.native_choose(arm, planner, actions, logits, valid, local, phase,
                                                    now, base_data, allow_recovery)
        if detail is not None and reason == 'added_prevention':
            detail = dict(detail, **details)
            reason = 'fleet_transformer_prevention'
        return new, detail, reason


def run(spec, mode, payload=None, threshold=None, transport=None, max_steps=None, network_weight=1.):
    if mode not in MODES or (mode != 'v21' and payload is None):
        raise ValueError('Invalid Final V1 Transformer upgrade mode')
    initialize()
    stock = runner.parent.MODEL
    if mode == 'v21':
        row, arrays = runner.run(spec, 'v21', transport=transport, max_steps=max_steps)
    else:
        timing, network = load_heads(payload)
        with updated_timing(stock, timing) as active:
            if mode == 'v21_timing':
                row, arrays = runner.run(spec, 'v21', transport=transport, max_steps=max_steps)
            else:
                if threshold is None:
                    raise ValueError('Fleet risk needs a frozen alert threshold')
                with watch(stock) as capture:
                    gate = FleetGate(capture, active, network, threshold, network_weight)
                    with patch.object(runner, 'choose', gate.choose):
                        row, arrays = runner.run(spec, 'v21', transport=transport, max_steps=max_steps)
                    row.update(fleet_scored=gate.scored, fleet_qualified=gate.qualified)
    row['upgrade_arm'] = mode
    return row, arrays
