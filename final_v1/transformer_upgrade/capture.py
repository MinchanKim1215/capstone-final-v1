"""Capture exactly the causal inputs seen by the frozen V7 controller."""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from adaptive_joint_expanded.targets import timing_labels
from fast_halfsec.context import initialize
from final_v1 import runner
from forecast_avoidance.controller import TIMES, fragile_margin
from forecast_learning.features import COLUMNS, extract

BACKUP_FRAMES = np.flatnonzero(np.isin(TIMES, [3., 4., 5., 6.]))
N_GEOMETRY = len(COLUMNS) + 2 * len(BACKUP_FRAMES)
RF_STEP = .5
COMMAND_STEP = 3.


def route_features(planner, base_data):
    x, _, _, graphs, _ = base_data
    n_search = planner.design.n_search
    sample = graphs[:, BACKUP_FRAMES]
    backup = fragile_margin(sample.reshape(-1, *sample.shape[-2:]), n_search)
    return np.r_[x, np.clip(backup, -150., 150.)].astype(np.float32)


class Capture:
    def __init__(self, model, store):
        self.stock_policy = model.policy
        self.store = store
        self.embedding = self.support = None
        self.embeddings, self.masks, self.routes = [], [], []

    def head_hook(self, module, inputs, output):
        self.embedding = inputs[0].detach().cpu().numpy().reshape(*self.support.shape, 32)

    def policy(self, observation, mode):
        self.support = observation['support'].cpu().numpy()
        self.embedding = None
        value = self.stock_policy(observation, mode)
        if self.embedding is None or self.embedding.shape[:-1] != self.support.shape:
            raise ValueError('V7 link representation and observation mask disagree')
        if self.store:
            self.embeddings.append(self.embedding)
            self.masks.append(self.support)
            self.routes.append(np.full(N_GEOMETRY, np.nan, np.float32))
        return value

    def extract(self, planner, actions, local, phase):
        base_data = extract(planner, actions, local, phase)
        self.routes[-1] = route_features(planner, base_data)
        return base_data


@contextmanager
def watch(model, store=False):
    capture = Capture(model, store)
    hook = model.forecaster.timing.register_forward_hook(capture.head_hook)
    try:
        with patch.object(model, 'policy', capture.policy):
            yield capture
    finally:
        hook.remove()


def label_network(network, cuts, times):
    frames = np.rint(times / RF_STEP).astype(int)
    valid = (frames > 0) & (frames + 14 < len(network))
    valid &= network[np.maximum(0, frames - 1)]
    onsets = np.asarray([a for a, _ in cuts if a > 0.])
    next_index = np.searchsorted(onsets, times)
    delay = np.r_[onsets, np.inf][next_index] - times
    # 0 = no near onset, 1 = under 3s, 2 = 3–5s, 3 = 5–7s.
    labels = np.select([delay < 3., delay < 5., delay < 7.], [1, 2, 3], default=0)
    return np.where(valid, labels, -1).astype(np.int8)


def collect_flight(spec, max_steps=None):
    initialize()
    model = runner.parent.MODEL
    with watch(model, store=True) as capture, patch.object(runner, 'extract', capture.extract):
        row, data = runner.run(spec, 'v7', max_steps=max_steps)
    xy = np.asarray(capture.embeddings, dtype=np.float32)
    support = np.asarray(capture.masks, dtype=bool)
    routes = np.asarray(capture.routes, dtype=np.float32)
    np.testing.assert_array_equal(routes[:, :len(COLUMNS)], data['forecast_features'])
    frames = np.arange(len(xy), dtype=int) * round(COMMAND_STEP / RF_STEP)
    if len(frames) != len(data['commands']) or len(frames) * 6 != len(data['network']):
        raise ValueError('Forecast capture and RF samples disagree')
    labels, valid = timing_labels(data['links'], frames[1:], xy.shape[2] - xy.shape[1])
    link_y = np.full(support.shape, -1, np.int8)
    link_y[1:] = np.where(valid & support[1:], labels, -1)
    network_y = label_network(data['network'], row['persistent_cut_intervals'], frames * RF_STEP)
    return row, dict(embedding=xy, support=support, route=routes, link_y=link_y,
                     network_y=network_y, network=data['network'], links=data['links'])
