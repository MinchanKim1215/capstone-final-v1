"""Isolated prediction study: causal 1-second inputs and next-outage timing.

An RF frame q describes [q/2, (q+1)/2) and becomes available at
(q+1)/2. Predictions at tau use frames strictly before 2*tau.
An outage starts at the beginning of its first bad frame and must last
at least two frames. The 16 classes cover [0,.5), ... [7.5,8) seconds;
class 16 means no qualifying onset within [0,8). Unknown tails are masked.
"""
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]/'drone_forecast_joint_training'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))


def timing_labels(links, ends):
    up = np.asarray(links, bool)
    ends = np.asarray(ends, int)
    if up.ndim != 3 or up.shape[1:] != (8, 8) or np.any(ends < 1):
        raise ValueError('Expected RF links [frames,8,8] and positive history endpoints')
    if np.any(ends > len(up)):
        raise ValueError('Input endpoint exceeds available frames')
    bad = ~up
    start = np.zeros_like(up)
    start[1:-1] = bad[1:-1] & up[:-2] & bad[2:]
    pair = bad[:-1] & bad[1:]
    y = np.full((len(ends), 5, 8), 16, np.int64)
    valid = np.zeros_like(y, bool)
    legacy = np.zeros((*y.shape, 4), np.float32)
    next_frame = np.full_like(y, -1)
    for t, end in enumerate(ends):
        if end+16 >= len(up):
            continue
        eligible = up[end-1, 3:8].copy()
        eligible[np.arange(5), np.arange(3, 8)] = False
        future = start[end:end+16, 3:8]
        any_start = future.any(0)
        first = future.argmax(0)
        y[t] = np.where(any_start, first, 16)
        next_frame[t] = np.where(any_start, end+first, -1)
        valid[t] = eligible
        legacy[t, ..., 0] = pair[end:end+9, 3:8].any(0)
        legacy[t, ..., 1] = start[end+6:end+9, 3:8].any(0)
        legacy[t, ..., 2] = start[end+6:end+15, 3:8].any(0)
        legacy[t, ..., 3] = start[end:end+15, 3:8].any(0)
    return dict(label=y, valid=valid, legacy=legacy, next_frame=next_frame)


def causal_support(values, masks, ends):
    ends = np.asarray(ends, int)
    x, m = values[ends-1], masks[ends-1]
    support = m[..., 41] & m[..., 46] & (x[..., 46] > .5) & (ends[:, None, None] >= 40)
    support[:, np.arange(5), np.arange(3, 8)] = False
    return support


def windows(values, masks, intents, ends, rows):
    t, r, p = np.asarray(rows, int).T
    frame = np.asarray(ends)[t, None] + np.arange(-40, 0)
    if np.any(frame < 0) or np.any(frame >= np.asarray(ends)[t, None]):
        raise ValueError('Insufficient history or future-observation leak')
    return tuple(array[frame, r[:, None], p[:, None]] for array in [values, masks, intents])


def next_targets(label):
    y = np.asarray(label)
    return np.stack([y < 10, (y >= 6) & (y <= 8), (y >= 6) & (y <= 14), y <= 14], -1)


def next_probabilities(probability):
    p = np.asarray(probability)
    return np.stack([p[..., :10].sum(-1), p[..., 6:9].sum(-1),
                     p[..., 6:15].sum(-1), p[..., :15].sum(-1)], -1)


def event_metrics(probability, support, links, times, threshold=.5, cadence=1):
    from solo.metrics import events_from_links
    times = np.asarray(times, float)
    duration = len(links)*.5
    use = np.flatnonzero((times >= 21) & (times <= duration-8) & (np.mod(times, cadence) == 0))
    first = correct = hits = total = 0
    for (r, p), events in events_from_links(links).items():
        active = False
        alarms = []
        for k in use:
            score = probability[k, r, p]
            new = bool(support[k, r, p] and np.isfinite(score) and score >= (threshold*.5 if active else threshold))
            if new and not active:
                alarms.append(times[k])
            active = new
        alarms = np.asarray(alarms)
        first += len(alarms)
        correct += sum(bool(((events >= alarm) & (events <= alarm+8)).any()) for alarm in alarms)
        eligible = events[(events >= 24) & (events <= duration-8)]
        total += len(eligible)
        hits += sum(bool(((alarms >= onset-8) & (alarms <= onset-3)).any()) for onset in eligible)
    # The same observation interval is the exposure for both prediction cadences.
    hours = max(0., duration-8-21)*5/3600
    return dict(first_alerts=first, correct_first_alerts=correct, eligible_events=total,
                event_hits_3s=hits, drone_hours=hours, threshold=threshold, cadence_s=cadence)


def make_model(checkpoint):
    import torch
    from torch import nn
    from solo.model import RiskTransformer

    class TimeForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = RiskTransformer()
            self.timing = nn.Linear(32, 17)
            nn.init.normal_(self.timing.weight, std=.01)
            nn.init.zeros_(self.timing.bias)

        def forward(self, values, masks, intents, geometry):
            z, legacy = self.backbone.encode_links(values, masks, intents, geometry)
            return self.timing(z), legacy

    model = TimeForecaster()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['format'] != 'standalone-transformer-pretrain-v5':
        raise ValueError('Expected the pinned supervised warm-start checkpoint')
    model.backbone.load_state_dict(saved['model'])
    return model
