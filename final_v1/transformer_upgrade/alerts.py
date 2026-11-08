"""Compare next-outage warnings on identical traces and operating budgets."""
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from forecast_learning.control import load_profile
from forecast_learning.features import COLUMNS
from forecast_learning.training import make_score

from .capture import COMMAND_STEP
from .model import fleet_inputs
from .training import row_arrays

VARIANTS = ('v21', 'fleet')
SHIELD = COLUMNS.index('held_shield')
FLEET_WEIGHTS = (.25, .5, .75, 1.)


@torch.no_grad()
def cases(rows, timing, model):
    _, profile, v21, _ = load_profile()
    for row in rows:
        with row_arrays(row) as archive:
            route = archive['route']
            times = np.arange(len(route)) * COMMAND_STEP
            supported = np.isfinite(route[:, 0]) & (times + 7. <= row['duration_s'])
            x = route[supported, :len(COLUMNS)]
            candidate = fleet_inputs(archive['embedding'][supported], archive['support'][supported],
                                     route[supported], timing)
            scores = {name: np.full(len(route), np.nan) for name in VARIANTS}
            if supported.any():
                scores['v21'][supported] = make_score(v21, profile, x, x[:, SHIELD])
                scores['fleet'][supported] = model.early_score(candidate).numpy()
            label = archive['network_y']
        onsets = np.asarray([a for a, _ in row['persistent_cut_intervals'] if a > 0.])
        yield dict(times=times, scores=scores, label=label, onsets=onsets,
                   hours=row['duration_s'] / 3600., group=row['group'])


def match(case, variant, threshold):
    active = case['scores'][variant] >= threshold
    warnings = case['times'][active & ~np.r_[False, active[:-1]]]
    onsets = case['onsets']
    matched = np.zeros(len(onsets), dtype=bool)
    leads = []
    for when in warnings:
        low, high = np.searchsorted(onsets, [when + 3., when + 7.])
        for index in range(low, high):
            if not matched[index]:
                matched[index] = True
                leads.append(float(onsets[index] - when))
                break
    return len(warnings), leads


def evaluate(flights, variant, threshold, with_ap=False):
    hours = sum(row['hours'] for row in flights)
    events = sum(len(row['onsets']) for row in flights)
    matches = [match(row, variant, threshold) for row in flights]
    alerts = sum(count for count, _ in matches)
    leads = [lead for _, values in matches for lead in values]
    hits = len(leads)
    stats = dict(events=events, hits=hits, alerts=alerts, false_or_late=alerts - hits,
                 hours=hours, recall=hits / max(1, events), precision=hits / max(1, alerts),
                 false_per_h=(alerts - hits) / hours,
                 median_lead_s=float(np.median(leads)) if leads else None)
    if with_ap:
        mask = [np.isfinite(row['scores'][variant]) & (row['label'] >= 0) for row in flights]
        scores = np.concatenate([row['scores'][variant][valid] for row, valid in zip(flights, mask)])
        labels = np.concatenate([row['label'][valid] >= 2 for row, valid in zip(flights, mask)])
        stats.update(average_precision=float(average_precision_score(labels, scores)),
                     ranking_rows=len(scores), ranking_positives=int(labels.sum()))
    return stats


def blend(flights, weight):
    return [dict(row, scores=dict(row['scores'], fleet=(weight * row['scores']['fleet']
                                                     + (1. - weight) * row['scores']['v21'])))
            for row in flights]


def calibrate_one(flights, variant, cap):
    values = np.concatenate([row['scores'][variant] for row in flights])
    finite = values[np.isfinite(values)]
    grid = np.unique(np.r_[np.nextafter(finite.max(), np.inf),
                           np.quantile(finite, np.linspace(0., 1., 161))])
    feasible = []
    for threshold in grid:
        stats = evaluate(flights, variant, threshold)
        if stats['false_per_h'] <= cap:
            feasible.append((float(threshold), stats))
    threshold, _ = max(feasible, key=lambda pair: (pair[1]['hits'], pair[1]['precision'],
                                                  -pair[1]['false_per_h'], pair[0]))
    return dict(threshold=threshold, **evaluate(flights, variant, threshold, True))


def calibrate(flights, cap):
    options = [dict(network_weight=weight, **calibrate_one(blend(flights, weight), 'fleet', cap))
               for weight in FLEET_WEIGHTS]
    chosen = max(options, key=lambda row: (row['hits'], row['precision'],
                                          row['average_precision'], -row['network_weight']))
    return dict(v21=calibrate_one(flights, 'v21', cap), fleet=chosen, blend_candidates=options)


def compare(flights, thresholds):
    raw = flights
    flights = blend(flights, thresholds['fleet']['network_weight'])
    fleet_only = thresholds['blend_candidates'][-1]['threshold']
    thresholds = {name: thresholds[name]['threshold'] for name in VARIANTS}
    summary = {name: evaluate(flights, name, thresholds[name], True) for name in VARIANTS}
    groups = defaultdict(lambda: np.zeros(3))
    for row in flights:
        groups[row['group']] += [len(row['onsets']),
                                len(match(row, 'v21', thresholds['v21'])[1]),
                                len(match(row, 'fleet', thresholds['fleet'])[1])]
    values = np.stack(list(groups.values()))
    rng = np.random.default_rng(20261108)
    samples = values[rng.integers(0, len(values), size=(5000, len(values)))].sum(axis=1)
    events, v21, fleet = samples[samples[:, 0] > 0].T
    limits = np.quantile(100. * (fleet - v21) / events, [.025, .975])
    return dict(modes=summary, standalone_network=evaluate(raw, 'fleet', fleet_only, True),
                groups=len(groups), recall_delta_pp=100. * (
                    summary['fleet']['recall'] - summary['v21']['recall']),
                recall_delta_ci95_pp=limits.tolist())
