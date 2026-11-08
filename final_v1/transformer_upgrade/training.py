"""Train the V7 timing layer first and then the fleet-wide forecast."""
from copy import deepcopy
from time import perf_counter
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from fast_halfsec.context import initialize
from final_v1 import runner
from forecast_rebuild.contracts import BASELINE, now, read, sha, write

from . import ARTIFACTS, ROOT
from .collection import MANIFEST, PLAN, status
from .model import FleetForecaster, fleet_inputs

TIMING_FILE = ARTIFACTS / 'models/timing.pt'
NETWORK_FILE = ARTIFACTS / 'models/network.pt'
FIT_FILE = ARTIFACTS / 'training.json'
TIMING_EPOCHS = 24
NETWORK_EPOCHS = 60
SEED = 20261108


def commit(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.pt')
    torch.save(payload, pending)
    os.replace(pending, path)
    return sha(path)


def cpu_weights(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def row_arrays(row):
    path = row['archive_path']
    if sha(path) != row['archive_sha256']:
        raise ValueError(f"Training archive changed: {row['spec']['region']}")
    return np.load(path, allow_pickle=False)


def link_data(rows, device):
    x, y = [], []
    for row in rows:
        with row_arrays(row) as archive:
            labels = archive['link_y']
            good = labels >= 0
            x.append(archive['embedding'][good])
            y.append(labels[good])
    return (torch.as_tensor(np.concatenate(x), device=device),
            torch.as_tensor(np.concatenate(y), dtype=torch.long, device=device))


def early_logits(logits):
    other = torch.cat((logits[:, :6], logits[:, 14:]), dim=-1)
    return logits[:, 6:14].logsumexp(dim=-1) - other.logsumexp(dim=-1)


@torch.no_grad()
def link_scores(head, x, y):
    logits = head(x)
    p = logits.softmax(dim=-1)
    early = (y >= 6) & (y < 14)
    risk = p[:, 6:14].sum(dim=-1)
    return dict(rows=len(y), early_cases=int(early.sum().item()),
                early_ap=float(average_precision_score(early.cpu().numpy(), risk.cpu().numpy())),
                five_ap=float(average_precision_score((y < 10).cpu().numpy(), p[:, :10].sum(-1).cpu().numpy())),
                early_brier=float((risk - early.float()).square().mean().item()),
                timing_nll=float(F.cross_entropy(logits, y).item()))


def fit_timing(train, calibration, device, pins):
    if TIMING_FILE.exists():
        saved = torch.load(TIMING_FILE, map_location='cpu', weights_only=True)
        if saved['pins'] != pins:
            raise ValueError('Timing checkpoint inputs changed')
        return saved
    status('loading_link_training_data')
    x, y = link_data(train, device)
    x_val, y_val = link_data(calibration, device)
    initialize()
    old = deepcopy(runner.parent.MODEL.forecaster.timing).to(device).eval().requires_grad_(False)
    head = deepcopy(old).requires_grad_(True)
    before = link_scores(old, x_val, y_val)
    if not (0 < before['early_cases'] < before['rows']) or not ((y >= 6) & (y < 14)).any():
        raise ValueError('Missing early link labels')
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TIMING_EPOCHS, eta_min=.00025)
    generator = torch.Generator(device=device).manual_seed(SEED)
    pos_weight = torch.tensor(2., device=device)
    best, backup, history = None, None, []
    for epoch in range(1, TIMING_EPOCHS + 1):
        for indices in torch.randperm(len(x), device=device, generator=generator).split(4096):
            values, labels = x[indices], y[indices]
            logits = head(values)
            early = ((labels >= 6) & (labels < 14)).float()
            with torch.no_grad():
                reference = old(values).softmax(dim=-1)
            loss = (F.cross_entropy(logits, labels)
                    + .35 * F.binary_cross_entropy_with_logits(early_logits(logits), early, pos_weight=pos_weight)
                    + .08 * F.kl_div(logits.log_softmax(dim=-1), reference, reduction='batchmean'))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % 3 == 0:
            stats = link_scores(head, x_val, y_val)
            history.append(dict(epoch=epoch, loss=float(loss.detach()), **stats))
            candidate = (stats['early_ap'], -stats['early_brier'], epoch, stats)
            if backup is None or candidate[:2] > backup[0][:2]:
                backup = candidate, cpu_weights(head)
            if stats['five_ap'] >= before['five_ap'] - .01 and (best is None or candidate[:2] > best[0][:2]):
                best = candidate, cpu_weights(head)
            print(f"timing epoch {epoch}/{TIMING_EPOCHS} validation early AP={stats['early_ap']:.4f}", flush=True)
        status('training_v7_timing', epoch=epoch, total=TIMING_EPOCHS, train_links=len(x), device=str(device))
    selected, weights = best if best is not None else backup
    _, _, epoch, after = selected
    delta = sum(float((value - cpu_weights(old)[name]).square().sum()) for name, value in weights.items()) ** .5
    if not np.isfinite(delta) or delta <= 0.:
        raise ValueError('V7 timing parameters did not update')
    saved = dict(pins=pins, timing=weights, old_calibration=before, calibration=after, selected_epoch=epoch,
                 five_ap_guard_met=best is not None, train_rows=len(x), history=history,
                 delta_l2=delta, trained_at=now(), device=str(device))
    commit(TIMING_FILE, saved)
    status('v7_timing_trained', epoch=epoch, train_links=len(x), before_ap=before['early_ap'],
           after_ap=after['early_ap'])
    return saved


@torch.no_grad()
def network_data(rows, timing, device):
    x, y = [], []
    for row in rows:
        with row_arrays(row) as archive:
            labels, routes = archive['network_y'], archive['route']
            good = (labels >= 0) & np.isfinite(routes[:, 0])
            x.append(fleet_inputs(archive['embedding'][good], archive['support'][good], routes[good], timing))
            y.append(labels[good])
    return (torch.cat(x).to(device), torch.as_tensor(np.concatenate(y), dtype=torch.long, device=device))


@torch.no_grad()
def network_scores(model, x, y):
    p = model.early_score(x)
    early = y >= 2
    return dict(rows=len(y), early_cases=int(early.sum().item()),
                early_ap=float(average_precision_score(early.cpu().numpy(), p.cpu().numpy())),
                early_brier=float((p - early.float()).square().mean().item()))


def fit_network(train, calibration, timing_state, device, pins):
    if NETWORK_FILE.exists():
        saved = torch.load(NETWORK_FILE, map_location='cpu', weights_only=True)
        if saved['pins'] != pins:
            raise ValueError('Fleet checkpoint inputs changed')
        return saved
    status('loading_fleet_training_data')
    timing = nn.Linear(32, 17).eval()
    timing.load_state_dict(timing_state)
    x, y = network_data(train, timing, device)
    x_val, y_val = network_data(calibration, timing, device)
    if not (y >= 2).any() or not (y_val >= 2).any():
        raise ValueError('Missing early fleet labels')
    model = FleetForecaster().to(device)
    model.normalize_from(x)
    class_count = torch.bincount(y, minlength=4)
    weights = (class_count[0] / class_count.clamp_min(1)).sqrt().clamp(1., 10.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0012, weight_decay=.008)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NETWORK_EPOCHS, eta_min=.00008)
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    history, best = [], None
    for epoch in range(1, NETWORK_EPOCHS + 1):
        model.train()
        for indices in torch.randperm(len(x), device=device, generator=generator).split(1024):
            logits = model(x[indices])
            loss = F.cross_entropy(logits, y[indices], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % 4 == 0:
            stats = network_scores(model.eval(), x_val, y_val)
            history.append(dict(epoch=epoch, loss=float(loss.detach()), **stats))
            rank = (stats['early_ap'], -stats['early_brier'])
            if best is None or rank > best[0]:
                best = rank, epoch, stats, cpu_weights(model)
            print(f"fleet epoch {epoch}/{NETWORK_EPOCHS} validation early AP={stats['early_ap']:.4f}", flush=True)
        status('training_fleet_forecast', epoch=epoch, total=NETWORK_EPOCHS, train_ticks=len(x), device=str(device))
    _, epoch, calibration_stats, state = best
    saved = dict(pins=pins, network=state, calibration=calibration_stats, selected_epoch=epoch,
                 class_counts=class_count.cpu().tolist(), history=history, train_rows=len(x),
                 trained_at=now(), device=str(device))
    commit(NETWORK_FILE, saved)
    return saved


def fit(manifest):
    pins = dict(protocol=sha(PLAN), data=sha(MANIFEST), baseline=sha(BASELINE),
                trainer=sha(__file__), architecture=sha(ROOT / 'model.py'))
    train, calibration = ([row for row in manifest['records'] if row['role'] == role]
                          for role in ('train', 'calibration'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    start = perf_counter()
    timing = fit_timing(train, calibration, device, pins)
    network = fit_network(train, calibration, timing['timing'], device,
                          dict(pins, timing=sha(TIMING_FILE)))
    models = dict(timing=sha(TIMING_FILE), network=sha(NETWORK_FILE))
    if FIT_FILE.exists():
        saved = read(FIT_FILE)
        if saved['models'] != models or saved['pins'] != pins:
            raise ValueError('Trained checkpoints changed')
    else:
        write(FIT_FILE, dict(pins=pins, models=models, wall_s=perf_counter() - start,
                            completed_at=now(), device=str(device),
                            timing={name: value for name, value in timing.items() if name != 'timing'},
                            network={name: value for name, value in network.items() if name != 'network'}))
    status('models_trained', timing_links=timing['train_rows'], fleet_ticks=network['train_rows'],
           device=str(device))
    return dict(timing=timing['timing'], network=network['network'])
