"""Supervised warm start and matched on-policy PPO updates."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from forecast_learning.training import load
from . import ARTIFACTS
from .policy import Policy, SEED, supervised_loss, tensor, timing_labels
from .protocol import FLIGHTS_PER_UPDATE, UPDATES, V21, read, sha, status, write
from .runner import initialize, run

MODES = ("ppo_base", "ppo_continuous")
EPOCHS = 8


def safe_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    torch.save(value, pending)
    pending.replace(path)


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_train():
    sources = read(V21/"train_manifest.json")["records"]
    if len(sources) != 245:
        raise ValueError("Supervised train map count changed")
    raw, labels = [], []
    for source in sources:
        f = load(source)
        if f.row["contract"]["scope"] != "train":
            raise ValueError("Non-training supervision")
        with np.load(f.row["archive_path"], allow_pickle=False) as data:
            network = data["network"]
        onsets = np.array([e["onset_s"] for e in f.row["events"]])
        y = timing_labels(f.raw[:, 0], onsets, network)
        raw.append(f.x[f.usable])
        labels.append(y[f.usable])
    return np.concatenate(raw), np.concatenate(labels)


def warm_start(raw, labels, weights, device):
    torch.manual_seed(SEED)
    model = Policy(True).to(device)
    # Some onset-related fields are missing by design when no cut is predicted.
    with np.errstate(invalid="ignore"):
        center = np.nan_to_num(np.nanmean(raw, axis=0))
        scale = np.nan_to_num(np.nanstd(raw, axis=0), nan=1.)
    model.center.copy_(tensor(center, device))
    model.scale.copy_(tensor(np.maximum(scale, .1), device))
    x, y = tensor(raw, device), tensor(labels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    rng = np.random.default_rng(SEED)
    losses = []
    for epoch in range(EPOCHS):
        batch_losses = []
        order = rng.permutation(len(x))
        for batch in np.array_split(order, int(np.ceil(len(order)/1024))):
            ix = tensor(batch, device)
            loss = supervised_loss(model, x[ix], y[ix], weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(batch_losses)))
        print("supervised epoch", epoch+1, "/", EPOCHS, losses[-1], flush=True)
    return model, losses


def rollout(task):
    spec, mode, checkpoint = task
    row, data = run(spec, mode, read_checkpoint(Path(checkpoint))["state"], training=True)
    return row, data


def merged(episodes):
    merged = {}
    sizes = [d["chosen"].shape[1] for _, d in episodes]
    width = max(sizes)
    for key in ("context", "candidates", "base", "chosen", "logp"):
        parts = []
        for n, (_, d) in zip(sizes, episodes):
            source = d[key]
            shape = (len(source), width, *source.shape[2:])
            result = np.zeros(shape, dtype=source.dtype)
            if key == "base":
                result[:, :, 1:] = -1e9
            result[:, :n] = source
            parts.append(result)
        merged[key] = np.concatenate(parts)
    merged["agent"] = np.concatenate([np.broadcast_to(np.arange(width)<n, (len(d["rewards"]),width))
                                      for n, (_, d) in zip(sizes, episodes)])
    for key in ("raw", "critic", "value"):
        merged[key] = np.concatenate([d[key] for _, d in episodes]).astype(np.float32)
    returns, advantages = [], []
    for _, data in episodes:
        network = data["network"]
        onset = np.flatnonzero(np.r_[False, network[:-2] & ~network[1:-1] & ~network[2:], False])
        cost = np.bincount(onset//6, minlength=len(data["rewards"]))
        reward = data["rewards"]-.25*cost
        values = data["estimate"]
        gae = 0.
        advantage = np.zeros(len(values))
        for t in reversed(range(len(values))):
            next_value = values[t+1] if t+1 < len(values) else 0.
            gae = reward[t]+.99*next_value-values[t]+.99*.95*gae
            advantage[t] = gae
        advantages.append(advantage)
        returns.append(advantage+values)
    advantage = np.concatenate(advantages)
    merged["adv"] = ((advantage-advantage.mean())/max(1e-6,advantage.std())).astype(np.float32)
    merged["return"] = np.concatenate(returns).astype(np.float32)
    return merged


def update(model, optimizer, data, raw, labels, weights, seed):
    device = next(model.parameters()).device
    d = {k:tensor(v, device) for k,v in data.items()}
    model.train()
    inputs = ("raw", "context", "candidates", "base", "critic", "value")
    rng = np.random.default_rng(seed)
    before = {k:v.clone().detach() for k,v in model.named_parameters()}
    scale = float(d["return"].std().clamp_min(.1))
    errors = []
    with torch.no_grad():
        for ix in tensor(np.arange(len(d["adv"])), device).split(256):
            logits, _ = model(*(d[k][ix] for k in inputs))
            lp = torch.distributions.Categorical(logits=logits).log_prob(d["chosen"][ix])
            errors.append(float((lp-d["logp"][ix])[d["agent"][ix]].abs().max()))
    if max(errors) > 5e-4:
        raise ValueError(f"On-policy replay mismatch: {max(errors):.6g}")
    max_policy_gradient = 0.
    losses = []
    for _ in range(2):
        for batch in np.array_split(rng.permutation(len(d["adv"])), int(np.ceil(len(d["adv"])/192))):
            ix = tensor(batch, device)
            logits, values = model(*(d[k][ix] for k in inputs))
            dist = torch.distributions.Categorical(logits=logits)
            old = torch.distributions.Categorical(logits=d["base"][ix])
            active = d["agent"][ix]
            ratio = (dist.log_prob(d["chosen"][ix])-d["logp"][ix]).exp()
            a = d["adv"][ix, None]
            objective = -torch.minimum(ratio*a, ratio.clamp(.85, 1.15)*a)[active].mean()
            if model.continuous and max_policy_gradient == 0.:
                grad = torch.autograd.grad(objective, tuple(model.encoder.parameters())+tuple(model.timing.parameters()),
                                           retain_graph=True, allow_unused=True)
                max_policy_gradient = float(sum(float(g.detach().square().sum()) for g in grad if g is not None)**.5)
            critic = ((values-d["return"][ix])/scale).square().mean()
            penalty = torch.distributions.kl_divergence(dist, old)[active].mean()
            loss = objective+.5*critic+.03*penalty-.005*dist.entropy()[active].mean()
            if model.continuous:
                aux = rng.integers(0, len(raw), size=512)
                loss += .2*supervised_loss(model, tensor(raw[aux], device), tensor(labels[aux], device), weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(clip_grad_norm_(model.parameters(), .5)):
                raise ValueError("Non-finite PPO gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    change = float(sum(float((v.detach()-before[k]).square().sum()) for k,v in model.named_parameters())**.5)
    if change <= 0 or (model.continuous and max_policy_gradient <= 0):
        raise ValueError("PPO did not learn through the expected inputs")
    return dict(steps=len(d["adv"]), behavior_logp_max_error=max(errors), loss=float(np.mean(losses)),
                weight_change_l2=change, ppo_gradient_into_predictor_l2=max_policy_gradient)


def fit(plan):
    root = ARTIFACTS/"models"
    warm = root/"warm.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    raw, labels = load_train()
    counts = np.bincount(labels, minlength=4)
    weights = tensor(np.sqrt(counts.max()/counts).clip(1., 15.).astype(np.float32), device)
    if warm.exists():
        initial = read_checkpoint(warm)
        if initial["train_manifest_sha256"] != sha(V21/"train_manifest.json"):
            raise ValueError("Warm checkpoint training cohort changed")
    else:
        status("2_supervised_forecast", rows=len(raw))
        model, losses = warm_start(raw, labels, weights, device)
        initial = dict(state=model.cpu().state_dict(), losses=losses, rows=len(raw), label_counts=counts.tolist(),
                       train_manifest_sha256=sha(V21/"train_manifest.json"), device=device)
        safe_save(warm, initial)
    all_updates = {mode: [] for mode in MODES}
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"), initializer=initialize) as pool:
        for step in range(UPDATES):
            specs = plan["train"][step*FLIGHTS_PER_UPDATE:(step+1)*FLIGHTS_PER_UPDATE]
            pending = {}
            for mode in MODES:
                output = root/f"{mode}_{step+1}.pt"
                if output.exists():
                    saved = read_checkpoint(output)
                    if saved["training_specs"] != specs:
                        raise ValueError("Cached PPO update has a different cohort")
                    all_updates[mode].append(saved["stats"])
                    continue
                source = root/f"{mode}_{step}.pt" if step else warm
                pending[mode] = [pool.submit(rollout, (spec, mode, str(source))) for spec in specs]
            if pending:
                status("2_integrated_ppo", update=step+1, total=UPDATES)
            for mode, futures in pending.items():
                source = root/f"{mode}_{step}.pt" if step else warm
                source_data = read_checkpoint(source)
                model = Policy(mode == "ppo_continuous").to(device)
                model.load_state_dict(source_data["state"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
                if step:
                    optimizer.load_state_dict(source_data["optimizer"])
                episodes = [f.result() for f in futures]
                stats = update(model, optimizer, merged(episodes), raw, labels, weights, SEED+step)
                stats["flights"] = [dict(spec=r["spec"], deployment=r["deployment"], finished=r["mission_completed"],
                                         cuts=r["persistent_onsets"], timely_pdr=r["timely_pdr"]) for r,_ in episodes]
                all_updates[mode].append(stats)
                output = root/f"{mode}_{step+1}.pt"
                safe_save(output, dict(state=model.cpu().state_dict(), optimizer=optimizer.state_dict(),
                                       training_specs=specs, stats=stats))
                print(mode, "PPO update", step+1, "steps", stats["steps"], "gradient",
                      stats["ppo_gradient_into_predictor_l2"], flush=True)
    models = {mode:dict(path=str(root/f"{mode}_{UPDATES}.pt"), sha256=sha(root/f"{mode}_{UPDATES}.pt"))
              for mode in MODES}
    result = dict(supervised={k:v for k,v in initial.items() if k != "state"}, ppo=all_updates, models=models)
    write(ARTIFACTS/"training.json", result)
    return result


if __name__ == "__main__":
    import ctypes
    import traceback
    from adaptive_joint_expanded.contracts import lock
    from .protocol import prepare

    with lock(ARTIFACTS/"training.lock"):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            fit(prepare())
            status("2_training_finished")
        except BaseException as error:
            status("2_training", state="failed", error=repr(error))
            traceback.print_exc()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
