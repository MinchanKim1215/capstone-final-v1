"""같은 지도·입력·정답의 트리/Transformer 비교. 시험 자료 미사용."""

import argparse, copy, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import lightgbm as lgb
from scipy.special import expit
from sklearn.metrics import average_precision_score
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from usable.model import RiskTransformer
from advance.features import VERSION, FEATURES
from advance.data import DATA, RESULT, BASE
from upgrade.train import configure, lr_at

RUNS = RESULT / "runs"
HEADS = [0, 3, 4, 6]


def persist(tag):
    helper = os.environ.get("ADVANCE_PERSIST_SCRIPT")
    if helper:
        subprocess.run([sys.executable, helper, tag], check=True)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_state(path, data):
    temp = path.with_name(path.name + ".tmp")
    torch.save(data, temp)
    temp.replace(path)


def bank(name):
    f = DATA / name
    return {
        k: np.load(f / (v + ".npy"), mmap_mode="r")
        for k, v in [
            ("features", "features"),
            ("values", "values"),
            ("mask", "mask"),
            ("y", "targets"),
        ]
    }


def train_tree(leaves=31, seed=51):
    configure(seed, 2)
    folder = RUNS / f"gbdt{leaves}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "run.json").exists():
        return
    tr = bank("train")
    va = bank("select")
    n = len(tr["y"])
    original = read_json(DATA / "train/bank.json")["original_n"]
    weight = np.r_[
        np.full(original, n / (2 * original)),
        np.full(n - original, n / (2 * (n - original))),
    ].astype(np.float32)
    conf = {
        "family": "GBDT",
        "leaves": leaves,
        "seed": seed,
        "training_samples": n,
        "heads": HEADS,
        "maximum_rounds": 1800,
        "early_stopping": 120,
        "learning_rate": 0.04,
        "feature_version": VERSION,
        "training_only": True,
        "group_weights": [0.5, 0.5],
    }
    atomic_json(folder / "config.json", conf)
    started = time.perf_counter()
    trees = {}
    z = np.zeros((len(va["y"]), 7), np.float32)
    rounds = {}
    dataset = lgb.Dataset(
        tr["features"],
        label=tr["y"][:, 0],
        weight=weight,
        free_raw_data=True,
        params={"num_threads": 2, "min_data_in_leaf": 100},
    )
    dataset.construct()
    for h in HEADS:
        text = folder / "heads" / f"{h}.txt"
        cached = folder / "heads" / f"{h}_validation_logits.npy"
        meta = folder / "heads" / f"{h}.json"
        if text.exists() and cached.exists() and meta.exists():
            trees[str(h)] = text.read_text()
            z[:, h] = np.load(cached)
            rounds[str(h)] = read_json(meta)["rounds"]
            continue
        dataset.set_label(tr["y"][:, h])
        valid = lgb.Dataset(
            va["features"], label=va["y"][:, h], reference=dataset, free_raw_data=True
        )
        params = dict(
            objective="binary",
            metric="average_precision",
            num_leaves=leaves,
            min_data_in_leaf=100,
            learning_rate=0.04,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            lambda_l2=5,
            verbosity=-1,
            num_threads=2,
            seed=55115 + seed - 51,
            force_col_wise=True,
        )
        hist = {}
        b = lgb.train(
            params,
            dataset,
            num_boost_round=1800,
            valid_sets=[valid],
            callbacks=[
                lgb.early_stopping(120, verbose=False),
                lgb.record_evaluation(hist),
            ],
        )
        trees[str(h)] = b.model_to_string(num_iteration=b.best_iteration)
        rounds[str(h)] = b.best_iteration
        z[:, h] = b.predict(
            va["features"],
            raw_score=True,
            num_iteration=b.best_iteration,
            num_threads=2,
        ).astype(np.float32)
        ap = float(average_precision_score(va["y"][:, h], expit(z[:, h])))
        text.parent.mkdir(exist_ok=True)
        temp = text.with_suffix(".tmp")
        temp.write_text(trees[str(h)])
        temp.replace(text)
        np.save(cached, z[:, h])
        atomic_json(
            meta,
            {
                "rounds": b.best_iteration,
                "validation_ap": ap,
                "head": h,
                "history": hist,
            },
        )
        print(
            "tree",
            leaves,
            seed,
            "head",
            h,
            "AP",
            ap,
            "rounds",
            b.best_iteration,
            "elapsed",
            round(time.perf_counter() - started, 1),
            flush=True,
        )
        persist(f"gbdt{leaves}_s{seed}_head{h}")
    source = torch.load(BASE / "best.pt", map_location="cpu", weights_only=False)
    state = {
        "architecture": "gbdt",
        "trees": trees,
        "trained_heads": HEADS,
        "normalizer": source["normalizer"],
        "extra_normalizer": read_json(DATA / "extra_normalizer.json"),
        "feature_version": VERSION,
        "config": conf,
    }
    save_state(folder / "best.pt", state)
    np.save(folder / "best_validation_logits.npy", z)
    atomic_json(
        folder / "run.json",
        {
            **conf,
            "validation_ap": float(
                average_precision_score(va["y"][:, 0], expit(z[:, 0]))
            ),
            "early_ap": float(average_precision_score(va["y"][:, 3], expit(z[:, 3]))),
            "rounds_per_head": rounds,
            "checkpoint_sha256": sha256_file(folder / "best.pt"),
            "elapsed_s": time.perf_counter() - started,
        },
    )
    persist(f"gbdt{leaves}_s{seed}_complete")


class ResidualTransformer(nn.Module):
    def __init__(self, base_spec):
        super().__init__()
        self.base_spec = base_spec
        self.base = RiskTransformer(**base_spec)
        self.base.forecast[-1] = nn.Linear(32, 7)
        self.extra = nn.Sequential(
            nn.LayerNorm(137),
            nn.Linear(137, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 7),
        )
        self.stats = nn.Sequential(
            nn.LayerNorm(660),
            nn.Linear(660, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 7),
        )
        for head in (self.extra, self.stats):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x, m, e, s):
        return self.base(x, m) + self.extra(e) + self.stats(s)


def nn_batch(b, rows, normalizer):
    x = b["values"][rows].astype(np.float32)
    m = b["mask"][rows].copy()
    mean = np.asarray(normalizer["mean"], np.float32)[FEATURES]
    std = np.asarray(normalizer["std"], np.float32)[FEATURES]
    x = np.where(m, np.clip((x - mean) / std, -8, 8), 0).astype(np.float32)
    f = b["features"][rows]
    return (
        torch.from_numpy(x),
        torch.from_numpy(m),
        torch.from_numpy(f[:, 660:].copy()),
        torch.from_numpy(f[:, :660].copy()),
        torch.from_numpy(b["y"][rows].copy()),
    )


@torch.inference_mode()
def nn_predict(model, b, normalizer):
    model.eval()
    out = np.empty((len(b["y"]), 7), np.float32)
    for first in range(0, len(out), 512):
        x, m, e, s, _ = nn_batch(
            b, np.arange(first, min(first + 512, len(out))), normalizer
        )
        out[first : first + len(x)] = model(x, m, e, s).numpy()
    return out


def train_transformer(seed=51, updates=6000):
    configure(seed, 2)
    folder = RUNS / f"transformer_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "run.json").exists():
        return
    source = torch.load(BASE / "best.pt", map_location="cpu", weights_only=False)
    tr = bank("train")
    va = bank("select")
    original = read_json(DATA / "train/bank.json")["original_n"]
    model = ResidualTransformer(source["spec"])
    old = source["state_dict"]
    last = str(len(model.base.forecast) - 1)
    model.base.load_state_dict(
        {k: v for k, v in old.items() if not k.startswith("forecast." + last + ".")},
        strict=False,
    )
    with torch.no_grad():
        model.base.forecast[-1].weight[:3].copy_(old["forecast." + last + ".weight"])
        model.base.forecast[-1].bias[:3].copy_(old["forecast." + last + ".bias"])
        p = np.clip(np.asarray(tr["y"][:, 3:]).mean(0), 1e-5, 1 - 1e-5)
        model.base.forecast[-1].bias[3:].copy_(torch.from_numpy(np.log(p / (1 - p))))
    conf = {
        "family": "Transformer",
        "seed": seed,
        "training_samples": len(tr["y"]),
        "maximum_updates": updates,
        "batch_size": 256,
        "base_lr": 0.00006,
        "added_head_lr": 0.0003,
        "source_checkpoint_sha256": sha256_file(BASE / "best.pt"),
        "feature_version": VERSION,
        "training_only": True,
        "loss": "BCE5 + .3BCE10 + .1Huber + .8BCE3to4 + .5BCE3to7 + .3BCE0to4 + .3BCE0to7; prior importance correction",
        "initial_source": "old B3 encoder/first three outputs; new residual paths initialized to zero",
    }
    atomic_json(folder / "config.json", conf)
    opt = torch.optim.AdamW(
        [
            {"params": model.base.parameters(), "lr": 0.00006},
            {
                "params": list(model.extra.parameters())
                + list(model.stats.parameters()),
                "lr": 0.0003,
            },
        ],
        weight_decay=0.0001,
    )
    rng = np.random.default_rng(55121 + seed)
    pools = []
    for lo, hi in [(0, original), (original, len(tr["y"]))]:
        y = tr["y"][lo:hi, 0]
        pos = np.flatnonzero(y) + lo
        neg = np.flatnonzero(y == 0) + lo
        pools.append((pos, neg, len(pos) / (hi - lo)))
    weight = torch.from_numpy(
        np.concatenate(
            [
                np.r_[np.full(32, p / 0.25), np.full(96, (1 - p) / 0.75)]
                for _, _, p in pools
            ]
        ).astype(np.float32)
    )
    best = -1
    bestearly = -1
    beststep = 0
    earlystep = 0
    history = []
    begin = 0
    elapsed_previous = 0.0
    started = time.perf_counter()
    total = 0.0
    lastfile = folder / "resume.pt"
    if lastfile.exists():
        state = torch.load(lastfile, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        begin = state["step"]
        best = state["best"]
        bestearly = state["bestearly"]
        beststep = state["beststep"]
        earlystep = state["earlystep"]
        history = state["history"]
        elapsed_previous = state["elapsed_s"]

    def state(step):
        return {
            "architecture": "transformer",
            "state_dict": model.state_dict(),
            "base_spec": source["spec"],
            "normalizer": source["normalizer"],
            "extra_normalizer": read_json(DATA / "extra_normalizer.json"),
            "feature_version": VERSION,
            "config": conf,
            "step": step,
        }

    for step in range(begin, updates):
        rows = np.concatenate(
            [np.r_[rng.choice(pos, 32), rng.choice(neg, 96)] for pos, neg, _ in pools]
        )
        x, m, e, s, y = nn_batch(tr, rows, source["normalizer"])
        model.train()
        opt.zero_grad(set_to_none=True)
        for group, lr in zip(opt.param_groups, (0.00006, 0.0003)):
            group["lr"] = lr_at(step, updates, lr)
        z = model(x, m, e, s)

        def bce(q):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                z[:, q], y[:, q], reduction="none"
            )

        loss = (
            bce(0)
            + 0.3 * bce(1)
            + 0.1
            * torch.nn.functional.smooth_l1_loss(z[:, 2], y[:, 2], reduction="none")
            + 0.8 * bce(3)
            + 0.5 * bce(4)
            + 0.3 * bce(5)
            + 0.3 * bce(6)
        ) * weight
        loss = loss.mean()
        if not torch.isfinite(loss):
            raise ValueError("학습 손실 비정상")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        total += float(loss.detach())
        if (step + 1) % 500 == 0:
            logits = nn_predict(model, va, source["normalizer"])
            ap = float(average_precision_score(va["y"][:, 0], expit(logits[:, 0])))
            early = float(average_precision_score(va["y"][:, 3], expit(logits[:, 3])))
            if ap > best + 1e-6:
                best = ap
                beststep = step + 1
                save_state(folder / "best.pt", state(step + 1))
                np.save(folder / "best_validation_logits.npy", logits)
            if early > bestearly + 1e-6:
                bestearly = early
                earlystep = step + 1
                save_state(folder / "early.pt", state(step + 1))
                np.save(folder / "early_validation_logits.npy", logits)
            row = {
                "step": step + 1,
                "loss": total / 500,
                "validation_ap": ap,
                "early_ap": early,
                "elapsed_s": elapsed_previous + time.perf_counter() - started,
            }
            history.append(row)
            total = 0.0
            atomic_json(folder / "history.json", history)
            print(json.dumps(row), flush=True)
            resume = state(step + 1)
            resume.update(
                optimizer=opt.state_dict(),
                rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                best=best,
                bestearly=bestearly,
                beststep=beststep,
                earlystep=earlystep,
                history=history,
                elapsed_s=row["elapsed_s"],
            )
            save_state(lastfile, resume)
            persist(f"transformer_s{seed}_step{step+1}")
    atomic_json(
        folder / "run.json",
        {
            **conf,
            "validation_ap": best,
            "early_ap": bestearly,
            "best_step": beststep,
            "early_step": earlystep,
            "elapsed_s": elapsed_previous + time.perf_counter() - started,
            "checkpoint_sha256": sha256_file(folder / "best.pt"),
            "early_checkpoint_sha256": sha256_file(folder / "early.pt"),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    persist(f"transformer_s{seed}_complete")


def freeze():
    plan = read_json(ROOT / "configs/advance_plan.json")
    y = np.load(DATA / "select/targets.npy")
    rows = []
    for name in ["gbdt31_s51", "gbdt63_s51", "transformer_s51"]:
        folder = RUNS / name
        if not (folder / "run.json").exists():
            raise ValueError("계획한 비교군 완료 필요: " + name)
        meta = read_json(folder / "run.json")
        for checkpoint in ["best", "early"]:
            f = folder / (checkpoint + "_validation_logits.npy")
            if not f.exists():
                continue
            z = np.load(f)
            rows.append(
                {
                    "run": name,
                    "checkpoint": checkpoint + ".pt",
                    "family": meta["family"],
                    "ap5": float(average_precision_score(y[:, 0], expit(z[:, 0]))),
                    "early_ap": float(average_precision_score(y[:, 3], expit(z[:, 3]))),
                    "checkpoint_sha256": sha256_file(folder / (checkpoint + ".pt")),
                }
            )
    highest = max(r["ap5"] for r in rows)
    valid = [r for r in rows if r["ap5"] >= highest - plan["main_ap_tolerance"]]
    chosen = max(valid, key=lambda r: (r["early_ap"], r["ap5"]))
    tf = max(
        [r for r in rows if r["family"] == "Transformer"],
        key=lambda r: (r["ap5"], r["early_ap"]),
    )
    out = {
        "rules": plan,
        "selected": chosen,
        "transformer_reference": tf,
        "all_candidates": rows,
        "test_used": False,
    }
    path = RESULT / "selection.json"
    if path.exists() and read_json(path) != out:
        raise ValueError("최종 선택 동결 위반")
    atomic_json(path, out)
    persist("model_selection_frozen")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["tree", "transformer", "freeze"])
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--updates", type=int, default=6000)
    a = p.parse_args()
    if a.kind == "tree":
        train_tree(a.leaves, a.seed)
    elif a.kind == "transformer":
        train_transformer(a.seed, a.updates)
    else:
        freeze()
