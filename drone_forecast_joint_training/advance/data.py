"""비행·지역 분리와 정답 독립 표본 추출. 원시 기록은 별도 보존."""

import json, time
import numpy as np
import torch
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from swarm_sim.city import City
from swarm_sim.config import Scenario
from usable.data import collection, index_episode, validation_partition
from advance.features import read_record, windows, extras, statistics, targets, VERSION

DATA = ROOT / "datasets/advance"
RESULT = ROOT / "results/advance"
BASE = ROOT / "results/usable/runs/B3_beacon_map_multitask_s11"


def samples(records, per_region, seed):
    p = read_json(ROOT / "configs/usable_protocol.json")
    rng = np.random.default_rng(seed)
    parts = []
    for key in sorted({r["region"] for r in records}):
        candidates = []
        for ep, r in enumerate(records):
            if r["region"] != key:
                continue
            ix = index_episode(r, p)
            q = np.flatnonzero(ix["current_available"])
            candidates.append(np.c_[np.full(len(q), ep), ix["index"][q]])
        values = np.concatenate(candidates)
        parts.append(
            values[rng.choice(len(values), min(per_region, len(values)), replace=False)]
        )
    return np.concatenate(parts).astype(np.int32)


def plan():
    dest = ROOT / "configs/advance_plan.json"
    spec = {
        "feature_version": VERSION,
        "training_original_windows": 327680,
        "extra_training_windows": 184320,
        "validation_select_windows": 57344,
        "validation_calibration_windows": 114688,
        "representative_seed": 51,
        "replication_seeds": [62, 73],
        "candidate_families": ["LightGBM31", "LightGBM63", "ResidualTransformer"],
        "selection": "within 0.005 absolute AP of highest validation-select 5s AP, maximize NEW onset3to4 AP, then 5s AP",
        "main_ap_tolerance": 0.005,
        "no_test_model_selection": True,
        "no_prediction_averaging": True,
        "new_resume_run": "different static-extra normalization after interrupted work; former validation-only values are not final test results",
        "primary_label": "same original currently-available cohort and >=1s unavailable inside next5s",
        "fixed_scenario": "45 maps; N8,search3,relay5,V40m,altitude50m",
        "learned_repositioning_implemented": False,
        "learning": "offline supervised learning; no emergency-time training",
        "beacon_goal_extension_bytes": 20,
        "beacon_total_bytes": 128,
        "goal_contents": "CURRENT issued xy float64 (16B) + uint32 issue time (4B), peer updated only after received beacon",
        "abstract_serialization_budget_only": True,
    }
    if dest.exists() and read_json(dest) != spec:
        raise ValueError("실험 계획 변경 금지")
    write_json(dest, spec)
    return spec


def build(name, records, selected, original_n=None):
    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    n = len(selected)
    required = [
        "features.npy",
        "values.npy",
        "mask.npy",
        "targets.npy",
        "selected.npy",
        "bank.json",
    ]
    if all((folder / f).exists() for f in required):
        return
    base = torch.load(BASE / "best.pt", map_location="cpu", weights_only=False)
    maps = {
        e["key"]: e for e in read_json(ROOT / "configs/usable_protocol.json")["maps"]
    }
    f = np.lib.format.open_memmap(
        folder / "features_raw.npy", mode="w+", dtype=np.float32, shape=(n, 797)
    )
    v = np.lib.format.open_memmap(
        folder / "values.npy", mode="w+", dtype=np.float16, shape=(n, 40, 55)
    )
    mask = np.lib.format.open_memmap(
        folder / "mask.npy", mode="w+", dtype=bool, shape=v.shape
    )
    y = np.lib.format.open_memmap(
        folder / "targets.npy", mode="w+", dtype=np.float32, shape=(n, 7)
    )
    previous = None
    city = None
    started = time.perf_counter()
    for ep in np.unique(selected[:, 0]):
        r = records[int(ep)]
        q = np.flatnonzero(selected[:, 0] == ep)
        index = selected[q, 1:]
        if previous != r["region"]:
            city = City(ROOT / maps[r["region"]]["path"], Scenario())
            previous = r["region"]
        x, m, intent = read_record(r)
        with np.load(ROOT / r["path"] / "truth.npz") as z:
            y[q] = targets(
                z["link_available"], z["minimum_bidirectional_sinr_db"], index
            )
        for first in range(0, len(q), 2048):
            rows = q[first : first + 2048]
            ix = selected[rows, 1:]
            xx, mm, raw = windows(x, m, ix, base["normalizer"])
            e = extras(city, x, m, ix, raw, mm, intent)
            f[rows, :660] = statistics(xx, mm)
            f[rows, 660:] = e
            v[rows] = raw.astype(np.float16)
            mask[rows] = mm
        print(
            "prepared",
            name,
            int(ep) + 1,
            len(records),
            r["id"],
            round(time.perf_counter() - started, 1),
            flush=True,
        )
    for a in (f, v, mask, y):
        a.flush()
    if name == "train":
        raw = np.asarray(f[:original_n, 660:], np.float64)
        mean = raw.mean(0)
        std = raw.std(0)
        std[std < 1e-6] = 1
        norm = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "fit": "original training subset only; no validation/test",
            "features": 137,
            "sample_count": original_n,
            "version": VERSION,
        }
        write_json(DATA / "extra_normalizer.json", norm)
    else:
        norm = read_json(DATA / "extra_normalizer.json")
    out = np.lib.format.open_memmap(
        folder / "features.npy", mode="w+", dtype=np.float32, shape=f.shape
    )
    for first in range(0, n, 4096):
        end = min(first + 4096, n)
        out[first:end, :660] = f[first:end, :660]
        out[first:end, 660:] = np.clip(
            (f[first:end, 660:] - np.asarray(norm["mean"], np.float32))
            / np.asarray(norm["std"], np.float32),
            -8,
            8,
        )
    out.flush()
    np.save(folder / "selected.npy", selected)
    meta = {
        "name": name,
        "samples": n,
        "records": records,
        "original_n": original_n,
        "feature_version": VERSION,
        "base_checkpoint_sha256": sha256_file(BASE / "best.pt"),
        "normalizer_sha256": sha256_file(DATA / "extra_normalizer.json"),
        "target_means": np.asarray(y).mean(0).tolist(),
        "sampling": "labels excluded; currently available offline cohort, whole-flight disjoint splits",
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(folder / "bank.json", meta)
    del f
    (folder / "features_raw.npy").unlink()


def prepare():
    torch.set_num_threads(1)
    DATA.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    plan()
    manifest, p = collection()
    original = [r for r in manifest["records"] if r["split"] == "train"]
    added = read_json(ROOT / "datasets/enhanced_extra/collection_manifest.json")[
        "records"
    ]
    assert len(original) == 180 and len(added) == 180
    path = DATA / "training_rows.npz"
    if path.exists():
        with np.load(path) as z:
            rows = z["rows"]
            original_n = int(z["original_n"])
    else:
        rows = samples(original, 32768, 55101)
        q = np.sort(
            np.random.default_rng(55107).choice(len(rows), 327680, replace=False)
        )
        rows = rows[q]
        original_n = len(rows)
        rng = np.random.default_rng(55110)
        extra = []
        for ep, r in enumerate(added):
            ix = index_episode(r, p)
            take = np.flatnonzero(ix["current_available"])
            q = rng.choice(take, min(1024, len(take)), replace=False)
            extra.append(np.c_[np.full(len(q), ep + len(original)), ix["index"][q]])
        rows = np.r_[rows, np.concatenate(extra)].astype(np.int32)
        np.savez_compressed(path, rows=rows, original_n=original_n)
    build("train", original + added, rows, original_n)
    val, cal = validation_partition(
        [r for r in manifest["records"] if r["split"] == "val"]
    )
    for name, records, limit, seed in [
        ("select", val, 8192, 55102),
        ("calibrate", cal, 16384, 55103),
    ]:
        rows = samples(records, limit, seed)
        build(name, records, rows)
    print("PREPARATION COMPLETE", flush=True)


if __name__ == "__main__":
    prepare()
