"""학습-실행 입력 동등성, 로컬 관측, 전체 임무 산출물 검수."""

import argparse, json, time, unittest
from pathlib import Path
import numpy as np
import torch
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from swarm_sim.city import City
from swarm_sim.config import Scenario
from usable.data import collection, load_observation
from advance.data import BASE, DATA, RESULT
from advance.features import read_record, VERSION
from advance.infer import feature_batch, selected
from advance.predictor import LinkRiskPredictor, LocalGoalCache


def bank_parity():
    torch.set_num_threads(1)
    base = torch.load(BASE / "best.pt", map_location="cpu", weights_only=False)
    state = {
        "normalizer": base["normalizer"],
        "extra_normalizer": read_json(DATA / "extra_normalizer.json"),
    }
    meta = read_json(DATA / "select/bank.json")
    rows = np.load(DATA / "select/selected.npy")
    f = np.load(DATA / "select/features.npy", mmap_mode="r")
    take = np.sort(np.random.default_rng(55120).choice(len(rows), 128, replace=False))
    maps = {
        e["key"]: e for e in read_json(ROOT / "configs/usable_protocol.json")["maps"]
    }
    delta = 0.0
    previous = None
    for ep in np.unique(rows[take, 0]):
        r = meta["records"][int(ep)]
        q = take[rows[take, 0] == ep]
        if previous != r["region"]:
            city = City(ROOT / maps[r["region"]]["path"], Scenario())
            previous = r["region"]
        x, m, intent = read_record(r)
        z, known, e, s = feature_batch(state, city, x, m, rows[q, 1:], intent)
        actual = np.concatenate((s, e), -1)
        difference = float(np.max(np.abs(actual - f[q])))
        delta = max(delta, difference)
    if delta > 2e-5:
        raise AssertionError("학습 은행/원시 기록 입력 차이 " + str(delta))
    out = {
        "samples": len(take),
        "feature_version": VERSION,
        "maximum_feature_difference": delta,
        "same_training_and_runtime_extractor": True,
        "validation_only": True,
    }
    write_json(RESULT / "bank_parity.json", out)
    print(json.dumps(out))
    return out


def runtime(output=None):
    torch.set_num_threads(1)
    manifest, p = collection()
    r = next(
        r
        for r in manifest["records"]
        if r["region"] == "seoul_20261023_03"
        and r["metadata"]["direction"] == "north_south"
        and r["metadata"]["kind"] == "intermittent"
    )
    entry = next(e for e in p["maps"] if e["key"] == r["region"])
    x, m, intents = read_record(r)
    folder = ROOT / "model/advance/selected"
    a = LinkRiskPredictor(folder / "best.pt", ROOT / entry["path"])
    b = LinkRiskPredictor(folder / "best.pt", ROOT / entry["path"])
    c = LinkRiskPredictor(folder / "best.pt", ROOT / entry["path"])
    caches = [LocalGoalCache(8, i) for i in range(8)]
    with np.load(
        ROOT / "cache/usable" / f"{r['region']}_{r['metadata']['direction']}_motion.npz"
    ) as z:
        desired = z["desired"]
    difference = 0.0
    local_difference = 0.0
    goal_difference = 0.0
    times = []
    local_times = []
    full_times = []
    decisions = 0
    for t in range(340):
        now = (t + 1) * 0.5
        segment = t // 6
        live = np.zeros((8, 8, 9), np.float32)
        for i, cache in enumerate(caches):
            cache.update_own(segment * 3.0, desired[segment + 1, i])
            for j in range(8):
                if i != j and m[t, i, j, 33] and x[t, i, j, 33] > 0:
                    cache.received_beacon(
                        j, segment * 3.0, desired[segment + 1, j], now
                    )
            live[i] = cache.encode(now, x[t, i, :, :48], m[t, i, :, :48])
        goal_difference = max(goal_difference, float(np.max(np.abs(live - intents[t]))))
        before = time.perf_counter()
        ra = a.observe(now, x[t], m[t], intents[t])
        ta = time.perf_counter() - before
        before = time.perf_counter()
        rb = b.observe(now, x[t, :, :, :48], m[t, :, :, :48], live)
        tb = time.perf_counter() - before
        before = time.perf_counter()
        rc = c.observe_local(now, 0, x[t, 0, :, :48], m[t, 0, :, :48], live[0])
        tc = time.perf_counter() - before
        if ra["status"] != "ready":
            continue
        decisions += 1
        full_times.append(ta)
        times.append(tb)
        local_times.append(tc)
        for u, v in zip(ra["links"], rb["links"]):
            assert u["status"] == v["status"]
            for key in ["risk_5s", "risk_onset_3to4s", "risk_onset_3to7s"]:
                if u[key] is not None:
                    difference = max(difference, abs(u[key] - v[key]))
        for u, v in zip([q for q in ra["links"] if q["source"] == 0], rc["links"]):
            assert u["status"] == v["status"]
            for key in ["risk_5s", "risk_onset_3to4s", "risk_onset_3to7s"]:
                if u[key] is not None:
                    local_difference = max(local_difference, abs(u[key] - v[key]))
    assert goal_difference < 1e-5 and difference < 2e-5 and local_difference < 2e-5

    def latency(v):
        return {
            "median_ms": float(np.median(v) * 1000),
            "p95_ms": float(np.percentile(v, 95) * 1000),
        }

    out = {
        "checkpoint_sha256": sha256_file(folder / "best.pt"),
        "checked_decisions": decisions,
        "causal_goal_cache_difference": goal_difference,
        "precomputed_vs_internal_map_difference": difference,
        "full_vs_local_difference": local_difference,
        "status_matches": True,
        "cpu_threads": 1,
        "full_56_links": latency(full_times),
        "full_56_with_map": latency(times),
        "one_observer_7_links": latency(local_times),
        "timing_scope": "this CPU only; initialization excluded; no flight-controller timing claim",
    }
    write_json(Path(output) if output else RESULT / "runtime_verification.json", out)
    print(json.dumps(out, indent=2))
    return out


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    runtime(ROOT / "cache/advance_last_runtime_check.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bank-only", action="store_true")
    p.add_argument("--runtime-only", action="store_true")
    a = p.parse_args()
    if a.bank_only:
        bank_parity()
    elif a.runtime_only:
        runtime()
    else:
        main()
