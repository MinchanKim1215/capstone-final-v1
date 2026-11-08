"""가중치·문턱 고정 후 동일 비행의 분류/운용/사건 지표 평가."""

import argparse, json
import numpy as np
import torch
from scipy.special import expit
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from swarm_sim.city import City
from swarm_sim.config import Scenario
from swarm_sim.forecast_evaluation import binary_metrics
from usable.data import collection, index_episode, load_observation
from usable.evaluate import load as old_load, predict_episode
from usable.alarm_policy import summarize
from upgrade.evaluate import grouped
from advance.data import RESULT, BASE
from advance.infer import load, selected, score_records, probabilities, eligible
from advance.features import targets
from advance.events import count
from advance.calibrate import baseline_transform
from advance.train import persist


def old_scores(r, ix, current, operational, union, fresh):
    if not fresh:
        with np.load(BASE / "dense" / (r["id"] + ".npz")) as z:
            np.testing.assert_array_equal(ix["index"][current], z["index"])
            a = z["probability"]
        with np.load(BASE / "runtime" / (r["id"] + ".npz")) as z:
            np.testing.assert_array_equal(ix["index"][operational], z["index"])
            b = z["probability"]
        return a, b
    cache = RESULT / "baseline_confirmation_scores"
    cache.mkdir(exist_ok=True)
    file = cache / (r["id"] + ".npz")
    index = ix["index"][union]
    if file.exists():
        with np.load(file) as z:
            np.testing.assert_array_equal(z["index"], index)
            p = z["probability"]
    else:
        model, state, _ = old_load(BASE)
        cal = read_json(BASE / "calibration.json")
        p = expit(predict_episode(model, state, r, index) / cal["temperature"]).astype(
            np.float32
        )
        np.savez_compressed(
            file,
            index=index,
            probability=p,
            checkpoint_sha256=sha256_file(BASE / "best.pt"),
        )
    return p[current[union]], p[operational[union]]


def cluster_interval(a, b, protocol):
    group = {e["key"]: e["spatial_group"] for e in protocol["maps"]}
    v = {}
    for r in a:
        if (
            a[r]["average_precision"] is not None
            and b[r]["average_precision"] is not None
        ):
            v.setdefault(group[r], []).append(
                a[r]["average_precision"] - b[r]["average_precision"]
            )
    x = np.array([np.mean(q) for q in v.values()])
    rng = np.random.default_rng(55113)
    bootstrap = x[rng.integers(len(x), size=(10000, len(x)))].mean(1)
    return {
        "clusters": len(x),
        "mean_difference": float(x.mean()),
        "percentile95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "method": "paired spatial cluster mean AP differences; exploratory known geographies",
    }


def main(fresh=False):
    torch.set_num_threads(2)
    path, info = selected()
    refpath, refinfo = selected(True)
    model, state = load(path)
    models = {"default": (model, state, path)}
    if sha256_file(path) != sha256_file(refpath):
        rm, rs = load(refpath)
        models["transformer"] = (rm, rs, refpath)
    cal = read_json(RESULT / "selected/calibration.json")
    refcal = read_json(RESULT / "transformer_reference/calibration.json")
    oldcal = read_json(RESULT / "baseline_calibration.json")
    originalcal = read_json(BASE / "calibration.json")
    policy = read_json(RESULT / "selected/alarm_policy.json")
    policyfit = read_json(RESULT / "policy_calibration.json")
    quiet = read_json(BASE / "alarm_policy.json")["selected"]
    quiet = {"head": 0, **quiet} if quiet else None
    manifest, p = collection()
    records = (
        read_json(ROOT / "datasets/advance_confirmation/collection_manifest.json")[
            "records"
        ]
        if fresh
        else [r for r in manifest["records"] if r["split"] == "test"]
    )
    maps = {e["key"]: e for e in p["maps"]}
    name = "confirmation" if fresh else "regression"
    frozen = {
        "checkpoint_sha256": sha256_file(path),
        "reference_sha256": sha256_file(refpath),
        "calibration_sha256": sha256_file(RESULT / "selected/calibration.json"),
        "policy_sha256": sha256_file(RESULT / "selected/alarm_policy.json"),
        "before_evaluation": True,
        "test_used_for_selection": False,
    }
    f = RESULT / (name + "_frozen.json")
    if f.exists() and read_json(f) != frozen:
        raise ValueError("평가 중 모델/문턱 변경")
    write_json(f, frozen)
    ys = []
    new = []
    old = []
    ref = []
    earlyy = []
    earlyp = []
    regions = []
    op_y = []
    op_p = []
    op_old = []
    op_ref = []
    conditions = {}
    episodes = []
    duration = 0
    history_windows = 0
    previous = None
    alarm_keys = [
        "new_5s",
        "baseline_retuned_5s",
        "baseline_original_quiet_5s",
        "new_8s",
        "baseline_retuned_8s",
    ]
    counts = {k: np.zeros(9, np.int64) for k in alarm_keys}
    by_alarm = {}
    cut = cal["heads"]["0"]["threshold_conservative"]
    cut = 1.1 if cut is None else cut
    oldcut = oldcal["threshold_conservative"]
    oldcut = 1.1 if oldcut is None else oldcut
    refcut = refcal["heads"]["0"]["threshold_conservative"]
    refcut = 1.1 if refcut is None else refcut
    for r in records:
        ix = index_episode(r, p)
        x, m = load_observation(r)
        cur = ix["current_available"]
        op = eligible(x, m, ix["index"])
        union = cur | op
        index = ix["index"][union]
        history_windows += len(ix["index"])
        if previous != r["region"]:
            city = City(ROOT / maps[r["region"]]["path"], Scenario())
            previous = r["region"]
        z = score_records(r, index, models, city, RESULT / (name + "_scores"))
        probs = probabilities(z["default"], cal)
        rp = probabilities(z.get("transformer", z["default"]), refcal)
        bb, bo = old_scores(r, ix, cur, op, union, fresh)
        with np.load(ROOT / r["path"] / "truth.npz") as f:
            up = f["link_available"]
            y = targets(up, f["minimum_bidirectional_sinr_db"], index)
        np.testing.assert_array_equal(y[:, 0].astype(np.uint8), ix["y"][union])
        offline = cur[union]
        online = op[union]
        yy = y[offline, 0].astype(np.uint8)
        pp = probs[0][offline]
        rr = rp[0][offline]
        ys.append(yy)
        new.append(pp)
        old.append(bb)
        ref.append(rr)
        earlyy.append(y[offline, 3].astype(np.uint8))
        earlyp.append(probs.get(3, probs[0])[offline])
        regions.append(np.full(len(yy), r["region"]))
        op_y.append(y[online, 0].astype(np.uint8))
        op_p.append(probs[0][online])
        op_ref.append(rp[0][online])
        op_old.append(baseline_transform(bo, oldcal))
        oi = index[online]
        sp = {h: v[online] for h, v in probs.items()}
        bp = {0: baseline_transform(bo, oldcal)}
        cc = {
            "new_5s": count(up, oi, sp, policy["5s"], 5),
            "baseline_retuned_5s": count(
                up, oi, bp, policyfit["5s"]["baseline_same_grid"]["selected"], 5
            ),
            "baseline_original_quiet_5s": count(up, oi, {0: bo}, quiet, 5),
            "new_8s": count(up, oi, sp, policy["8s"], 8),
            "baseline_retuned_8s": count(
                up, oi, bp, policyfit["8s"]["baseline_same_grid"]["selected"], 8
            ),
        }
        dur = r["metadata"]["duration_s"]
        duration += dur
        rc = by_alarm.setdefault(
            r["region"],
            {"duration": 0, **{k: np.zeros(9, np.int64) for k in alarm_keys}},
        )
        rc["duration"] += dur
        for k in alarm_keys:
            counts[k] += cc[k]
            rc[k] += cc[k]
        d = conditions.setdefault(
            r["metadata"]["kind"], {"y": [], "new": [], "old": []}
        )
        d["y"].append(yy)
        d["new"].append(pp)
        d["old"].append(bb)
        episodes.append(
            {
                "id": r["id"],
                "region": r["region"],
                "duration_s": dur,
                "samples": len(yy),
                "positives": int(yy.sum()),
                "operational_candidates": int(online.sum()),
                "new": binary_metrics(yy, pp, cut),
                "baseline": binary_metrics(yy, baseline_transform(bb, oldcal), oldcut),
                "reference": binary_metrics(yy, rr, refcut),
                "alarm_counts": {k: v.tolist() for k, v in cc.items()},
            }
        )
        print(name, "completed", len(episodes), len(records), r["id"], flush=True)
    y = np.concatenate(ys)
    a = np.concatenate(new)
    b = np.concatenate(old)
    t = np.concatenate(ref)
    reg = np.concatenate(regions)
    yr = np.concatenate(op_y)
    aop = np.concatenate(op_p)
    bop = np.concatenate(op_old)
    top = np.concatenate(op_ref)
    out = {
        "frozen_inputs": frozen,
        "flights": len(records),
        "samples": len(y),
        "positive_fraction": float(y.mean()),
        "new": grouped(y, a, reg, cut),
        "baseline_retuned": grouped(y, baseline_transform(b, oldcal), reg, oldcut),
        "baseline_original": grouped(y, b, reg, originalcal["threshold_conservative"]),
        "transformer_reference": grouped(y, t, reg, refcut),
        "early_target": {
            "baseline_main_score": binary_metrics(np.concatenate(earlyy), b, 0.5),
            "new_specialized_score": binary_metrics(
                np.concatenate(earlyy), np.concatenate(earlyp), 0.5
            ),
        },
        "alarms": {k: summarize(v, duration) for k, v in counts.items()},
        "by_region_alarms": {
            r: {k: summarize(v[k], v["duration"]) for k in alarm_keys}
            for r, v in by_alarm.items()
        },
        "by_condition": {
            k: {
                "baseline": binary_metrics(
                    np.concatenate(v["y"]),
                    baseline_transform(np.concatenate(v["old"]), oldcal),
                    oldcut,
                ),
                "new": binary_metrics(
                    np.concatenate(v["y"]), np.concatenate(v["new"]), cut
                ),
            }
            for k, v in conditions.items()
        },
        "runtime": {
            "new": binary_metrics(yr, aop, cut),
            "baseline": binary_metrics(yr, bop, oldcut),
            "reference": binary_metrics(yr, top, refcut),
            "history_eligible_windows": history_windows,
            "operational_windows": len(yr),
            "operational_fraction": len(yr) / history_windows,
            "true_current_availability_used_for_gating": False,
        },
        "episodes": episodes,
        "control_repositioning_implemented": False,
        "geography": "same eight previously inspected evaluation regions; regression and new-randomness assessment, not untouched new-city validation",
        "horizon_8s_separate": True,
        "same_fixed_formation": True,
        "first20_s_warmup": True,
        "last10_s_decisions_excluded_due_to_aux_label": True,
    }
    out["cluster_macro_ap_interval"] = cluster_interval(
        out["new"]["by_region"], out["baseline_retuned"]["by_region"], p
    )
    if not fresh and len(y) != 4866832:
        raise ValueError("원본 비교 창 수 변경")
    write_json(RESULT / (name + "_evaluation.json"), out)
    np.savez_compressed(
        RESULT / (name + "_aggregate.npz"),
        y=y,
        new=a,
        baseline=b,
        reference=t,
        regions=reg,
        early_y=np.concatenate(earlyy),
        early_new=np.concatenate(earlyp),
    )
    persist(name + "_evaluation_complete")
    print(
        json.dumps(
            {
                "baseline_ap": out["baseline_original"]["all"]["average_precision"],
                "new_ap": out["new"]["all"]["average_precision"],
                "reference_ap": out["transformer_reference"]["all"][
                    "average_precision"
                ],
                "alarms": out["alarms"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    main(a.fresh)
