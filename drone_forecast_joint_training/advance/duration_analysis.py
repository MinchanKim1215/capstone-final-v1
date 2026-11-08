"""고정 정책의 단절 길이별 포착 분해. 정답/경고 문턱 변경 없음."""

import argparse, json
import numpy as np
from swarm_sim.regions import ROOT, read_json, write_json
from usable.data import collection, load_observation
from usable.alarm_policy import simulate_policy, summarize
from advance.data import RESULT, BASE
from advance.infer import probabilities, eligible
from advance.calibrate import baseline_transform
from advance.events import arrays


def lengths(up):
    out = []
    for i in range(up.shape[1]):
        for j in range(up.shape[2]):
            if i == j:
                continue
            bounds = np.diff(np.r_[False, ~up[:, i, j], False].astype(np.int8))
            for start, stop in zip(
                np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1)
            ):
                if stop - start >= 2 and start * 0.5 >= 20:
                    out.append((stop - start) * 0.5)
    return np.asarray(out)


def subgroup(base, event_lengths, min_duration):
    p, y, new, event3, n = base
    assert n == len(event_lengths)
    keep = event_lengths >= min_duration
    mapping = np.full(n, -1, np.int32)
    mapping[keep] = np.arange(keep.sum())
    changed = event3.copy()
    known = changed >= 0
    changed[known] = mapping[changed[known]]
    # 최초 경고 정밀도/오경보의 원래 정의 유지. 사건 포착 분모·ID만 부분집합.
    return p, y, new, changed, int(keep.sum())


def main(scope="validation"):
    manifest, protocol = collection()
    cal = read_json(RESULT / "selected/calibration.json")
    oldcal = read_json(RESULT / "baseline_calibration.json")
    policy = read_json(RESULT / "selected/alarm_policy.json")
    fit = read_json(RESULT / "policy_calibration.json")
    if scope == "validation":
        from usable.data import validation_partition

        _, records = validation_partition(
            [r for r in manifest["records"] if r["split"] == "val"]
        )
    elif scope == "regression":
        records = [r for r in manifest["records"] if r["split"] == "test"]
    else:
        records = read_json(
            ROOT / "datasets/advance_confirmation/collection_manifest.json"
        )["records"]
    keys = [
        (model, h, minimum)
        for model in ["new", "baseline"]
        for h in [5, 8]
        for minimum in [1, 3, 5]
    ]
    counts = {k: np.zeros(9, np.int64) for k in keys}
    duration = 0
    for r in records:
        if scope == "validation":
            with np.load(RESULT / "selected/policy_scores" / (r["id"] + ".npz")) as z:
                index = z["index"]
                logits = z["default"]
            with np.load(BASE / "alarm_calibration_scores" / (r["id"] + ".npz")) as z:
                np.testing.assert_array_equal(z["index"], index)
                old = z["probability"]
        else:
            with np.load(RESULT / (scope + "_scores") / (r["id"] + ".npz")) as z:
                index = z["index"]
                logits = z["default"]
            x, m = load_observation(r)
            op = eligible(x, m, index)
            fullindex = index
            index = index[op]
            logits = logits[op]
            folder = (
                BASE / "runtime"
                if scope == "regression"
                else RESULT / "baseline_confirmation_scores"
            )
            with np.load(folder / (r["id"] + ".npz")) as z:
                if scope == "regression":
                    np.testing.assert_array_equal(z["index"], index)
                    old = z["probability"]
                else:
                    np.testing.assert_array_equal(z["index"], fullindex)
                    old = z["probability"][op]
        p = probabilities(logits, cal)
        b = {0: baseline_transform(old, oldcal)}
        with np.load(ROOT / r["path"] / "truth.npz") as z:
            up = z["link_available"]
        event_lengths = lengths(up)
        duration += r["metadata"]["duration_s"]
        for model, scores in [("new", p), ("baseline", b)]:
            for horizon in [5, 8]:
                rule = (
                    policy[str(horizon) + "s"]
                    if model == "new"
                    else fit[str(horizon) + "s"]["baseline_same_grid"]["selected"]
                )
                if rule is None:
                    rule = {"head": 0, "enter_threshold": 1.1, "exit_threshold": 1.1}
                original = arrays(up, index, scores[rule["head"]], horizon)
                for minimum in [1, 3, 5]:
                    counts[model, horizon, minimum] += simulate_policy(
                        *subgroup(original, event_lengths, minimum),
                        rule["enter_threshold"],
                        rule["exit_threshold"]
                    )[0]
        print("duration analysis", scope, r["id"], flush=True)
    rows = []
    for model, horizon, minimum in keys:
        v = summarize(counts[model, horizon, minimum], duration)
        rows.append(
            {
                "model": model,
                "horizon_s": horizon,
                "actual_outage_minimum_s": minimum,
                "events": v["all_new_directed_events"],
                "first_alert_3s_hits": int(counts[model, horizon, minimum][4]),
                "first_alert_3s_recall": v["first_alert_event_recall_3s"],
                "all_alerts": v["first_alerts"],
                "all_1s_contract_precision": v["first_alert_precision"],
                "all_1s_contract_false_per_drone_hour": v[
                    "false_alerts_per_drone_hour"
                ],
            }
        )
    out = {
        "scope": scope,
        "flights": len(records),
        "unchanged_policy": True,
        "original_label_minimum_s": 1,
        "subgroup_is_actual_duration_not_new_prediction_target": True,
        "all_alert_precision_definition_unchanged": True,
        "rows": rows,
    }
    write_json(RESULT / (scope + "_duration_analysis.json"), out)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scope",
        choices=["validation", "regression", "confirmation"],
        default="validation",
    )
    a = p.parse_args()
    main(a.scope)
