"""검증 비행의 사건 경고 정책. 첫 경고 품질·빈도 제약을 먼저 고정."""

import json
import numpy as np
import torch
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from swarm_sim.city import City
from swarm_sim.config import Scenario
from usable.data import (
    collection,
    index_episode,
    load_observation,
    validation_partition,
)
from usable.alarm_policy import simulate_policy, summarize
from advance.data import RESULT, BASE
from advance.infer import load, selected, score_records, probabilities, eligible
from advance.calibrate import baseline_transform
from advance.events import arrays
from advance.train import persist

GOALS = {
    "minimum_first_alert_precision": 0.8,
    "maximum_false_alerts_per_drone_hour": 5.0,
    "minimum_validation_alerts": 30,
}
THRESHOLDS = np.unique(
    np.r_[np.geomspace(0.0005, 0.3, 22), np.linspace(0.3, 0.99, 32), 0.995, 0.999]
)


def optimize(records, heads, horizon):
    duration = sum(r["duration"] for r in records)
    candidates = []
    for head in heads:
        prepared = [
            arrays(r["up"], r["index"], r["scores"][head], horizon) for r in records
        ]
        for threshold in THRESHOLDS:
            for ratio in (0.25, 0.5, 0.75, 1.0):
                count = np.zeros(9, np.int64)
                for a in prepared:
                    count += simulate_policy(*a, threshold, threshold * ratio)[0]
                v = summarize(count, duration)
                okay = (
                    v["first_alerts"] >= 30
                    and v["first_alert_precision"] >= 0.8
                    and v["false_alerts_per_drone_hour"] <= 5
                )
                candidates.append(
                    {
                        "head": head,
                        "enter_threshold": float(threshold),
                        "exit_threshold": float(threshold * ratio),
                        "qualifies": bool(okay),
                        **v,
                    }
                )
    valid = [r for r in candidates if r["qualifies"]]
    key = lambda r: (
        r["first_alert_event_recall_3s"],
        r["event_recall_3s"],
        -r["false_alerts_per_drone_hour"],
    )
    chosen = max(valid, key=key) if valid else None
    alternatives = {}
    for budget in (20, 60):
        v = [
            r
            for r in candidates
            if r["first_alerts"] >= 30
            and r["first_alert_precision"] >= 0.6
            and r["false_alerts_per_drone_hour"] <= budget
        ]
        alternatives[str(budget)] = max(v, key=key) if v else None
    return {
        "selected": chosen,
        "all_candidates": candidates,
        "alternative_budgets_separate": alternatives,
    }


def main():
    torch.set_num_threads(2)
    path, _ = selected()
    if (RESULT / "selected/alarm_policy.json").exists() and (
        RESULT / "policy_calibration.json"
    ).exists():
        if read_json(RESULT / "selected/alarm_policy.json")[
            "checkpoint_sha256"
        ] != sha256_file(path):
            raise ValueError("기존 경고 정책의 모델 불일치")
        print("frozen policy reused", flush=True)
        return
    model, state = load(path)
    cal = read_json(RESULT / "selected/calibration.json")
    oldcal = read_json(RESULT / "baseline_calibration.json")
    manifest, p = collection()
    _, records = validation_partition(
        [r for r in manifest["records"] if r["split"] == "val"]
    )
    maps = {e["key"]: e for e in p["maps"]}
    data = []
    old = []
    previous = None
    plan = {
        "model_sha256": sha256_file(path),
        "goals": GOALS,
        "thresholds": THRESHOLDS.tolist(),
        "exit_ratios": [0.25, 0.5, 0.75, 1.0],
        "fit": "21 disjoint validation-calibration full flights; no test",
        "same_grid_applied_to_baseline": True,
        "horizons_reported_separately": [5, 8],
    }
    dest = ROOT / "configs/advance_alarm_plan.json"
    if dest.exists() and read_json(dest) != plan:
        raise ValueError("경고 선정 계획 변경 금지")
    write_json(dest, plan)
    for r in records:
        ix = index_episode(r, p)
        x, m = load_observation(r)
        q = eligible(x, m, ix["index"])
        index = ix["index"][q]
        if previous != r["region"]:
            city = City(ROOT / maps[r["region"]]["path"], Scenario())
            previous = r["region"]
        z = score_records(
            r,
            index,
            {"default": (model, state, path)},
            city,
            RESULT / "selected/policy_scores",
        )["default"]
        sp = probabilities(z, cal)
        with np.load(ROOT / r["path"] / "truth.npz") as f:
            up = f["link_available"]
        data.append(
            {
                "up": up,
                "index": index,
                "scores": sp,
                "duration": r["metadata"]["duration_s"],
            }
        )
        with np.load(BASE / "alarm_calibration_scores" / (r["id"] + ".npz")) as f:
            np.testing.assert_array_equal(index, f["index"])
            bp = f["probability"]
        old.append(
            {
                "up": up,
                "index": index,
                "scores": {0: baseline_transform(bp, oldcal)},
                "duration": r["metadata"]["duration_s"],
            }
        )
        print("policy observations", r["id"], len(index), flush=True)
    out = {
        "checkpoint_sha256": sha256_file(path),
        "goals": GOALS,
        "fit_source": plan["fit"],
    }
    details = {}
    for horizon, heads in [(5, [0, 3, 5]), (8, [4, 6])]:
        heads = [h for h in heads if str(h) in cal["heads"]]
        new = optimize(data, heads, horizon)
        baseline = optimize(old, [0], horizon)
        out[str(horizon) + "s"] = new["selected"]
        details[str(horizon) + "s"] = {"new": new, "baseline_same_grid": baseline}
    write_json(RESULT / "selected/alarm_policy.json", out)
    write_json(RESULT / "policy_calibration.json", details)
    write_json(ROOT / "model/advance/selected/alarm_policy.json", out)
    persist("alarm_policy_frozen")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
