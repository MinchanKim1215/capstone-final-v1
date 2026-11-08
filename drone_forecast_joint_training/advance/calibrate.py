"""독립 보정 검증 창에서만 확률과 창 단위 동작점 고정."""

import shutil, json
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import precision_recall_curve
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from swarm_sim.forecast_evaluation import binary_metrics
from upgrade.evaluate import threshold_at_fpr
from advance.data import DATA, RESULT, BASE
from advance.train import bank, nn_predict, persist
from advance.infer import load, selected, probabilities


def fit(y, z):
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    def obj(v):
        zz = np.exp(v[0]) * z + v[1]
        return np.mean(np.logaddexp(0, zz) - y * zz) + 1e-5 * np.sum(v * v)

    result = minimize(obj, np.zeros(2), method="L-BFGS-B", bounds=[(-3, 3), (-5, 5)])
    a = float(np.exp(result.x[0]))
    b = float(result.x[1])
    p = expit(a * z + b).astype(np.float32)
    tight = threshold_at_fpr(y, p, 0.02)
    precision, recall, cut = precision_recall_curve(y, p)
    okay = (precision[:-1] >= 0.7) & (cut >= tight) & (recall[:-1] > 0)
    threshold = float(cut[np.flatnonzero(okay)[0]]) if okay.any() else None
    return {
        "a": a,
        "b": b,
        "threshold_balanced": threshold_at_fpr(y, p, 0.05),
        "threshold_conservative": threshold,
        "fit_success": bool(result.success),
        "validation_metrics": binary_metrics(
            y, p, threshold if threshold is not None else 1.1
        ),
    }


def baseline_transform(p, cal):
    return expit(cal["a"] * logit(np.clip(p, 1e-7, 1 - 1e-7)) + cal["b"]).astype(
        np.float32
    )


def main():
    torch.set_num_threads(2)
    if all(
        (RESULT / p).exists()
        for p in [
            "selected/calibration.json",
            "transformer_reference/calibration.json",
            "baseline_calibration.json",
            "baseline_validation_metrics.json",
        ]
    ):
        for name, reference in [("selected", False), ("transformer_reference", True)]:
            path, _ = selected(reference)
            if read_json(RESULT / name / "calibration.json")[
                "checkpoint_sha256"
            ] != sha256_file(path):
                raise ValueError("기존 보정의 모델 불일치")
        print("frozen calibration reused", flush=True)
        return
    data = bank("calibrate")
    y = np.asarray(data["y"])
    items = {}
    for name, reference in [("selected", False), ("transformer_reference", True)]:
        path, info = selected(reference)
        model, state = load(path)
        if state["architecture"] == "gbdt":
            z = model.predict(data["features"])
        else:
            z = nn_predict(model, data, state["normalizer"])
        heads = state.get("trained_heads", [0, 1, 3, 4, 5, 6])
        cal = {
            "checkpoint_sha256": sha256_file(path),
            "heads": {str(h): fit(y[:, h], z[:, h]) for h in heads},
            "fit": "114688 windows from disjoint 21 validation-calibration flights only",
            "goals": {"precision_min": 0.7, "fpr_max": 0.02},
            "goals_are_project_choices": True,
            "method": "positive slope logistic; subset output caps for early-event heads",
        }
        dest = RESULT / name
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest / "calibration.json", cal)
        np.savez_compressed(dest / "calibration_predictions.npz", logits=z, targets=y)
        export = ROOT / "model/advance" / name
        export.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, export / "best.pt")
        shutil.copyfile(dest / "calibration.json", export / "calibration.json")
        write_json(export / "selection.json", info)
        print("calibrated", name, cal["heads"]["0"]["validation_metrics"], flush=True)
    # 같은 보정 창에서 기존 B3도 다시 보정. 원본 가중치와 순위는 유지.
    from usable.evaluate import load as old_load, predict_episode

    model, state, _ = old_load(BASE)
    oldcal = read_json(BASE / "calibration.json")
    meta = read_json(DATA / "calibrate/bank.json")
    index = np.load(DATA / "calibrate/selected.npy")
    z = np.empty(len(index), np.float32)
    for ep, r in enumerate(meta["records"]):
        q = np.flatnonzero(index[:, 0] == ep)
        z[q] = predict_episode(model, state, r, index[q, 1:]) / oldcal["temperature"]
    oldp = expit(z).astype(np.float32)
    cal = fit(y[:, 0], logit(np.clip(oldp, 1e-7, 1 - 1e-7)))
    cal.update(
        checkpoint_sha256=sha256_file(BASE / "best.pt"),
        fit="same 114688 calibration windows; old stored probability logit input",
    )
    write_json(RESULT / "baseline_calibration.json", cal)
    meta = read_json(DATA / "select/bank.json")
    index = np.load(DATA / "select/selected.npy")
    yy = np.load(DATA / "select/targets.npy")
    score = np.empty(len(index), np.float32)
    for ep, r in enumerate(meta["records"]):
        q = np.flatnonzero(index[:, 0] == ep)
        score[q] = expit(
            predict_episode(model, state, r, index[q, 1:]) / oldcal["temperature"]
        )
    write_json(
        RESULT / "baseline_validation_metrics.json",
        {
            "main": binary_metrics(yy[:, 0], score, 0.5),
            "early3_score": binary_metrics(yy[:, 3], score, 0.5),
            "same_validation_select_rows": True,
        },
    )
    persist("probability_calibration_frozen")


if __name__ == "__main__":
    main()
