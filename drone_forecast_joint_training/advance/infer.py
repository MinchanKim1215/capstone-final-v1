"""정답 파일을 읽지 않는 예측 공통부. 코드·가중치 지문으로 캐시 검사."""

from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
from scipy.special import expit
from swarm_sim.regions import ROOT, read_json, sha256_file
from advance.features import VERSION, read_record, windows, extras, statistics
from advance.data import DATA, RESULT
from advance.train import ResidualTransformer, RUNS


class TreeModel:
    def __init__(self, state):
        self.trees = {
            int(h): lgb.Booster(model_str=s) for h, s in state["trees"].items()
        }

    def predict(self, f):
        out = np.zeros((len(f), 7), np.float32)
        for h, b in self.trees.items():
            out[:, h] = b.predict(f, raw_score=True, num_threads=1).astype(np.float32)
        return out


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["feature_version"] != VERSION:
        raise ValueError("입력 버전 불일치")
    if state["architecture"] == "gbdt":
        return TreeModel(state), state
    model = ResidualTransformer(state["base_spec"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state


def feature_batch(state, city, x, m, index, intent):
    z, known, raw = windows(x, m, index, state["normalizer"])
    e = extras(city, x, m, index, raw, known, intent)
    norm = state["extra_normalizer"]
    e = np.clip(
        (e - np.asarray(norm["mean"], np.float32))
        / np.asarray(norm["std"], np.float32),
        -8,
        8,
    ).astype(np.float32)
    s = statistics(z, known)
    return z, known, e, s


@torch.inference_mode()
def predict_batch(model, state, z, m, e, s):
    if state["architecture"] == "gbdt":
        return model.predict(np.concatenate((s, e), -1))
    out = np.empty((len(z), 7), np.float32)
    for first in range(0, len(z), 512):
        stop = min(first + 512, len(z))
        out[first:stop] = model(
            torch.from_numpy(z[first:stop]),
            torch.from_numpy(m[first:stop]),
            torch.from_numpy(e[first:stop]),
            torch.from_numpy(s[first:stop]),
        ).numpy()
    return out


def selected(reference=False):
    item = read_json(RESULT / "selection.json")[
        "transformer_reference" if reference else "selected"
    ]
    path = RUNS / item["run"] / item["checkpoint"]
    if sha256_file(path) != item["checkpoint_sha256"]:
        raise ValueError("선정 가중치 동결 위반")
    return path, item


def score_records(r, index, models, city, cache):
    """여러 모델의 동일 입력 재사용. 모델별 예측 평균 없음."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    file = cache / (r["id"] + ".npz")
    sign = {name: sha256_file(path) for name, (_, _, path) in models.items()}
    version = sha256_file(ROOT / "advance/features.py")
    if file.exists():
        with np.load(file) as z:
            if str(z["sign"]) != str(sign) or str(z["feature_source"]) != version:
                raise ValueError("예측 캐시 입력/가중치 변경")
            np.testing.assert_array_equal(z["index"], index)
            return {name: z[name] for name in models}
    x, m, intent = read_record(r)
    out = {name: np.empty((len(index), 7), np.float32) for name in models}
    shared = next(iter(models.values()))[1]
    for name, (_, state, _) in models.items():
        if (
            state["normalizer"] != shared["normalizer"]
            or state["extra_normalizer"] != shared["extra_normalizer"]
        ):
            raise ValueError("공통 입력 보정값 불일치")
    for first in range(0, len(index), 2048):
        stop = min(first + 2048, len(index))
        z, known, e, s = feature_batch(shared, city, x, m, index[first:stop], intent)
        for name, (model, state, _) in models.items():
            out[name][first:stop] = predict_batch(model, state, z, known, e, s)
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("비유한 예측")
    temp = file.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp, index=index, sign=str(sign), feature_source=version, **out
    )
    temp.replace(file)
    return out


def probabilities(z, cal):
    out = {
        int(h): expit(v["a"] * z[:, int(h)] + v["b"]).astype(np.float32)
        for h, v in cal["heads"].items()
    }
    if 3 in out:
        out[3] = np.minimum(out[3], out[0])
    if 5 in out:
        out[5] = np.minimum(out[5], out[0])
    if 4 in out and 6 in out:
        out[4] = np.minimum(out[4], out[6])
    return out


def eligible(x, m, index):
    k, i, j = index.T
    return (
        m[k, i, j, 46] & (x[k, i, j, 46] > 0.5) & m[k, i, j, 41] & (x[k, i, j, 41] > 0)
    )
