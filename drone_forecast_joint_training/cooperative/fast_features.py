"""고정 예측기의 동일 기하 연산을 묶어서 실행하는 최적화. 예측 수식/가중치 변경 없음."""

import numpy as np
from swarm_sim.config import Scenario
from advance.features import H, _path_positions, windows, physics, statistics


def geometry_profile(city, x, mask, intent=None):
    """13개 중간 시점의 손실 변화·LoS·측정 여유. 측정값은 기하 캐시에서 제외."""
    out = np.zeros((len(x), 46), np.float32)
    good = mask[:, [0, 1, 2, 12, 13, 30, 34, 35, 36, 37, 41]].all(1) & (x[:, 30] <= 2)
    if intent is not None:
        good &= intent[:, 8] > 0.5
    if not good.any():
        return out
    v = x[good]
    a = np.c_[v[:, 12:14], v[:, 2]]
    b = a + np.c_[v[:, 34:36], np.zeros(len(v))]
    va = np.c_[v[:, :2], np.zeros(len(v))]
    vb = va + np.c_[v[:, 36:38], np.zeros(len(v))]
    packed = np.c_[a, b, va, vb]
    if intent is not None:
        packed = np.c_[packed, intent[good, :4], intent[good, 5]]
    unique, inverse = np.unique(packed, axis=0, return_inverse=True)
    cache_name = "_goal_profile_cache" if intent is not None else "_cv_profile_cache"
    if not hasattr(city, cache_name):
        setattr(city, cache_name, {})
    cache = getattr(city, cache_name)
    if len(cache) > 120000:
        cache.clear()
    keys = [u.tobytes() for u in unique]
    geo = np.empty((len(unique), 26), np.float32)
    missing = []
    for q, key in enumerate(keys):
        if key in cache:
            geo[q] = cache[key]
        else:
            missing.append(q)
    for start in range(0, len(missing), 4096):
        take = missing[start : start + 4096]
        u = unique[take]
        aa, bb, v1, v2 = u[:, :3], u[:, 3:6], u[:, 6:9], u[:, 9:12]
        now, _, _ = city.radio.propagation(aa, bb, Scenario())
        if intent is None:
            pa = aa[:, None] + H[None, :, None] * v1[:, None]
            pb = bb[:, None] + H[None, :, None] * v2[:, None]
            pa[:, :, :2] = np.clip(pa[:, :, :2], -195, 195)
            pb[:, :, :2] = np.clip(pb[:, :, :2], -195, 195)
        else:
            c = u[:, 12:]
            remaining = c[:, 4]
            end_a = np.clip(aa[:, :2] + v1[:, :2] * remaining[:, None], -195, 195)
            end_b = np.clip(bb[:, :2] + v2[:, :2] * remaining[:, None], -195, 195)
            ea = city.tree.query(end_a)[1].astype(np.int32)
            eb = city.tree.query(end_b)[1].astype(np.int32)
            ga = city.tree.query(np.clip(aa[:, :2] + c[:, :2], -195, 195))[1].astype(
                np.int32
            )
            gb = city.tree.query(np.clip(bb[:, :2] + c[:, 2:4], -195, 195))[1].astype(
                np.int32
            )
            pa = _path_positions(
                city.points, city.predecessor, aa, v1, ea, ga, remaining, H
            )
            pb = _path_positions(
                city.points, city.predecessor, bb, v2, eb, gb, remaining, H
            )
        z = np.zeros((len(u), 26), np.float32)
        loss, los, _ = city.radio.propagation(
            pa.reshape(-1, 3), pb.reshape(-1, 3), Scenario()
        )
        z[:, :13] = loss.reshape(len(u), 13) - now[:, None]
        z[:, 13:26] = los.reshape(len(u), 13)
        geo[take] = z
        for k, row in zip(take, z):
            cache[keys[k]] = row.copy()
    z = np.zeros((len(v), 46), np.float32)
    z[:, :26] = geo[inverse]
    z[:, 26:39] = v[:, 41, None] - z[:, :13]
    for q, (low, high) in enumerate([(0, 5), (3, 4), (3, 7)]):
        choose = (H >= low) & (H <= high)
        z[:, 39 + 2 * q] = z[:, :13][:, choose].max(1)
        z[:, 40 + 2 * q] = z[:, 26:39][:, choose].min(1)
    z[:, -1] = 1
    out[good] = z
    return out


def extras(city, x, m, index, raw, mask, intent):
    k, i, j = index.T
    look = geometry_profile(city, x[k, i, j, :48], m[k, i, j, :48])
    phy = physics(raw, mask, look)
    goal = geometry_profile(city, x[k, i, j, :48], m[k, i, j, :48], intent[k, i, j])
    return np.concatenate((look, phy, goal, intent[k, i, j]), -1)


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
    return z, known, e, statistics(z, known)
