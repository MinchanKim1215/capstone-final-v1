"""현재 수신 관측과 현재 발행 목표만 이용한 입력. 미래 정답 조회 없음.

관측 40시점(0.5초) × 55항목, 과거 통계 660개, 물리·지도·목표 보조 137개.
미래 위치는 실제 좌표가 아닌 현재 속도/현재 명목 목표 유지 가정의 외삽.
"""

import numpy as np
from numba import njit
from swarm_sim.regions import ROOT, read_json
from swarm_sim.city import City
from swarm_sim.config import Scenario
from usable.train import CAUSAL_FEATURES
from scipy.special import ndtr

FEATURES = CAUSAL_FEATURES + list(range(48, 60))
H = np.asarray([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8], np.float64)
VERSION = "causal_goal_centered_float64_v3"


@njit(cache=True)
def command_observations(x, mask, desired):
    nt, n, _, _ = x.shape
    out = np.zeros((nt, n, n, 9), np.float32)
    last = np.zeros((n, n, 2))
    stamp = np.full((n, n), -1.0)
    for t in range(nt):
        segment = t // 6
        issued = segment * 3.0
        target = desired[segment + 1]
        now = (t + 1) * 0.5
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if mask[t, i, j, 33] and x[t, i, j, 33] > 0:
                    last[i, j] = target[j]
                    stamp[i, j] = issued
                dx = target[i, 0] - x[t, i, j, 12]
                dy = target[i, 1] - x[t, i, j, 13]
                out[t, i, j, 0] = dx
                out[t, i, j, 1] = dy
                out[t, i, j, 5] = max(0.0, 3.0 - (t % 6 + 1) * 0.5)
                out[t, i, j, 6] = np.sqrt(dx * dx + dy * dy)
                if (
                    stamp[i, j] < 0
                    or not mask[t, i, j, 34]
                    or not mask[t, i, j, 35]
                    or not mask[t, i, j, 30]
                    or x[t, i, j, 30] > 2
                ):
                    continue
                peerx = x[t, i, j, 12] + x[t, i, j, 34]
                peery = x[t, i, j, 13] + x[t, i, j, 35]
                dx = last[i, j, 0] - peerx
                dy = last[i, j, 1] - peery
                out[t, i, j, 2] = dx
                out[t, i, j, 3] = dy
                out[t, i, j, 4] = now - stamp[i, j]
                out[t, i, j, 7] = np.sqrt(dx * dx + dy * dy)
                out[t, i, j, 8] = 1
    return out


def observed_commands(r, x, m):
    with np.load(
        ROOT / "cache/usable" / f"{r['region']}_{r['metadata']['direction']}_motion.npz"
    ) as z:
        desired = z["desired"]
    return command_observations(x, m, desired)


@njit(cache=True)
def _path_positions(
    points, pred, a, velocity, end_nodes, goal_nodes, remaining, horizons
):
    out = np.zeros((len(a), len(horizons), 3), np.float64)
    for q in range(len(a)):
        for z in range(len(horizons)):
            h = horizons[z]
            out[q, z, 2] = 50
            if h <= remaining[q]:
                out[q, z, :2] = a[q, :2] + velocity[q, :2] * h
                continue
            elapsed = h - remaining[q]
            node = end_nodes[q]
            goal = goal_nodes[q]
            while elapsed >= 3 and node != goal:
                next_ = pred[goal, node]
                if next_ < 0:
                    break
                node = next_
                elapsed -= 3
            next_ = node if node == goal else pred[goal, node]
            if next_ < 0:
                next_ = node
            fraction = min(elapsed / 3.0, 1.0)
            out[q, z, :2] = points[node] * (1 - fraction) + points[next_] * fraction
    return out


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
        for q in range(len(H)):
            loss, los, _ = city.radio.propagation(pa[:, q], pb[:, q], Scenario())
            z[:, q] = loss - now
            z[:, 13 + q] = los
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


def physics(values, mask, look):
    c = {f: FEATURES.index(f) for f in (28, 29, 31, 40, 41, 48)}
    get = lambda f: values[..., c[f]]
    valid = mask[:, -1, [c[f] for f in (28, 29, 31, 41, 48)]].all(1) & (
        look[:, -1] > 0.5
    )
    p = get(28)
    sn = get(29)
    rev = get(31)
    loss = get(48)
    shadow = np.clip(10 - loss[:, -1] - p[:, -1], -6, 6)
    den = np.clip(p - sn, -110, -40)
    rden = np.clip(p - rev, -110, -40)
    dk = mask[..., c[28]] & mask[..., c[29]]
    rk = mask[..., c[28]] & mask[..., c[31]]
    sk = mask[..., c[28]] & mask[..., c[48]]
    rho = np.exp(-H / 4)
    margin = get(41)[:, -1, None] - look[:, :13] + shadow[:, None] * (1 - rho[None])
    sigma = np.sqrt(1.5**2 * (1 - rho**2) + 0.5**2)
    risk = ndtr(-margin / sigma[None]).astype(np.float32)
    first = H <= 5
    early = (H >= 3) & (H <= 4)
    before = H < 3
    proxy = risk[:, early].max(1) * (1 - risk[:, before].max(1))
    residual = 10 - loss - p

    def mean(a, k):
        return np.where(k[:, -10:], a[:, -10:], 0).sum(1) / np.maximum(
            k[:, -10:].sum(1), 1
        )

    def change(a, k):
        return np.where(k[:, -1] & k[:, -10], a[:, -1] - a[:, -10], 0)

    tail = np.c_[
        shadow,
        den[:, -1],
        rden[:, -1],
        mean(den, dk),
        mean(rden, rk),
        change(den, dk),
        change(rden, rk),
        np.clip(change(residual, sk), -12, 12),
        risk[:, first].max(1),
        proxy,
    ]
    out = np.concatenate(
        (margin.astype(np.float32), risk, tail.astype(np.float32)), axis=-1
    )
    out[~valid] = 0
    return out


def normalize_sequence(x, m, normalizer):
    mean = np.asarray(normalizer["mean"], np.float32)[FEATURES]
    std = np.asarray(normalizer["std"], np.float32)[FEATURES]
    return np.where(
        m,
        np.clip((np.asarray(x, np.float16).astype(np.float32) - mean) / std, -8, 8),
        0,
    ).astype(np.float32)


def statistics(x, m):
    a = np.ascontiguousarray(x[:, 1::2], dtype=np.float64)
    known = np.ascontiguousarray(m[:, 1::2], dtype=bool)
    v = np.where(known, a, 0)
    parts = [v[:, -1], known[:, -1].astype(np.float64)]
    for length in (5, 20):
        xx = v[:, -length:]
        ok = known[:, -length:]
        count = np.maximum(ok.sum(1), 1)
        mean = xx.sum(1) / count
        std = np.sqrt(np.where(ok, (xx - mean[:, None]) ** 2, 0).sum(1) / count)
        low = np.where(ok, xx, np.inf).min(1)
        high = np.where(ok, xx, -np.inf).max(1)
        low[~np.isfinite(low)] = 0
        high[~np.isfinite(high)] = 0
        change = np.where(ok[:, -1] & ok[:, 0], xx[:, -1] - xx[:, 0], 0)
        parts.extend([mean, std, low, high, change])
    return np.concatenate(parts, -1).astype(np.float32)


def windows(x, m, index, normalizer):
    k, i, j = index.T
    t = k[:, None] + np.arange(-39, 1)
    raw = x[t, i[:, None], j[:, None]][:, :, FEATURES]
    mask = m[t, i[:, None], j[:, None]][:, :, FEATURES]
    quant = np.asarray(raw, np.float16).astype(np.float32)
    return normalize_sequence(quant, mask, normalizer), mask, quant


def extras(city, x, m, index, raw, mask, intent):
    k, i, j = index.T
    look = geometry_profile(city, x[k, i, j, :48], m[k, i, j, :48])
    phy = physics(raw, mask, look)
    goal = geometry_profile(city, x[k, i, j, :48], m[k, i, j, :48], intent[k, i, j])
    return np.concatenate((look, phy, goal, intent[k, i, j]), -1)


def read_record(r):
    path = ROOT / r["path"]
    with np.load(path / "observations.npz") as z:
        x, m = z["features"], z["mask"]
    intent = observed_commands(r, x, m)
    with np.load(path / "spatial_observations.npz") as z:
        x = np.concatenate((x, z["features"]), -1)
        m = np.concatenate((m, z["mask"]), -1)
    return x, m, intent


def targets(up, sinr, index):
    from usable.beacons import data_threshold_db

    k, i, j = index.T
    bad = ~up
    start = bad.copy()
    start[1:] &= up[:-1]
    start[:-1] &= bad[1:]
    start[-1] = False
    onset = start[k[:, None] + np.arange(1, 21), i[:, None], j[:, None]]
    lead = np.arange(20) * 0.5
    pair = bad[:-1] & bad[1:]
    yp5 = np.any(pair[k[:, None] + np.arange(1, 10), i[:, None], j[:, None]], 1)
    yp10 = np.any(pair[k[:, None] + np.arange(1, 20), i[:, None], j[:, None]], 1)
    aux = np.clip(
        (
            np.min(sinr[k[:, None] + np.arange(1, 11), i[:, None], j[:, None]], 1)
            - data_threshold_db()
        )
        / 10,
        -3,
        3,
    )
    flags = [
        np.any(onset[:, choose], 1)
        for choose in (
            (lead >= 3) & (lead <= 4),
            (lead >= 3) & (lead <= 7),
            lead <= 4,
            lead <= 7,
        )
    ]
    return np.column_stack((yp5, yp10, aux, *flags)).astype(np.float32)
