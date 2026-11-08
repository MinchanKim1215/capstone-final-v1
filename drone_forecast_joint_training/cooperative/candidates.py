"""로컬 수신 정보와 지도에 한정한 안전 후보·연결성 신념 계산.

다른 드론의 실제 위치·실제 채널·미래 간섭을 인자로 받지 않습니다.
자기 명목 목표는 비행 계획기에서 제공하는 제어 명세입니다.
"""

import numpy as np
import shapely
from scipy.special import expit
from usable.beacons import data_threshold_db

# 기준 보정 0을 항상 0번에 배치. 나머지는 축별 ±40m의 제약 후보.
OFFSETS = np.array(
    [[0.0, 0.0]]
    + [
        [x, y]
        for x in [-40.0, -20.0, 0.0, 20.0, 40.0]
        for y in [-40.0, -20.0, 0.0, 20.0, 40.0]
        if x or y
    ]
)
N_ACTIONS = len(OFFSETS)
CANDIDATE_DIM = 12
THRESHOLD = data_threshold_db()


def widest(margin):
    """최대-최소 경로 폐포: 경로 중 최소 링크 여유가 가장 큰 연결."""
    value = margin.copy()
    n = value.shape[-1]
    for k in range(n):
        value = np.maximum(
            value, np.minimum(value[..., :, k, None], value[..., None, k, :])
        )
    return value[..., [0, 0, 1], [1, 2, 2]]


def local_candidates(
    city, observer, values, mask, nominal, time_s, risk, early, predictive=True
):
    n = 8
    i = observer
    x = np.asarray(values)
    m = np.asarray(mask)
    other = np.array([j for j in range(n) if j != i])
    own = x[other[0], 12:14].astype(float)
    vel = x[other[0], :2].astype(float)
    peer = np.tile(own, (n, 1))
    peer[i] = own
    peer_v = np.tile(vel, (n, 1))
    seen = np.zeros(n, bool)
    seen[i] = True
    age = np.full(n, 10.0)
    age[i] = 0.0
    for j in other:
        if m[j, [30, 34, 35, 36, 37]].all() and x[j, 30] <= 10:
            peer[j] = own + x[j, 34:36]
            peer_v[j] = vel + x[j, 36:38]
            seen[j] = True
            age[j] = x[j, 30]
        elif m[j, [15, 16, 17, 18, 19]].all() and x[j, 15] <= 10:
            peer[j] = own + x[j, 16:18]
            peer_v[j] = vel + x[j, 18:20]
            seen[j] = True
            age[j] = x[j, 15]
    peer[:, :2] = np.clip(peer[:, :2], -195, 195)
    obs_noise = np.full(n, -107.0)
    inbound = []
    for j in other:
        if m[j, [28, 29, 30]].all() and x[j, 30] <= 2:
            inbound.append(float(x[j, 28] - x[j, 29]))
        if m[j, [28, 31, 32]].all() and x[j, 32] <= 2:
            obs_noise[j] = np.clip(x[j, 28] - x[j, 31], -110, -40)
    if inbound:
        obs_noise[i] = np.clip(np.median(inbound), -110, -40)
    h = np.array([3.0, 6.0]) if predictive else np.array([3.0, 3.0])
    poses = peer[None] + (
        peer_v[None] * h[:, None, None] if predictive else np.zeros((2, n, 2))
    )
    poses = np.clip(poses, -195, 195)
    poses = np.concatenate((poses, np.full((2, n, 1), 50.0)), axis=-1)
    tx = np.repeat(poses, n, axis=1).reshape(-1, 3)
    rx = np.tile(poses, (1, n, 1)).reshape(-1, 3)
    base_loss = city.radio.propagation(tx, rx, city._cooperative_scenario)[0].reshape(
        2, n, n
    )
    base = (
        10
        - base_loss
        - np.maximum(obs_noise[None, :, None], obs_noise[None, None, :])
        - THRESHOLD
    )
    base -= 0.15 * np.maximum(age[None, :, None], age[None, None, :])
    base[:, ~seen, :] = -30
    base[:, :, ~seen] = -30
    base[:, np.arange(n), np.arange(n)] = 30
    candidate = nominal[None] + OFFSETS
    valid = np.all(np.abs(candidate) <= 195.00001, axis=1) & ~shapely.intersects_xy(
        city.flight_blocker, candidate[:, 0], candidate[:, 1]
    )
    valid[0] = True
    current_node = city.nearest(own)
    goal = city.tree.query(np.clip(candidate, -195, 195))[1]
    nextnode = city.predecessor[goal, current_node].copy()
    nextnode = np.where(
        (nextnode < 0) | (goal == current_node), current_node, nextnode
    ).astype(int)
    after = city.predecessor[goal, nextnode].copy()
    after = np.where((after < 0) | (goal == nextnode), nextnode, after).astype(int)
    future = np.stack((city.points[nextnode], city.points[after]))
    own3 = np.concatenate((future, np.full((2, N_ACTIONS, 1), 50.0)), axis=-1)
    a = np.repeat(own3, n, axis=1).reshape(-1, 3)
    b = np.tile(poses[:, None], (1, N_ACTIONS, 1, 1)).reshape(-1, 3)
    loss = city.radio.propagation(a, b, city._cooperative_scenario)[0].reshape(
        2, N_ACTIONS, n
    )
    shadow = np.zeros(n)
    a_now = np.c_[np.tile(own, (n, 1)), np.full(n, 50.0)]
    b_now = np.c_[peer, np.full(n, 50.0)]
    now_loss = city.radio.propagation(a_now, b_now, city._cooperative_scenario)[0]
    for j in other:
        if m[j, [28, 30]].all() and x[j, 30] <= 2:
            shadow[j] = np.clip(10 - now_loss[j] - x[j, 28], -6, 6)
    margins = 10 - loss - np.maximum(obs_noise[i], obs_noise)[None, None, :] - THRESHOLD
    margins -= shadow[None, None, :] * np.exp(-h[:, None, None] / 4)
    margins -= 0.15 * age[None, None, :]
    margins[:, :, ~seen] = -30
    margins[:, :, i] = 30
    graphs = np.tile(base[:, None], (1, N_ACTIONS, 1, 1))
    graphs[:, :, i, :] = margins
    graphs[:, :, :, i] = margins
    paths = widest(graphs)
    b3 = paths[0].min(-1)
    b6 = paths[1].min(-1)
    mean3 = paths[0].mean(-1)
    mean6 = paths[1].mean(-1)
    r = np.nan_to_num(np.asarray(risk, float), nan=0.0)
    ev = np.nan_to_num(np.asarray(early, float), nan=0.0)
    if not predictive:
        r[:] = 0
        ev[:] = 0
    weights = 0.15 + r + ev
    improvement = (margins[1] - margins[1, 0]) * weights[None]
    weighted = improvement[:, other].mean(1)
    displacement = np.linalg.norm(OFFSETS, axis=1) / 40.0
    deviation = np.linalg.norm(future[0] - nominal, axis=1) / 40.0
    travel = np.linalg.norm(future[0] - own, axis=1) / 15.0
    current_local = np.array([x[j, 41] if m[j, 41] else -20 for j in other])
    cand = np.column_stack(
        (
            np.clip(b3, -20, 15) / 10,
            np.clip(b6, -20, 15) / 10,
            np.clip(mean3, -20, 15) / 10,
            np.clip(mean6, -20, 15) / 10,
            np.clip(margins[0, :, :3].min(-1), -20, 15) / 10,
            np.clip(weighted, -30, 30) / 10,
            displacement,
            deviation,
            travel,
            np.full(N_ACTIONS, seen[:3].mean()),
            OFFSETS / 40.0,
        )
    ).astype(np.float32)
    per = []
    for j in other:
        margin = x[j, 41] / 10 if m[j, 41] else 0.0
        per.extend(
            [
                *(peer[j] - own) / 200,
                *(peer_v[j] - vel) / 5,
                margin,
                age[j] / 10,
                r[j],
                ev[j],
                float(np.isfinite(risk[j])),
                float(seen[j]),
                float(j < 3),
            ]
        )
    context = np.r_[
        own / 200,
        vel / 5,
        (nominal - own) / 40,
        time_s / 3000.0,
        np.eye(5)[i - 3],
        x[other[0], 11] / 64.0,
        seen[:3].mean(),
        per,
    ].astype(np.float32)
    context = np.clip(context, -8, 8)
    cand = np.clip(cand, -8, 8)
    # 이웃 위치가 전혀 없는 초기 구간은 명목 비행으로 복귀.
    if np.sum(seen[:3]) < 1:
        valid[1:] = False
    score = (
        0.65 * np.clip(b3, -10, 6)
        + 0.35 * np.clip(b6, -10, 6)
        + 0.1 * np.clip(mean6, -10, 6)
        + 0.15 * weighted
        - 0.16 * displacement
        - 0.03 * travel
    )
    score[~valid] = -1e6
    return context, cand, valid, score


def observe_policy(env, predictive=True, use_risk=True):
    env.city._cooperative_scenario = env.scenario
    contexts = []
    candidates = []
    masks = []
    scores = []
    for i in range(3, 8):
        r = env.risk[i].copy() if use_risk else np.full(8, np.nan)
        er = env.early[i].copy() if use_risk else np.full(8, np.nan)
        a, b, c, d = local_candidates(
            env.city,
            i,
            env.last_x[i],
            env.last_mask[i],
            env.proposal["nominal"][i],
            env.k * 3.0,
            r,
            er,
            predictive,
        )
        if env.proposal["phase"] == 2:
            c[1:] = False
            d[1:] = -1e6
        contexts.append(a)
        candidates.append(b)
        masks.append(c)
        scores.append(d)
    return (
        np.asarray(contexts),
        np.asarray(candidates),
        np.asarray(masks),
        np.asarray(scores),
    )


def indices_to_correction(indices):
    a = np.asarray(indices)
    if (
        a.shape != (5,)
        or not np.isfinite(a).all()
        or np.any(a != a.astype(int))
        or np.any((a < 0) | (a >= N_ACTIONS))
    ):
        raise ValueError("중계 5대의 0~24 정수 행동 필요")
    out = np.zeros((8, 2))
    out[3:] = OFFSETS[a.astype(int)]
    return out


def heuristic_action(env, predictive=True, use_risk=True):
    obs = observe_policy(env, predictive, use_risk)
    return np.argmax(obs[-1], axis=1), obs
