"""개선안 1. 단절 중 복구 재배치.

V18(forecast_avoidance.controller.choose)은 수색기 전원의 비콘이 3초 안에 들어왔을 때만
판단한다(insufficient_observation). 단절이 3초를 넘기면 끊긴 수색기가 "모름"이 되어 복구
판단 자체를 건너뛴다. 이 모듈은 그 경우에만 실행되는 복구 전용 판단이다.

* 끊긴 수색기의 위치는 공유 항법 계획(planner.path → forecast_navigation.preview)의 예정
  위치를 쓴다. V18도 같은 계획으로 경로를 미리 본다.
* 끊긴 시간이 길수록 그 수색기의 링크 여유를 낮춰(위치 불확실성) 보수적으로 판단하고,
  허용 이동 폭을 줄인다.
* 후보는 "끊긴 수색기까지 가장 넓은 경로의 여유"가 가장 빨리 양수(반초 두 번 연속)가 되는
  순서로 고른다.
* 명령을 받을 수 있는 중계기(첫 중계기 자신과 3초 안에 비콘이 들어온 중계기)만 움직인다.
  명령을 못 받는 중계기는 개선안 2(transport.FallbackChannel)가 담당한다.

RadioState.known 자체는 바꾸지 않는다. known은 V21 LightGBM 입력(missing_search)에도
쓰이므로, 바꾸면 학습된 예측기의 입력 분포가 달라진다.
"""
import copy
from dataclasses import replace

import numpy as np

from adaptive_joint_expanded.features import OFFSETS
from forecast_avoidance.controller import TIMES

from .graph import closure, finite, first_sustained, lost_margin, pair_margin


def lost_searchers(state, n_search):
    """첫 중계기가 3초 넘게 비콘을 받지 못한 수색기."""
    return np.flatnonzero(~np.asarray(state.known[:n_search], dtype=bool))


def commandable(state, n_search, n_relay):
    """명령을 보낼 수 있다고 볼 중계기. 0번은 명령을 내는 첫 중계기 자신이다."""
    ok = np.zeros(n_relay, dtype=bool)
    ok[0] = True
    ok[1:] = np.asarray(state.known[n_search+1:n_search+n_relay], dtype=bool)
    return ok


def conservative_noise(state, lost):
    """끊긴 수색기의 잡음은 측정이 없으므로 측정된 잡음 중 가장 큰 값을 쓴다 (-107dBm 낙관 방지)."""
    noise = np.array(state.noise, dtype=float, copy=True)
    measured = np.asarray(state.known, dtype=bool).copy()
    measured[lost] = False
    if measured.any():
        noise[lost] = np.maximum(noise[lost], noise[measured].max())
    return noise


def candidate_moves(relay, base, logits, valid, original, lost, n_search, cap, toward):
    """PPO 상위 후보 + 끊긴 수색기 쪽 후보. 허용 이동 폭을 넘는 보정 변화는 제외한다."""
    current = int(base[relay])
    delta = OFFSETS-OFFSETS[current]
    size = np.linalg.norm(delta, axis=1)
    allowed = np.asarray(valid[relay], dtype=bool) & (size <= cap+1e-6)
    allowed[current] = False
    ranked = np.argsort(-np.asarray(logits[relay]), kind="stable")
    shortlist = [int(a) for a in ranked if allowed[a]][:7]
    if allowed[0] and 0 not in shortlist:
        shortlist.append(0)
    me = original[1, n_search+relay]
    target = lost[int(np.argmin(np.linalg.norm(original[1, lost]-me, axis=-1)))]
    direction = original[min(2, len(original)-1), target]-me
    length = np.linalg.norm(direction)
    if length > 1e-6 and toward > 0:
        cosine = np.where(size > 0, delta @ direction/np.maximum(size, 1e-9)/length, -1.)
        order = np.lexsort((size, -cosine))
        shortlist += [int(a) for a in order if allowed[a] and cosine[a] > .3][:toward]
    return list(dict.fromkeys(shortlist))


def model_says_cut(planner, original, lost):
    """불확실성 보정(나이 감점, 보수적 잡음) 없이 계산한 지금 시점의 재연결 여유가 음수인가."""
    _, graphs = planner.radio([original])
    margin = lost_margin(closure(graphs), lost, planner.design.n_search)
    return bool(margin[0, 0] < 0.), float(margin[0, 0])


def cut_recovery(planner, base, logits, valid, phase, now, options, persistent=False):
    """반환: (행동, 사건 또는 None, 이유). persistent: 직전 판단 주기에도 같은 수색기 소식이 끊겼는가."""
    state = planner.state
    ns = planner.design.n_search
    lost = lost_searchers(state, ns)
    if not len(lost):
        return base, None, "no_lost_search"
    if phase == 2:
        return base, None, "final_phase"
    evidence = options.cut_evidence
    if evidence in ("persistent", "both") and not persistent:
        return base, None, "no_persistent_loss"
    original = planner.path(base)
    plain_margin = None
    if evidence in ("model", "both"):
        cut, plain_margin = model_says_cut(planner, original, lost)
        if not cut:
            return base, None, "model_sees_connection"
    reachable = commandable(state, ns, len(base))
    age = np.maximum(np.asarray(state.age, dtype=float)[lost], options.min_lost_age_s)
    oldest = float(age.max())
    cap = options.move_cap_m(oldest)
    paths, actions, relays = [original], [base], [None]
    seen = {original.tobytes()}
    for relay in np.flatnonzero(reachable):
        relay = int(relay)
        for action in candidate_moves(relay, base, logits, valid, original, lost, ns, cap,
                                      options.toward_moves):
            candidate = base.copy()
            candidate[relay] = action
            path = planner.path(candidate)
            key = path.tobytes()
            # 실제로 이번 3초 구간의 위치가 달라지는 후보만 남긴다 (V18과 같은 조건).
            if key in seen or np.array_equal(path[1], original[1]):
                continue
            seen.add(key)
            paths.append(path)
            actions.append(candidate)
            relays.append(relay)
    if len(paths) == 1:
        return base, None, "no_distinct_feasible_move"
    shadow = copy.copy(planner)
    if options.conservative_lost_noise:
        shadow.state = replace(state, noise=conservative_noise(state, lost))
    _, graphs = shadow.radio(paths)
    graphs = np.array(graphs, dtype=float, copy=True)
    penalty = options.lost_age_db*age[:, None]+options.lost_horizon_db*TIMES[None, :]
    for k, l in enumerate(lost):
        graphs[:, :, l, :] -= penalty[k][None, :, None]
        graphs[:, :, :, l] -= penalty[k][None, :, None]
    close = closure(graphs)
    reconnect = lost_margin(close, lost, ns)
    when = first_sustained(reconnect)
    deficit = np.maximum(0., -reconnect[:, 1:]).sum(axis=-1)*.5
    safe = np.ones(len(paths), dtype=bool)
    kept = np.setdiff1d(np.arange(ns), lost)
    if len(kept) >= 2:
        healthy = pair_margin(close, kept)
        safe = (healthy[:, 1:6] >= np.minimum(healthy[0, 1:6], 0.)-options.early_drop_db).all(axis=-1)
    earlier = when < when[0]
    progress = (when == when[0]) & (deficit <= deficit[0]-options.min_deficit_gain)
    feasible = safe & (earlier | progress)
    feasible[0] = False
    choices = np.flatnonzero(feasible)
    if not len(choices):
        return base, None, "no_faster_reconnection"
    travel = np.array([np.linalg.norm(p[1]-p[0], axis=-1).sum() for p in paths])
    order = np.lexsort((deficit[choices]+.03*travel[choices], when[choices]))
    best = int(choices[order[0]])
    relay = int(relays[best])
    chosen = actions[best]
    event = dict(time_s=float(now), reason="recovery", expert="cut_recovery", changed_relay=ns+relay,
                 base_action=int(base[relay]), action=int(chosen[relay]),
                 lost_search=[int(l) for l in lost], lost_beacon_age_s=finite(age), move_cap_m=cap,
                 base_reconnect_in_s=float(when[0]) if np.isfinite(when[0]) else None,
                 projected_reconnect_in_s=float(when[best]) if np.isfinite(when[best]) else None,
                 base_reconnect_deficit_db_s=float(deficit[0]),
                 projected_reconnect_deficit_db_s=float(deficit[best]),
                 original_margin_db=finite(reconnect[0]), alternative_margin_db=finite(reconnect[best]),
                 candidates=len(paths)-1, estimated_connected=False, cut_evidence=evidence,
                 plain_reconnect_margin_db=plain_margin, persistent_loss=bool(persistent),
                 projected_fleet_endpoint=paths[best][1].tolist(),
                 projected_own_endpoint=paths[best][1, ns+relay].tolist())
    return chosen, event, "selected"
