"""개선안 4 (선택). 급락형 단절 대비 우회 경로 확보.

V21은 우회 여유(fragile_margin)를 후보끼리 비교할 때 가점으로만 쓴다. 이 모듈은 예측 점수와
무관하게, 지도에서 얻는 차폐 정도를 근거로 발동한다.

발동 조건 (모두 만족):
* 지금 연결돼 있고 7초 안에 단절이 예측되지 않는다 (예측 단절은 V18/V21의 몫).
* 3~7초 구간에서 링크 하나만 끊겨도 수색망이 갈라진다 (단일 링크 제거 여유 < 목표).
* 그렇게 망을 가르는 링크의 건물 감쇠가 shield_db 이상이다. 건물에 가린 링크는 마지막 0.5초에
  6dB 이상 급락하는 급락형 단절의 주된 원인이다.

후보는 V21과 같은 물리 검사(초기 3초 안전, 단절 수 증가 금지)를 통과하고, 링크 하나를 지워도
남는 여유를 bypass_min_gain_db 이상 올려야 한다.
"""
import numpy as np

from adaptive_joint_expanded.features import OFFSETS
from forecast_avoidance.controller import onset_mask, proposal_set, samples
from forecast_learning.paths import continuations
from forecast_shield.forecaster import geometry

from .graph import single_link_removal

WINDOW = slice(6, 15)   # 3~7초. V21 window_deficit과 같은 구간.
CLIP_DB = 30.


def building_extra(planner, actions):
    """계획 경로(held continuation)의 링크별 건물 감쇠 dB. [T, n, n]"""
    held, _ = continuations(planner, actions)
    _, extra = geometry(planner.city, planner.scenario, samples(held))
    return extra


def fragile(graphs, n_search):
    """두 continuation 중 나쁜 쪽 기준, 링크 하나를 지운 뒤 남는 수색망 여유의 최솟값."""
    return min(float(np.clip(single_link_removal(g[WINDOW], n_search)[0], -CLIP_DB, CLIP_DB).min())
               for g in graphs)


def shield_exposure(graph, extra, n_search):
    """끊기면 수색망이 갈라지는 링크들의 건물 감쇠 최댓값 (dB)."""
    removal, i, j = single_link_removal(graph[WINDOW], n_search)
    critical = removal < 0.
    if not critical.any():
        return 0.
    shield = np.asarray(extra)[WINDOW][:, i, j]
    return float(np.nan_to_num(shield[critical], nan=0., posinf=CLIP_DB).max())


def bypass(planner, actions, logits, valid, local, phase, now, base, options, extract, early_drop_db):
    """반환: (후보, 사건) 또는 (None, None)과 이유."""
    _, _, margins, graphs, endpoint = base
    ns = planner.design.n_search
    if phase == 2:
        return None, None, "final_phase"
    if margins[:, :WINDOW.stop].min() < 0.:
        return None, None, "cut_now_or_predicted"
    if not np.asarray(planner.state.known[:ns]).all():
        return None, None, "stale_search_beacon"
    base_fragile = fragile(graphs, ns)
    if base_fragile >= options.bypass_target_db:
        return None, None, "already_redundant"
    exposure = shield_exposure(graphs[0], building_extra(planner, actions), ns)
    if exposure < options.shield_db:
        return None, None, "low_building_shielding"
    counts = onset_mask(margins).sum(axis=-1)
    best = None
    for candidate, relay in proposal_set(actions, logits, valid):
        if relay is None:
            continue
        step = np.linalg.norm(OFFSETS[candidate[relay]]-OFFSETS[actions[relay]])
        if step > options.bypass_move_cap_m+1e-6:
            continue
        _, _, alt, alt_graphs, alt_endpoint = extract(planner, candidate, local, phase)
        if np.array_equal(alt_endpoint, endpoint):
            continue
        if alt[:, :WINDOW.stop].min() < 0.:
            continue
        if (alt[:, 1:7] < np.minimum(0., margins[:, 1:7])-early_drop_db).any():
            continue
        if (onset_mask(alt).sum(axis=-1) > counts).any():
            continue
        value_fragile = fragile(alt_graphs, ns)
        gain = value_fragile-base_fragile
        if gain < options.bypass_min_gain_db:
            continue
        distance = float(np.linalg.norm(alt_endpoint-endpoint, axis=-1).sum())
        value = min(gain, 15.)-options.bypass_distance_weight*distance
        if best is None or value > best[0]:
            best = value, candidate, alt_endpoint, int(relay), gain, value_fragile, distance
    if best is None:
        return None, None, "no_redundant_safe_move"
    _, candidate, alt_endpoint, relay, gain, value_fragile, distance = best
    event = dict(time_s=float(now), reason="bypass", expert="redundancy_preposition", changed_relay=ns+relay,
                 base_action=int(actions[relay]), action=int(candidate[relay]),
                 base_single_link_margin_db=base_fragile, single_link_margin_db=value_fragile,
                 redundancy_gain_db=gain, shield_exposure_db=exposure, command_distance_m=distance,
                 projected_fleet_endpoint=np.asarray(alt_endpoint).tolist())
    return candidate, event, "selected"
