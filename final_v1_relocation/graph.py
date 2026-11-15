"""개선안이 함께 쓰는 연결 그래프 계산."""
import numpy as np

from adaptive_joint_expanded.features import widest
from forecast_avoidance.controller import TIMES


def closure(margin):
    """모든 쌍의 가장 넓은 경로 여유 (max-min 폐포). margin [..., n, n]."""
    value = np.array(margin, dtype=float, copy=True)
    n = value.shape[-1]
    idx = np.arange(n)
    # 자기 자신 경유는 여유를 늘릴 수 없으므로 +inf로 두어 NaN 대각 성분의 전파를 막는다.
    value[..., idx, idx] = np.inf
    for k in range(n):
        value = np.maximum(value, np.minimum(value[..., :, k, None], value[..., None, k, :]))
    return value


def lost_margin(close, lost, n_search):
    """끊긴 수색기 각각에서 나머지 수색기까지 가장 넓은 경로 여유의 최솟값. [..., T]"""
    searchers = np.arange(n_search)
    values = [close[..., int(l), searchers[searchers != l]].min(axis=-1) for l in lost]
    return np.min(values, axis=0)


def pair_margin(close, members):
    """members 안 모든 쌍의 가장 넓은 경로 여유 최솟값. [..., T]"""
    members = np.asarray(members, dtype=int)
    i, j = np.triu_indices(len(members), 1)
    return close[..., members[i], members[j]].min(axis=-1)


def first_sustained(margin):
    """여유가 반초 두 번 연속 0 이상이 되는 첫 시각(초). 없으면 inf. margin [P, T]."""
    up = np.asarray(margin) >= 0.
    both = up[..., 1:-1] & up[..., 2:]
    result = np.full(both.shape[:-1], np.inf)
    hit = both.any(axis=-1)
    result[hit] = TIMES[1:-1][np.argmax(both[hit], axis=-1)]
    return result


def single_link_removal(graph, n_search):
    """링크 하나씩 지웠을 때의 수색망 여유. graph [T, n, n] → ([T, E], i, j)."""
    n = graph.shape[-1]
    i, j = np.triu_indices(n, 1)
    alternatives = np.repeat(np.asarray(graph, dtype=float)[:, None], len(i), axis=1)
    edges = np.arange(len(i))
    alternatives[:, edges, i, j] = -np.inf
    alternatives[:, edges, j, i] = -np.inf
    return widest(alternatives, n_search).min(axis=-1), i, j


def finite(values, bound=99.):
    """JSON(allow_nan=False)에 쓸 수 있도록 유한 값 목록으로 바꾼다."""
    return np.round(np.nan_to_num(np.asarray(values, dtype=float), nan=-bound, posinf=bound, neginf=-bound), 3).tolist()
