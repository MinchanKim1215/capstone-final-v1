"""창 단위 분류와 사건 단위 첫 경고의 분리. 기존 5초 정의 보존."""

import numpy as np
from usable.alarm_policy import simulate_policy, summarize


def arrays(up, index, score, horizon_s=5, minimum_lead_s=3):
    nt, n, _ = up.shape
    ends = np.arange(39, nt - 20, 2)
    rows = (index[:, 0] - 39) // 2
    if np.any(ends[rows] != index[:, 0]):
        raise ValueError("결정 시각 불일치")
    p = np.full((len(ends), n * n), -1.0, np.float32)
    p[rows, index[:, 1] * n + index[:, 2]] = score
    y = np.zeros_like(p, bool)
    new = y.copy()
    event3 = np.full(p.shape, -1, np.int32)
    events = 0
    horizon = int(horizon_s * 2)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bad = ~up[:, i, j]
            pair = bad[:-1] & bad[1:]
            pref = np.r_[0, np.cumsum(pair)]
            y[:, i * n + j] = (pref[ends + horizon] - pref[ends + 1]) > 0
            bounds = np.diff(np.r_[False, bad, False].astype(np.int8))
            for start, stop in zip(
                np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1)
            ):
                if stop - start < 2 or start * 0.5 < 20:
                    continue
                e = events
                events += 1
                c = np.flatnonzero((ends < start) & (start + 1 <= ends + horizon))
                new[c, i * n + j] = True
                lead = (start - ends[c] - 1) * 0.5
                event3[c[lead >= minimum_lead_s], i * n + j] = e
    return p, y, new, event3, events


def count(up, index, p, rule, horizon=5):
    if rule is None:
        rule = {"head": 0, "enter_threshold": 1.1, "exit_threshold": 1.1}
    return simulate_policy(
        *arrays(up, index, p[rule["head"]], horizon),
        rule["enter_threshold"],
        rule["exit_threshold"]
    )[0]
