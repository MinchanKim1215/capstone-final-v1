"""지역 지도 교체 및 드론 한 대의 관측만으로 실행하는 예측 인터페이스."""

from collections import deque
from pathlib import Path
import numpy as np
import torch
from swarm_sim.city import City
from swarm_sim.config import Scenario
from swarm_sim.regions import read_json, sha256_file
from usable.spatial import features as spatial_features
from advance.infer import load, predict_batch, probabilities
from cooperative.fast_features import feature_batch


class LocalGoalCache:
    """자기 명령은 제어기에서, 이웃 명령은 실제 수신 비콘에서만 갱신."""

    def __init__(self, n_drones, observer):
        if (
            isinstance(n_drones, bool)
            or isinstance(observer, bool)
            or not isinstance(n_drones, (int, np.integer))
            or not isinstance(observer, (int, np.integer))
            or n_drones < 2
            or not 0 <= observer < n_drones
        ):
            raise ValueError("유효한 드론 수와 정수 관측 드론 번호 필요")
        self.n = int(n_drones)
        self.observer = int(observer)
        self.target = np.zeros((self.n, 2), np.float64)
        self.issued = np.full(self.n, -1.0)
        self.received = np.full(self.n, -1.0)
        self.ready = False

    def update_own(self, issue_time_s, target_xy):
        goal = np.asarray(target_xy, np.float64)
        if (
            goal.shape != (2,)
            or not np.isfinite(goal).all()
            or not np.isfinite(issue_time_s)
            or issue_time_s < 0
            or issue_time_s < self.issued[self.observer]
        ):
            raise ValueError("현재 제어기의 유한한 목표와 발행 시각 필요")
        self.target[self.observer] = goal
        self.issued[self.observer] = issue_time_s
        self.ready = True

    def received_beacon(self, peer, issue_time_s, target_xy, received_time_s):
        goal = np.asarray(target_xy, np.float64)
        if (
            isinstance(peer, bool)
            or not isinstance(peer, (int, np.integer))
            or not 0 <= peer < self.n
            or peer == self.observer
            or goal.shape != (2,)
            or not np.isfinite(goal).all()
            or not np.isfinite(issue_time_s)
            or not np.isfinite(received_time_s)
            or not 0 <= issue_time_s <= received_time_s
            or received_time_s < self.received[peer]
        ):
            raise ValueError("실제 수신한 이웃의 현재/과거 명령 필요")
        if issue_time_s < self.issued[peer]:
            return False
        self.target[peer] = goal
        self.issued[peer] = issue_time_s
        self.received[peer] = received_time_s
        return True

    def encode(self, time_end_s, values, mask):
        x = np.asarray(values, np.float32)
        m = np.asarray(mask, bool)
        if (
            not self.ready
            or x.shape != (self.n, 48)
            or m.shape != x.shape
            or not np.isfinite(time_end_s)
            or not np.isfinite(x).all()
            or time_end_s < self.issued[self.observer]
        ):
            raise ValueError("자기 명령과 유효한 로컬 [N,48] 관측 필요")
        out = np.zeros((self.n, 9), np.float32)
        for j in range(self.n):
            if j == self.observer:
                continue
            rel = self.target[self.observer] - x[j, 12:14]
            remaining = 3 - (time_end_s - self.issued[self.observer])
            if remaining < -1e-6:
                raise ValueError("3초 자기 명령 갱신 주기 초과")
            out[j, :2] = rel
            out[j, 5] = max(0.0, remaining)
            out[j, 6] = np.linalg.norm(rel)
            if (
                self.issued[j] < 0
                or self.received[j] > time_end_s
                or not m[j, [30, 34, 35]].all()
                or x[j, 30] > 2
            ):
                continue
            rel = self.target[j] - (x[j, 12:14] + x[j, 34:36])
            out[j, 2:4] = rel
            out[j, 4] = time_end_s - self.issued[j]
            out[j, 7] = np.linalg.norm(rel)
            out[j, 8] = 1
        return out


class ControlPredictor:
    """0.5초 관측 축적, 20초 이후 3초 제어 시점에 위험 출력."""

    def __init__(self, checkpoint, map_path, threads=1, observers=None):
        torch.set_num_threads(threads)
        self.path = Path(checkpoint)
        self.model, self.state = load(self.path)
        self.cal = read_json(self.path.with_name("calibration.json"))
        sha = sha256_file(self.path)
        if self.cal["checkpoint_sha256"] != sha:
            raise ValueError("가중치와 확률 보정 불일치")
        p = self.path.with_name("alarm_policy.json")
        self.policy = read_json(p) if p.exists() else {"5s": None, "8s": None}
        if self.policy.get("checkpoint_sha256", sha) != sha:
            raise ValueError("가중치와 경고 정책 불일치")
        self.city = City(map_path, Scenario())
        self.observers = None if observers is None else tuple(int(i) for i in observers)
        self.reset()

    def reset(self):
        self.values = deque(maxlen=40)
        self.masks = deque(maxlen=40)
        self.last_time = None
        self.shape = None
        self.observer = None
        self.active = None
        self.early_active = None

    def observe_local(self, time_end_s, observer, values, mask, intents):
        x = np.asarray(values, np.float32)
        m = np.asarray(mask, bool)
        c = np.asarray(intents, np.float32)
        if (
            x.ndim != 2
            or x.shape != m.shape
            or isinstance(observer, bool)
            or not isinstance(observer, (int, np.integer))
            or not 0 <= observer < len(x)
            or c.shape != (len(x), 9)
        ):
            raise ValueError("관측 드론 번호와 로컬 [N,48/60], 명령 [N,9] 필요")
        if self.observer is not None and self.observer != int(observer):
            raise ValueError("관측 드론 변경 전 reset 필요")
        self.observer = int(observer)
        full = np.zeros((len(x), len(x), x.shape[-1]), np.float32)
        known = np.zeros_like(full, bool)
        full[observer] = x
        known[observer] = m
        goal = np.zeros((len(x), len(x), 9), np.float32)
        goal[observer] = c
        out = self.observe(time_end_s, full, known, goal)
        out["links"] = [r for r in out["links"] if r["source"] == observer]
        return out

    @torch.inference_mode()
    def observe(self, time_end_s, values, mask, intents):
        if (
            isinstance(time_end_s, bool)
            or not np.isfinite(time_end_s)
            or time_end_s <= 0
            or not np.isclose(
                time_end_s / 0.5, round(time_end_s / 0.5), rtol=0, atol=1e-6
            )
        ):
            raise ValueError("양수 시각과 0.5초 정렬 필요")
        x = np.asarray(values, np.float32)
        m = np.asarray(mask, bool)
        c = np.asarray(intents, np.float32)
        if (
            x.ndim != 3
            or x.shape != m.shape
            or x.shape[0] != x.shape[1]
            or x.shape[-1] not in (48, 60)
            or c.shape != (*x.shape[:2], 9)
        ):
            raise ValueError("[N,N,48/60] 관측과 [N,N,9] 명령 필요")
        if not np.isfinite(x).all() or not np.isfinite(c).all() or np.any(x[~m] != 0):
            raise ValueError("유한한 관측과 결측 값 0·마스크 False 필요")
        if (
            np.any((c[..., 5] < 0) | (c[..., 5] > 3))
            or np.any(c[..., 4] < 0)
            or np.any((c[..., 8] != 0) & (c[..., 8] != 1))
        ):
            raise ValueError("유효한 명령 나이·주기·지원 플래그 필요")
        if self.last_time is not None and not np.isclose(
            time_end_s - self.last_time, 0.5, rtol=0, atol=1e-6
        ):
            raise ValueError("누락된 0.5초 프레임은 명시적 결측 관측으로 전달")
        if x.shape[-1] == 48:
            gm = m.copy()
            if self.observers is not None:
                selected = np.zeros(len(x), bool)
                selected[list(self.observers)] = True
                gm[~selected] = False
            gm[..., 34] &= m[..., [0, 1, 2, 12, 13, 30, 34, 35, 36, 37]].all(-1)
            z, zm = spatial_features(self.city, x, gm)
            x = np.concatenate((x, z), -1)
            m = np.concatenate((m, zm), -1)
        if self.shape is not None and self.shape != x.shape:
            raise ValueError("드론 수 변경 전 reset 필요")
        self.shape = x.shape
        self.last_time = float(time_end_s)
        self.values.append(x.copy())
        self.masks.append(m.copy())
        n = len(x)
        if self.active is None:
            self.active = np.zeros((n, n), bool)
            self.early_active = self.active.copy()
        if len(self.values) < 40:
            return {"time_end_s": float(time_end_s), "status": "warmup", "links": []}
        if not np.isclose(time_end_s % 3, 0, rtol=0, atol=1e-6):
            return {
                "time_end_s": float(time_end_s),
                "status": "between_decisions",
                "links": [],
            }
        history = np.asarray(self.values)
        known = np.asarray(self.masks)
        enough = (np.sum(history[..., 33] > 0, axis=0) >= 3) & ~np.eye(n, dtype=bool)
        if self.observers is not None:
            selected = np.zeros(n, bool)
            selected[list(self.observers)] = True
            enough[~selected] = False
        src, dst = np.where(enough)
        score = {}
        if len(src):
            intent = np.zeros((40, n, n, 9), np.float32)
            intent[-1] = c
            index = np.c_[np.full(len(src), 39), src, dst]
            z, mm, e, s = feature_batch(
                self.state, self.city, history, known, index, intent
            )
            p = probabilities(
                predict_batch(self.model, self.state, z, mm, e, s), self.cal
            )
            for q, (i, j) in enumerate(zip(src, dst)):
                score[int(i), int(j)] = {h: float(v[q]) for h, v in p.items()}
        rows = []
        for i, j in zip(*np.where(~np.eye(n, dtype=bool))):
            p = score.get((int(i), int(j)))
            old = bool(self.active[i, j])
            early_old = bool(self.early_active[i, j])
            margin = float(x[i, j, 41]) if m[i, j, 41] else None
            fresh = bool(m[i, j, 46] and x[i, j, 46] > 0.5)
            if p is None or margin is None:
                status = "insufficient_observations"
            elif not fresh:
                status = "stale_observations"
            elif margin <= 0:
                status = "currently_degraded"
            elif not m[i, j, 48:60].all():
                status = "insufficient_observations"
            elif c[i, j, 8] < 0.5:
                status = "insufficient_intent_observations"
            else:
                status = "supported"

            def alarm(rule, previous):
                return bool(
                    status == "supported"
                    and rule
                    and rule.get("enabled", True)
                    and (
                        (previous and p[rule["head"]] >= rule["exit_threshold"])
                        or p[rule["head"]] >= rule["enter_threshold"]
                    )
                )

            warning = alarm(self.policy.get("5s"), old)
            early = alarm(self.policy.get("8s"), early_old)
            self.active[i, j] = warning
            self.early_active[i, j] = early
            supported = status == "supported"
            if supported:
                status = (
                    "warning"
                    if warning
                    else (
                        "watch"
                        if p[0] >= self.cal["heads"]["0"]["threshold_balanced"]
                        else "normal"
                    )
                )
            rows.append(
                {
                    "source": int(i),
                    "peer": int(j),
                    "risk_5s": p[0] if supported else None,
                    "risk_onset_3to4s": p.get(3) if supported else None,
                    "risk_onset_3to7s": p.get(4) if supported else None,
                    "status": status,
                    "new_warning_episode": bool(warning and not old),
                    "early_warning_8s": early,
                    "new_early_warning_episode": bool(early and not early_old),
                    "observed_margin_db": margin,
                }
            )
        return {"time_end_s": float(time_end_s), "status": "ready", "links": rows}
