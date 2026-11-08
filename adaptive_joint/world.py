# Private generalization of v5 ClosedLoopWorld. Source remains frozen.
"""현재 이동·채널·비콘·실제 예측을 연결한 3초 제어 환경."""

from pathlib import Path
import json, time
import numpy as np
from swarm_sim.config import Design, Scenario
from swarm_sim.regions import ROOT, read_json
from usable.collect import prepare_city_without_unused_radio_cache
from cooperative.motion import motion_stream
from cooperative.packets import PacketSim, shadow_sequence, hop_probabilities
from adaptive_joint.telemetry import Telemetry
from adaptive_joint.contracts import DEPLOYMENT_BY_KEY, entry_for
from adaptive_joint.baseline import ensure_nominal, paired_shadow
from adaptive_joint import ARTIFACTS
from cooperative.continuity_world import reward_terms
from cooperative.prediction import ControlPredictor
from cooperative.regions import entries


class World:
    def __init__(
        self,
        region="teheran",
        seed=101,
        kind="intermittent",
        direction="east_west",
        forecast=True,
        capture=False,
        max_steps=None,
        deployment='v8_s3_r5_40',
        observer=None,
        fleet_cost=.002,
    ):
        self.deployment = DEPLOYMENT_BY_KEY[deployment]
        self.reward_age = 0.
        self.fleet_cost = float(fleet_cost)
        self.region = region
        self.seed = int(seed)
        self.kind = kind
        self.direction = direction
        self.forecast = bool(forecast)
        self.capture = bool(capture)
        self.max_steps = max_steps
        self.scenario = Scenario(
            duration_s=7200, motion_dt_s=3, direction=direction
        ).validate()
        self.design = self.deployment.design()
        self.n = self.design.n_drones
        self.ns = self.design.n_search
        n = self.n
        entry = entry_for(region)
        self.map_path = ROOT / entry["path"]
        from swarm_sim.regions import sha256_file

        if sha256_file(self.map_path) != entry["sha256"]:
            raise ValueError("지도 지문 변경: 기준 경로 재생성 후 새 키 등록 필요")
        self.city = prepare_city_without_unused_radio_cache(
            self.map_path, self.scenario, ARTIFACTS / "cache/city"
        )
        self.city.radio.enable_cache()
        file = ensure_nominal(self.city, self.deployment, self.scenario)
        with np.load(file) as z:
            self.baseline_nodes = z['nodes']
        self.nominal_completed = json.loads(file.with_suffix('.json').read_text(encoding='utf-8'))['completed']
        self.baseline_duration = (len(self.baseline_nodes)-1)*3.
        self.baseline_end_step = len(self.baseline_nodes)-1
        anchor = ensure_nominal(self.city, DEPLOYMENT_BY_KEY['v8_s3_r5_40'], self.scenario)
        with np.load(anchor) as z:
            anchor_duration = (len(z['nodes'])-1)*3.
        rng = np.random.default_rng(seed)
        self.jammer = np.r_[
            self.city.target_points[int(rng.integers(len(self.city.target_points)))],
            2.0,
        ]
        if kind == "continuous":
            self.jam_start = 0.0
            self.jam_end = float("inf")
            self.jam_power = float(rng.uniform(-3, 3))
        elif kind == "intermittent":
            self.jam_start = float(rng.uniform(0.15, 0.45) * anchor_duration)
            self.jam_end = float(rng.uniform(0.65, 0.9) * anchor_duration)
            self.jam_power = float(rng.uniform(-3, 3))
        elif kind == "clear":
            self.jam_start = self.jam_end = 0.0
            self.jam_power = 0.0
        else:
            raise ValueError("clear/continuous/intermittent 조건 필요")
        self.shadow = paired_shadow(
            seed,
            14401,
            n,
            0.5,
            self.scenario.shadow_sigma_db,
            self.scenario.shadow_correlation_s,
        )
        self.packets = PacketSim(
            n,
            self.ns,
            self.city.visible,
            0.5,
            seed + 1000003,
            max_duration_s=7203,
            reserved_slots=n,
        )
        self.telemetry = Telemetry(self.city, self.scenario, seed, n, self.ns, max_frames=14400 if max_steps is None else max_steps*6)
        self.predictor = observer
        if observer is not None:
            observer.bind(self.city, self.deployment)
        self.stream = motion_stream(self.city, self.design, self.scenario)
        self.proposal = next(self.stream)
        self.nodes = self.proposal["nodes"].copy()
        self.k = 0
        self.t = 0
        self.done = False
        self.mission = None
        self.p_now = np.zeros((n, n))
        self.stats = []
        self.frames = []
        self.trace = []
        self.history = []
        self.pending = {}
        self.settled = {}
        self.forecast_calls = 0
        self.inference_ms = []
        self.packet_frames = []
        self.channel_available = []
        self.node_log = [self.nodes.copy()]
        self.protection_steps = 0
        self.relocation_steps = 0
        self.requested_steps = 0
        self.min_separation = 1e9
        self.last_forecast = {"status": "warmup", "links": []}
        self.risk = np.full((n, n), np.nan)
        self.early = np.full((n, n), np.nan)
        self.prediction_status = np.full(
            (n, n), "warmup" if self.forecast else "prediction_disabled", dtype="<U48"
        )
        self.last_x = np.zeros((n, n, 48), np.float32)
        self.last_mask = np.zeros_like(self.last_x, bool)
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.last_x[i, j, [2, 3, 4, 12, 13]] = [
                        50.0,
                        i >= self.ns,
                        j >= self.ns,
                        *self.city.points[self.nodes[i]],
                    ]
                    self.last_mask[i, j, [2, 3, 4, 12, 13]] = True
        self.last_intent = np.zeros((n, n, 9), np.float32)
        self.previous_actions = np.zeros(self.deployment.nr, dtype=int)
        if self.capture:
            self.frames.append(
                {
                    "time_s": 0.0,
                    "positions": self.city.points[self.nodes].tolist(),
                    "risk": np.full((n, n), -1).tolist(),
                    "status": self.prediction_status.tolist(),
                    "early": np.full((n, n), -1).tolist(),
                    "links": np.zeros((n, n), int).tolist(),
                    "connected": None,
                    "jammer": bool(self.jam_start == 0 and self.jam_end > 0),
                    "correction": np.zeros((5, 2)).tolist(),
                    "beacon_supported": 0,
                    "protected": False,
                    "coverage": self.proposal["observed_cells"]
                    / self.proposal["observable_cells"],
                    "pdr": None,
                    "aoi": None,
                }
            )

    def _settle(self, final=False):
        g = int(self.packets.ic[0])
        born = self.packets.born[:g]
        status = self.packets.status[:g]
        delay = self.packets.delays[:g]
        for step in list(sorted(self.pending)):
            t0, t1 = step * 3.0, (step + 1) * 3.0
            if not final and self.packets.fc[0] < t1 - 1e-9:
                break
            lo = np.searchsorted(born, t0 - 1e-9)
            hi = np.searchsorted(born, t1 - 1e-9)
            st = status[lo:hi]
            if not final and np.any(st == 0):
                break
            ok = st == 1
            n = hi - lo
            P = float(ok.sum() / max(n, 1))
            L = float(
                np.minimum(1.0, 0.25 / np.maximum(delay[lo:hi][ok], 0.005)).sum()
                / max(n, 1)
            )
            Q = 0.5 * (P + L)
            row = self.pending.pop(step)
            row.update(Q=Q, P=P, L=L)
            # 비용 형태: 임무를 불필요하게 늘려 양의 보상을 누적하는 현상 방지.
            row["reward"] = (
                2 * (row["C"] - 1)
                + 0.5 * (Q - 1)
                - 0.002
                - 0.005 * row["extra_move"]
                - 0.01 * row["search_lag"]
                - 0.01 * row["protection"]
            )
            terms, self.reward_age = reward_terms(np.asarray(self.stats[step*6:(step+1)*6])[:,0],
                    self.reward_age, Q, row['extra_move'], row['search_lag'], row['protection'])
            terms = np.r_[terms, -self.fleet_cost*self.n/10.]
            row['reward_terms'] = terms
            row['reward'] = float(terms.sum())
            self.settled[step] = row

    def step(self, correction):
        if self.done:
            raise RuntimeError("종료한 환경")
        start = time.perf_counter()
        n = self.n
        old = self.nodes.copy()
        own = self.city.points[old]
        c = np.asarray(correction, float).copy()
        if c.shape != (n, 2) or np.any(c[:self.ns] != 0) or not np.isfinite(c).all():
            raise ValueError("Only relay rows may contain finite [N,2] corrections")
        if np.any(np.abs(c) > 40.00001):
            raise ValueError("축별 보정 범위 40m 초과")
        self.requested_steps += int(np.any(c))
        lag = (
            np.linalg.norm(own[:self.ns] - self.proposal["nominal"][:self.ns], axis=1).max() / 40.0
        )
        protect = bool(
            np.any(c)
            and (
                self.k * 3 > self.baseline_duration * 1.25
                or lag > 1.5
                or self.proposal["phase"] == 2
            )
        )
        if protect:
            c[:] = 0
        self.protection_steps += int(protect)
        self.relocation_steps += int(np.any(c))
        event = self.stream.send(c)
        assert event["kind"] == "motion"
        new = event["new"]
        self.nodes = new.copy()
        self.node_log.append(new.copy())
        self.telemetry.issue(self.k * 3.0, event["desired"])
        a, b = self.city.points[old], self.city.points[new]
        alpha = np.arange(6) / 6
        xy = a[None] * (1 - alpha[:, None, None]) + b[None] * alpha[:, None, None]
        xyz = np.concatenate((xy, np.full((6, n, 1), 50.0)), axis=-1)
        tx = np.repeat(xyz, n, axis=1).reshape(-1, 3)
        rx = np.tile(xyz, (1, n, 1)).reshape(-1, 3)
        loss = self.city.radio.propagation(tx, rx, self.scenario)[0].reshape(6, n, n)
        jloss = self.city.radio.propagation(
            self.jammer[None], xyz.reshape(-1, 3), self.scenario
        )[0].reshape(6, n)
        C = []
        pair = []
        for q in range(6):
            t = self.t
            now = t * 0.5
            active = self.jam_start <= now < self.jam_end
            interference = 10 ** ((self.jam_power - jloss[q]) / 10) * active
            received = np.empty((n, n))
            sinr = np.empty((n, n))
            p = hop_probabilities(loss[q], interference, self.shadow[t], received, sinr)
            st = self.packets.interval(p, old)
            self.p_now = p
            self.stats.append(st.copy())
            C.append(st[0])
            pair.append(st[1])
            f = (q + 1) / 6
            end = np.c_[a * (1 - f) + b * f, np.full(n, 50.0)]
            x, m, intent = self.telemetry.push(
                xyz[q],
                end,
                received,
                sinr,
                self.packets.attempts,
                self.packets.successes,
                self.packets.last_ack,
                self.packets.queue_now,
            )
            self.last_x = x
            self.last_mask = m
            self.last_intent = intent
            self.packet_frames.append(self.packets.frame.copy())
            up = (p >= 0.9) & (p.T >= 0.9)
            self.channel_available.append(up)
            if self.predictor:
                tick = time.perf_counter()
                out = self.predictor.observe((t + 1) * 0.5, x, m, intent)
                if out["status"] == "ready":
                    self.inference_ms.append((time.perf_counter() - tick) * 1000)
                    self.forecast_calls += 1
                    self.last_forecast = out
                    self.risk[:] = np.nan
                    self.early[:] = np.nan
                    for row in out["links"]:
                        i, j = row["source"], row["peer"]
                        self.prediction_status[i, j] = row["status"]
                        if row["risk_5s"] is not None:
                            self.risk[i, j] = row["risk_5s"]
                            self.early[i, j] = row["risk_onset_3to7s"]
            self.t += 1
        self.k += 1
        self.min_separation = min(self.min_separation, event["minimum_separation_m"])
        K = min(self.k, len(self.baseline_nodes) - 1)
        K0 = min(self.k - 1, len(self.baseline_nodes) - 1)
        base_change = (
            self.city.points[self.baseline_nodes[K]]
            - self.city.points[self.baseline_nodes[K0]]
        )
        delta = np.linalg.norm(b - a, axis=1)
        extra = (
            np.maximum(0, delta[self.ns:] - np.linalg.norm(base_change[self.ns:], axis=1)).mean()
            / 15.0
        )
        self.pending[self.k - 1] = {
            "C": float(np.mean(C)),
            "pair": float(np.mean(pair)),
            "extra_move": float(extra),
            "search_lag": float(min(lag, 4.0)),
            "protection": float(protect),
        }
        self._settle()
        self.history.append(
            {
                "time_s": self.k * 3.0,
                "phase": event["phase"],
                "correction": c.copy(),
                "protected": protect,
                "C": float(np.mean(C)),
                "observed_cells": event["observed_cells"],
                "step_seconds": time.perf_counter() - start,
            }
        )
        if self.capture:
            self.frames.append(
                {
                    "time_s": self.k * 3.0,
                    "positions": b.tolist(),
                    "risk": np.where(np.isfinite(self.risk), self.risk, -1).tolist(),
                    "status": self.prediction_status.tolist(),
                    "early": np.where(np.isfinite(self.early), self.early, -1).tolist(),
                    "connected": bool(C[-1]),
                    "links": self.channel_available[-1].astype(int).tolist(),
                    "pdr": float(self.packets.ic[1] / max(self.packets.ic[0], 1)),
                    "aoi": float(self.packets.frame[8]),
                    "jammer": bool(self.jam_start <= self.k * 3 < self.jam_end),
                    "correction": c[self.ns:].tolist(),
                    "beacon_supported": int(np.sum(self.last_x[..., 46] > 0.5)),
                    "protected": protect,
                    "coverage": event["observed_cells"] / event["observable_cells"],
                }
            )
        truncated = (
            self.max_steps is not None and self.k >= self.max_steps
        ) or self.k >= 2400
        if event["done"] or truncated:
            self.done = True
            if event["done"]:
                try:
                    next(self.stream)
                except StopIteration as e:
                    self.mission = e.value
            self.packets.drain()
            self._settle(True)
            if truncated and not event["done"]:
                self.settled[self.k - 1]["reward"] -= 10.0
                self.settled[self.k - 1]["reward_terms"][7] -= 10.0
            self.summary = self.metrics(event)
        else:
            self.proposal = next(self.stream)
        return self.done

    def metrics(self, event):
        values = self.packets.values()
        g = int(values[0])
        ok = self.packets.status[:g] == 1
        dl = self.packets.delays[:g][ok]
        tr = np.asarray(self.node_log)
        pos = self.city.points[tr]
        length = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
        st = np.asarray(self.stats)
        rewards = [self.settled[k]["reward"] for k in range(self.k)]
        f = self.packets.frame
        # 완주/실패에 관계없이 생성 패킷이 모두 전달 또는 폐기로 정산됐는지 확인.
        if g != int(values[1] + values[5] + values[6] + values[7]):
            raise RuntimeError("생성=전달+폐기의 패킷 보존식 위반")
        if len(st) != self.k * 6 or len(tr) != self.k + 1:
            raise RuntimeError("3초 이동과 0.5초 RF 기록 길이 불일치")
        from swarm_sim.motion import fleet_offsets

        offset = fleet_offsets(self.design, self.direction)
        ii, jj = np.triu_indices(self.ns, 1)
        expected = np.linalg.norm(offset[ii] - offset[jj], axis=-1)
        actual = np.linalg.norm(pos[:, ii] - pos[:, jj], axis=-1)
        formation_rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
        terms = np.stack([self.settled[k]['reward_terms'] for k in range(self.k)])
        if not np.allclose(terms.sum(1), rewards, rtol=0, atol=1e-10):
            raise ValueError('Reward component conservation failed')
        return {
            'deployment': self.deployment.key,
            'total_drones': self.n, 'search_drones': self.ns,
            'reserved_beacon_slots': self.n,
            'nominal_completed': self.nominal_completed,
            'reward_terms': terms.sum(0).tolist(),
            "region": self.region,
            "seed": self.seed,
            "kind": self.kind,
            "direction": self.direction,
            "mission_completed": bool(
                self.mission and self.mission["mission_completed"]
            ),
            "duration_s": self.k * 3.0,
            "baseline_duration_s": self.baseline_duration,
            "duration_ratio": self.k * 3 / self.baseline_duration if self.nominal_completed else None,
            "search_connected_mean": float(st[:, 0].mean()),
            "pair_reach_mean": float(st[:, 1].mean()),
            "outage_total_s": float(values[11]),
            "outage_max_s": float(values[12]),
            "outage_events": int(values[13]),
            "pdr": float(values[1] / max(g, 1)),
            "generated_packets": g,
            "delivered_packets": int(values[1]),
            "delay_mean_s": float(dl.mean()) if len(dl) else None,
            "delay_p95_s": float(np.percentile(dl, 95)) if len(dl) else None,
            "timely_pdr": float(values[8] / max(g, 1)),
            "retries": int(values[4]),
            "ttl_drops": int(values[5]),
            "retry_drops": int(values[6]),
            "overflow_drops": int(values[7]),
            "aoi_mean_s": float(values[10]),
            "relay_travel_m": float(length[:, self.ns:].sum()),
            "search_travel_m": float(length[:, :self.ns].sum()),
            "search_formation_rmse_m": formation_rmse,
            "actual_delivered_hops_mean": (
                float(self.packets.hops[:g][ok].mean()) if ok.any() else None
            ),
            "delivered_via_relay_fraction": (
                float((self.packets.relay_hops[:g][ok] > 0).mean())
                if ok.any()
                else None
            ),
            "minimum_separation_m": self.min_separation,
            "actual_safety_violations": 0,
            "observable_coverage": event["observed_cells"] / event["observable_cells"],
            "information_shared_fraction": float(
                np.all(self.packets.known, axis=0).mean()
            ),
            "relocation_fraction": self.relocation_steps / max(self.k, 1),
            "protection_fraction": self.protection_steps / max(self.k, 1),
            "reward_mean": float(np.mean(rewards)),
            "reward_sum": float(np.sum(rewards)),
            "beacon_tx": self.telemetry.tx_count,
            "beacon_rx": self.telemetry.rx_count,
            "propagation_cache_hits": self.city.radio.cache_hits,
            "propagation_cache_misses": self.city.radio.cache_misses,
            "forecast_calls": self.forecast_calls,
            "forecast_p95_ms": (
                float(np.percentile(self.inference_ms, 95))
                if self.inference_ms
                else None
            ),
        }

