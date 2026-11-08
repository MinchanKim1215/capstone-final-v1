# Private generalized copy of cooperative/telemetry.py. Original remains unchanged.
"""성공한 데이터/비콘 수신과 자기 상태로만 원시 48항목 생성."""

import math
import numpy as np
import shapely
from usable.beacons import data_threshold_db
from advance.predictor import LocalGoalCache
from cooperative.beacon_frame import beacon_frame
from cooperative.packets import NumbaRNG


class Telemetry:
    def __init__(self, city, scenario, seed, n=8, ns=3, max_frames=14400):
        self.city, self.scenario = city, scenario
        self.n = n
        self.ns = ns
        self.t = 0
        self.max_frames = max_frames
        nt, n = max_frames, self.n
        self.x = np.zeros((nt, n, n, 48), np.float32)
        self.mask = np.zeros_like(self.x, bool)
        self.received = np.zeros((nt, n, n))
        self.sinr = np.zeros_like(self.received)
        self.starts = np.zeros((nt, n, 3))
        self.ends = np.zeros_like(self.starts)
        self.beacon = np.zeros((nt, n, n, 20), np.float32)
        self.beacon_mask = np.zeros_like(self.beacon, bool)
        self.counts = np.zeros((nt, n, n), np.int16)
        self.last = np.full((n, n), -1.0)
        self.rxdb = np.zeros((n, n))
        self.rxsin = np.zeros((n, n))
        self.revstamp = np.full((n, n), -1.0)
        self.revsin = np.zeros((n, n))
        self.poses = np.zeros((n, n, 3))
        self.vels = np.zeros_like(self.poses)
        self.pose_stamp = np.full((n, n), -1.0)
        self.ack_time = np.full((n, n), -1.0)
        self.ack_power = np.zeros((n, n))
        self.ack_sinr = np.zeros((n, n))
        self.data_last = np.full((n, n), -1, np.int32)
        self.rng = NumbaRNG(seed + 2000003)
        self.goals = [LocalGoalCache(n, i) for i in range(n)]
        self.tx_count = self.rx_count = 0
        self.threshold = data_threshold_db()
        high = [g for g, h in zip(city.geometries, city.heights) if h >= 40]
        self.high = np.asarray(high, object) if high else None
        self.high_tree = shapely.STRtree(self.high) if high else None

    def issue(self, time_s, desired):
        self.issued_time = float(time_s)
        self.desired = np.asarray(desired, float).copy()
        for i, c in enumerate(self.goals):
            c.update_own(time_s, desired[i])

    def push(self, start, end, received, sinr, attempts, successes, last_ack, queues):
        t, n = self.t, self.n
        if t >= self.max_frames:
            raise RuntimeError("관측 버퍼 시간 상한 초과")
        self.starts[t] = start
        self.ends[t] = end
        self.received[t] = received
        self.sinr[t] = sinr
        velocity = (end - start) / 0.5
        now = (t + 1) * 0.5
        x = self.x[t]
        m = self.mask[t]
        geom_i = []
        geom_j = []
        geom_peer = []
        geom_v = []
        if self.high_tree is None:
            near = np.full(n, math.sqrt(2) * 400)
        else:
            points = shapely.points(end[:, :2])
            nearest = self.high_tree.nearest(points)
            near = shapely.distance(points, self.high[nearest])
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x[i, j, :7] = [
                    velocity[i, 0],
                    velocity[i, 1],
                    end[i, 2],
                    i >= self.ns,
                    j >= self.ns,
                    attempts[i, j],
                    successes[i, j],
                ]
                m[i, j, :7] = True
                if attempts[i, j] > 0:
                    x[i, j, 7] = successes[i, j] / attempts[i, j]
                    m[i, j, 7] = True
                if successes[i, j] > 0:
                    self.ack_time[i, j] = last_ack[i, j]
                    self.ack_power[i, j] = received[j, i]
                    self.ack_sinr[i, j] = sinr[j, i]
                if self.ack_time[i, j] >= 0:
                    x[i, j, 8:11] = [
                        self.ack_power[i, j],
                        self.ack_sinr[i, j],
                        max(0.0, now - self.ack_time[i, j]),
                    ]
                    m[i, j, 8:11] = True
                x[i, j, 11] = queues[i]
                m[i, j, 11] = True
                x[i, j, 12:15] = [end[i, 0], end[i, 1], successes[j, i]]
                m[i, j, 12:15] = True
                if successes[j, i] > 0:
                    self.data_last[i, j] = t
                last = self.data_last[i, j]
                if last >= 0:
                    age = (t + 1 - last) * 0.5
                    x[i, j, 15] = age
                    m[i, j, 15] = True
                    if age <= 10:
                        previous_velocity = (
                            self.ends[last, j] - self.starts[last, j]
                        ) / 0.5
                        peer = self.starts[last, j] + previous_velocity * age
                        peer[:2] = np.clip(peer[:2], -195, 195)
                        rel = peer - end[i]
                        dv = previous_velocity - velocity[i]
                        dist = np.linalg.norm(rel[:2])
                        x[i, j, 16:22] = [
                            *rel[:2],
                            *dv[:2],
                            dist,
                            np.dot(rel[:2], dv[:2]) / max(dist, 1.0),
                        ]
                        geom_i.append(i)
                        geom_j.append(j)
                        geom_peer.append(peer)
                        geom_v.append(previous_velocity)
                        m[i, j, 16:27] = True
                x[i, j, 27] = near[i]
                m[i, j, 27] = True
        if geom_i:
            ii = np.asarray(geom_i)
            jj = np.asarray(geom_j)
            a = end[ii]
            b = np.asarray(geom_peer)
            vb = np.asarray(geom_v)
            loss, los, _ = self.city.radio.propagation(a, b, self.scenario)
            a5 = a + 5 * velocity[ii]
            b5 = b + 5 * vb
            a5[:, :2] = np.clip(a5[:, :2], -195, 195)
            b5[:, :2] = np.clip(b5[:, :2], -195, 195)
            loss5, los5, _ = self.city.radio.propagation(a5, b5, self.scenario)
            x[ii, jj, 22:27] = np.column_stack((loss, loss5, los, los5, loss5 - loss))
        with self.rng:
            tx, rx = beacon_frame(
                self.received,
                self.sinr,
                self.starts,
                self.ends,
                self.ns,
                self.threshold,
                0.5,
                t,
                self.beacon,
                self.beacon_mask,
                self.last,
                self.rxdb,
                self.rxsin,
                self.revstamp,
                self.revsin,
                self.poses,
                self.vels,
                self.pose_stamp,
                self.counts,
            )
        self.tx_count += tx
        self.rx_count += rx
        x[..., 28:48] = self.beacon[t]
        m[..., 28:48] = self.beacon_mask[t]
        command = np.zeros((n, n, 9), np.float32)
        for i, c in enumerate(self.goals):
            for j in range(n):
                if i != j and self.counts[t, i, j] > 0:
                    c.received_beacon(j, self.issued_time, self.desired[j], now)
            command[i] = c.encode(now, x[i], m[i])
        self.t += 1
        return x, m, command
