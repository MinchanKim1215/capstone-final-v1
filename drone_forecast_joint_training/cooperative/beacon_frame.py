"""수신 비콘의 원래 배치 계산을 그대로 한 프레임으로 분리."""

import numpy as np
from numba import njit
from swarm_sim.fast_network import bpsk_probability


@njit(cache=True)
def beacon_frame(
    received,
    sinr,
    xyz_start,
    xyz_end,
    ns,
    threshold_db,
    noise_sigma_db,
    t,
    z,
    m,
    last,
    rxdb,
    rxsin,
    revstamp,
    revsin,
    poses,
    vels,
    pose_stamp,
    counts,
):
    n = sinr.shape[1]
    dt = 0.5
    tx_count = 0
    rx_count = 0
    velocity = (xyz_end[t] - xyz_start[t]) / dt
    # 0,1,... 秒마다 8개 슬롯의 서로 다른 송신자. 같은 슬롯의 전역 동시 송신 없음.
    if t % 2 == 0:
        for slot in range(n):
            j = slot
            now = t * dt + slot * 0.005
            # 송신 순간의 과거 캐시만 메시지에 넣습니다. 나중에 받은 내용을 소급하지 않습니다.
            old_last = last[j].copy()
            old_sin = rxsin[j].copy()
            tx_count += 1
            for i in range(n):
                if i == j:
                    continue
                probability = bpsk_probability(sinr[t, j, i], 1024)
                if np.random.random() < probability:
                    rx_count += 1
                    counts[t, i, j] += 1
                    last[i, j] = now + 0.001024
                    # 표시 해상도 0.01dB. 측정 잡음과 실제 채널 확률의 명시적 분리.
                    rxsin[i, j] = (
                        round(
                            (sinr[t, j, i] + np.random.normal(0, noise_sigma_db)) * 100
                        )
                        / 100
                    )
                    rxdb[i, j] = (
                        round(
                            (received[t, j, i] + np.random.normal(0, noise_sigma_db))
                            * 100
                        )
                        / 100
                    )
                    pose_stamp[i, j] = t * dt
                    poses[i, j] = xyz_start[t, j]
                    vels[i, j] = velocity[j]
                    if old_last[i] >= 0:
                        revstamp[i, j] = old_last[i]
                        revsin[i, j] = old_sin[i]
    end = (t + 1) * dt
    for i in range(n):
        acceleration = 0.0
        if t > 0:
            prev = (xyz_end[t - 1, i] - xyz_start[t - 1, i]) / dt
            acceleration = np.sqrt(np.sum((velocity[i] - prev) ** 2)) / dt
        for j in range(n):
            if i == j:
                continue
            z[t, i, j, 5] = counts[t, i, j]
            m[t, i, j, 5] = True
            z[t, i, j, 16] = np.sum(counts[max(0, t - 9) : t + 1, i, j])
            m[t, i, j, 16] = True
            z[t, i, j, 17] = np.sum(counts[max(0, t - 19) : t + 1, i, j])
            m[t, i, j, 17] = True
            z[t, i, j, 18] = 0.0
            m[t, i, j, 18] = True
            z[t, i, j, 19] = acceleration
            m[t, i, j, 19] = True
            if last[i, j] < 0:
                continue
            age = end - last[i, j]
            z[t, i, j, 0] = rxdb[i, j]
            z[t, i, j, 1] = rxsin[i, j]
            z[t, i, j, 2] = age
            m[t, i, j, 0:3] = True
            if revstamp[i, j] >= 0:
                revage = end - revstamp[i, j]
                z[t, i, j, 3] = revsin[i, j]
                z[t, i, j, 4] = revage
                m[t, i, j, 3:5] = True
                minimum = min(rxsin[i, j], revsin[i, j])
                z[t, i, j, 12] = minimum
                z[t, i, j, 13] = minimum - threshold_db
                m[t, i, j, 12:14] = True
                z[t, i, j, 18] = 1.0 if age <= 2.0 and revage <= 2.0 else 0.0
            poseage = end - pose_stamp[i, j]
            if poseage <= 10.0:
                predicted = poses[i, j] + vels[i, j] * poseage
                for q in range(2):
                    predicted[q] = min(195.0, max(-195.0, predicted[q]))
                rel = predicted - xyz_end[t, i]
                dv = vels[i, j] - velocity[i]
                distance = np.sqrt(rel[0] ** 2 + rel[1] ** 2)
                z[t, i, j, 6:8] = rel[:2]
                z[t, i, j, 8:10] = dv[:2]
                z[t, i, j, 10] = distance
                z[t, i, j, 11] = (rel[0] * dv[0] + rel[1] * dv[1]) / max(distance, 1.0)
                m[t, i, j, 6:12] = True
            if t >= 6:
                if m[t - 6, i, j, 1] and age <= 2.0 and z[t - 6, i, j, 2] <= 2.0:
                    z[t, i, j, 14] = (rxsin[i, j] - z[t - 6, i, j, 1]) / 3.0
                    m[t, i, j, 14] = True
                if (
                    m[t - 6, i, j, 3]
                    and revstamp[i, j] >= 0
                    and end - revstamp[i, j] <= 2.0
                    and z[t - 6, i, j, 4] <= 2.0
                ):
                    z[t, i, j, 15] = (revsin[i, j] - z[t - 6, i, j, 3]) / 3.0
                    m[t, i, j, 15] = True
    return tx_count, rx_count
