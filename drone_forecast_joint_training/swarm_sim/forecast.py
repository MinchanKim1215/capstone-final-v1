"""모델 선택과 독립적인, 인과 시각을 보존한 수동 관측 데이터 수집.

입력: 구간 끝까지 송신자가 얻은 ACK·시도 기록과 자기 상태.
정답용 진실값: 구간 시작에 계산되어 그 구간에 유지된 실제 홉 확률.
과거 HTML 프레임은 입력으로 사용하지 않는다. 기존 수식·트래픽·비행은 유지한다.
"""
from dataclasses import asdict
from pathlib import Path
import os
import tempfile

import numpy as np

from .config import Design, Scenario
from .fast_network import probabilities, packet_run
from .mission import MOVE_DT, full_mission
from .regions import write_json, sha256_file

FEATURES = [
    'self_velocity_x_mps', 'self_velocity_y_mps', 'self_altitude_m',
    'self_is_relay', 'peer_is_relay', 'tx_attempts_in_interval',
    'ack_successes_in_interval', 'ack_success_ratio_in_interval',
    'last_ack_received_power_dbm', 'last_ack_sinr_db',
    'seconds_since_last_ack', 'self_queue_packets',
]
FEATURE_DESCRIPTIONS = [
    '자기 드론이 방금 완료한 구간의 동쪽 속도', '자기 드론의 북쪽 속도',
    '자기 고도', '초기 임무에서 정한 자기 역할', '초기 임무에서 정한 상대 역할',
    '이 상대에게 직접 송신한 실제 시도 수', '데이터+ACK 교환에 성공한 실제 횟수',
    '성공/시도; 시도 0이면 값 0·마스크 false',
    '마지막 ACK 성공 때 수신기 측정을 가정한 역방향 수신 전력; 이후 유지',
    '같은 마지막 ACK의 SINR; 이후 유지',
    '마지막 성공 ACK 이후 경과 시간; 미수신이면 마스크 false',
    '구간 끝에 자기 드론이 직접 보유한 큐 길이',
]


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def local_observations(xyz_end, xyz_start, received, sinr, attempts, successes,
                       last_ack, queues, ns, dt):
    """성공하지 않은 프레임의 실제 신호값을 입력에 노출하지 않는다."""
    nt, n, _ = attempts.shape
    x = np.zeros((nt, n, n, len(FEATURES)), dtype=np.float32)
    mask = np.zeros_like(x, dtype=np.bool_)
    roles = np.arange(n) >= ns
    velocity = (xyz_end - xyz_start) / dt
    last_time = np.full((n, n), -1.)
    last_power = np.zeros((n, n))
    last_sinr = np.zeros((n, n))
    for k in range(nt):
        end = (k + 1) * dt
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                row = x[k, i, j]
                valid = mask[k, i, j]
                row[:7] = [velocity[k, i, 0], velocity[k, i, 1], xyz_end[k, i, 2],
                           roles[i], roles[j], attempts[k, i, j], successes[k, i, j]]
                valid[:7] = True
                if attempts[k, i, j] > 0:
                    row[7] = successes[k, i, j] / attempts[k, i, j]
                    valid[7] = True
                if successes[k, i, j] > 0:
                    when = last_ack[k, i, j]
                    if not k * dt <= when <= end + 1e-7:
                        raise RuntimeError('ACK 관측이 현재 구간 밖에 있습니다.')
                    # ACK는 j→i입니다. 잠재 신호 전체가 아닌, 실제 ACK 성공 시점만 기록합니다.
                    last_time[i, j] = when
                    last_power[i, j] = received[k, j, i]
                    last_sinr[i, j] = sinr[k, j, i]
                if last_time[i, j] >= 0:
                    row[8:11] = [last_power[i, j], last_sinr[i, j], max(0., end - last_time[i, j])]
                    valid[8:11] = True
                row[11] = queues[k, i]
                valid[11] = True
    if not np.isfinite(x).all():
        raise RuntimeError('학습 입력에 유한하지 않은 값이 있습니다.')
    return x, mask


def collect_episode(city, design, scenario, seed, jammer_on=True, radio_dt=0.5,
                    max_duration_s=None, jammer_window_s=None, mission=None):
    """한 번의 환경 실행이다. 모델 가중치나 과거 지역의 상태를 보관하지 않는다."""
    design.validate()
    scenario.validate()
    if (scenario.motion_dt_s != 3 or scenario.altitude_m != 50 or scenario.grid_m != 10
        or scenario.frequency_hz != 2.4e9 or scenario.tx_power_dbm != 10
        or scenario.gain_dbi != 0 or scenario.bandwidth_hz != 1e6
        or scenario.bit_rate_bps != 1e6 or scenario.noise_figure_db != 7
        or scenario.mac != 'tdma' or scenario.routing != 'expected_attempts'
        or scenario.radio_model != 'bullington' or scenario.slot_s != 0.005
        or scenario.link_success_threshold != 0.9 or scenario.total_packets_per_s != 30
        or scenario.traffic_mode != 'fixed_total' or scenario.packet_payload_bytes != 512
        or scenario.packet_header_bytes != 64 or scenario.ack_bytes != 14
        or scenario.queue_limit != 64 or scenario.max_hop_attempts != 4
        or scenario.packet_lifetime_s != 2 or scenario.timely_deadline_s != .25
        or scenario.min_separation_m != 5 or scenario.relay_failure_fraction is not None):
        raise ValueError('기존 전체 엔진의 고정 채널·패킷·안전 설정과 다릅니다.')
    if radio_dt not in (0.5, 3.0):
        raise ValueError('채널 주기는 0.5초 또는 3초입니다.')
    if jammer_on and scenario.jammer_power_dbm is None:
        raise ValueError('간섭 ON에는 수치 출력이 필요합니다.')
    mission = full_mission(city, design, scenario) if mission is None else mission
    planned = float(mission['duration_s'])
    duration = planned
    if max_duration_s is not None:
        if max_duration_s < MOVE_DT or not np.isfinite(max_duration_s) or not np.isclose(max_duration_s / MOVE_DT, round(max_duration_s / MOVE_DT)):
            raise ValueError('기록 상한은 3초 이상의 3초 배수여야 합니다.')
        duration = min(planned, float(max_duration_s))
    segments = int(round(duration / MOVE_DT))
    coarse = mission['nodes'][:segments + 1]
    ratio = int(round(MOVE_DT / radio_dt))
    k = np.repeat(np.arange(segments), ratio)
    alpha = np.tile(np.arange(ratio) / ratio, segments)
    start = city.points[coarse[k]]
    end = city.points[coarse[k + 1]]
    xy_start = start * (1-alpha[:, None, None]) + end * alpha[:, None, None]
    xy_end = start * (1-(alpha+1/ratio)[:, None, None]) + end * (alpha+1/ratio)[:, None, None]
    nt, n, _ = xy_start.shape
    xyz_start = np.concatenate([xy_start, np.full((nt, n, 1), 50.)], axis=2)
    xyz_end = np.concatenate([xy_end, np.full((nt, n, 1), 50.)], axis=2)
    times = np.arange(nt) * radio_dt
    if ratio == 1:
        loss = city.loss_cache[coarse[k, :, None], coarse[k, None, :]].astype(float)
        los = city.los_cache[coarse[k, :, None], coarse[k, None, :]]
    else:
        loss = np.empty((nt, n, n))
        los = np.empty((nt, n, n), bool)
        for first in range(0, nt, 100):
            block = xyz_start[first:first + 100]
            tx = np.repeat(block, n, axis=1).reshape(-1, 3)
            rx = np.tile(block, (1, n, 1)).reshape(-1, 3)
            a, b, _ = city.radio.propagation(tx, rx, scenario)
            loss[first:first+len(block)] = a.reshape(len(block), n, n)
            los[first:first+len(block)] = b.reshape(len(block), n, n)
    rng = np.random.default_rng(seed)
    jcell = int(rng.integers(len(city.target_points)))
    jammer = np.r_[city.target_points[jcell], scenario.jammer_height_m]
    if jammer_window_s is None:
        window = [scenario.jammer_start_fraction * duration, scenario.jammer_end_fraction * duration]
        window_mode = 'recording_fraction'
    else:
        window = list(jammer_window_s)
        if len(window) != 2 or not all(np.isfinite(window)) or not 0 <= window[0] < window[1]:
            raise ValueError('간섭 시각은 [시작초, 종료초]이며 0≤시작<종료여야 합니다.')
        window_mode = 'absolute_seconds'
    active = (times >= window[0]) & (times < window[1]) & jammer_on
    jloss, _, _ = city.radio.propagation(jammer[None, :], xyz_start.reshape(-1, 3), scenario)
    power = scenario.jammer_power_dbm if scenario.jammer_power_dbm is not None else 0.
    interference = 10.**((power-jloss.reshape(nt, n))/10) * active[:, None]
    received = np.empty_like(loss)
    sinr = np.empty_like(loss)
    probability = probabilities(loss, interference, radio_dt, seed,
                                scenario.shadow_sigma_db, scenario.shadow_correlation_s,
                                received, sinr)
    attempts = np.zeros((nt, n, n), dtype=np.int32)
    successes = np.zeros_like(attempts)
    ack = np.full((nt, n, n), -1.)
    queues = np.zeros((nt, n), dtype=np.int32)
    packet = packet_run(probability, coarse[k], city.visible, design.n_search,
                        radio_dt, seed + 1000003, False, attempts, successes, ack, queues)
    values, _, graph = packet[:3]
    if values[0] != values[1] + values[5] + values[6] + values[7]:
        raise RuntimeError('패킷 보존식 위반')
    x, mask = local_observations(xyz_end, xyz_start, received, sinr, attempts,
                                 successes, ack, queues, design.n_search, radio_dt)
    available = (probability >= .9) & (probability.transpose(0, 2, 1) >= .9)
    observations = {'time_end_s': times + radio_dt, 'features': x, 'feature_mask': mask,
                    'feature_names': np.asarray(FEATURES)}
    truth = {'time_start_s': times, 'time_end_s': times + radio_dt,
             'link_available': available, 'hop_success_probability': probability,
             'path_loss_db': loss.astype(np.float32), 'los': los,
             'positions_start_m': xyz_start.astype(np.float32),
             'external_interference_mw': interference.astype(np.float32),
             'search_connected': graph[:, 0].astype(bool), 'jammer_active': active}
    meta = {'region': city.key, 'map': city.summary(), 'design': asdict(design),
            'scenario': asdict(scenario), 'seed': int(seed), 'jammer_on': bool(jammer_on),
            'jammer_xyz_m': jammer.tolist(), 'jammer_window_s': window,
            'jammer_window_mode': window_mode, 'radio_dt_s': radio_dt,
            'duration_s': duration, 'planned_mission_duration_s': planned,
            'recording_truncated': duration < planned,
            'mission_completed': bool(mission['mission_completed'] and duration == planned),
            'frames': nt, 'features': FEATURES,
            'telemetry_excludes_post_mission_drain': True,
            'metrics': {'generated': int(values[0]), 'delivered': int(values[1]),
                        'pdr': float(values[1]/max(1, values[0])),
                        'search_connected_fraction': float(graph[:, 0].mean()),
                        'tx_attempts_including_drain': int(values[2]),
                        'drop_expired': int(values[5]), 'drop_retry_limit': int(values[6]),
                        'drop_queue_overflow': int(values[7])},
            'observation_contract': '송신자의 시도·성공 ACK·자기 상태. ACK 미수신 신호는 새로 관측하지 않음.',
            'limits': ['ACK 수신 전력/SINR은 해당 성공 구간의 모델값을 이상적 수신기 측정으로 사용함.',
                       'RSSI 측정 오차·추가 probe 트래픽은 모사하지 않음. 미송신 링크는 정보가 없을 수 있음.',
                       '패킷 라우팅은 기존과 동일한 이상적 전역 정보 가정. 이를 모델 입력으로 노출하지 않음.',
                       '학습용 데이터 수집이며 DL 모델 학습·예측 기반 재배치를 수행한 결과가 아님.']}
    return observations, truth, meta


def save_episode(folder, observations, truth, meta):
    folder = Path(folder)
    atomic_npz(folder / 'observations.npz', **observations)
    atomic_npz(folder / 'truth.npz', **truth)
    meta = dict(meta)
    meta['files_sha256'] = {name: sha256_file(folder / name) for name in ('observations.npz', 'truth.npz')}
    # episode.json이 마지막에 완성되어야 재개 시 완료된 실행으로 취급한다.
    write_json(folder / 'episode.json', meta)
    return meta
