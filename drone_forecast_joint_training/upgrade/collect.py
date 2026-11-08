"""전체 비행을 보존하는 수집기와 수신 데이터 헤더 기반의 관측 보강.

원래 BuildingRadio/BPSK/TDMA/라우팅/트래픽 수식은 그대로 호출합니다.
추가 위치·속도는 성공적으로 받은 데이터의 기존 64B 헤더 중 32B에 실린다고
명시합니다. 성공한 ACK만으로 상대 위치를 안다고 가정하지 않습니다.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
import math
import time

import numpy as np
import shapely

from swarm_sim.city import City
from swarm_sim.config import Design, Scenario
from swarm_sim.fast_network import probabilities, packet_run
from swarm_sim.forecast import atomic_npz
from swarm_sim.mission import full_mission
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from upgrade.map_setup import digest

BASE_FEATURES = [
    'self_velocity_x_mps', 'self_velocity_y_mps', 'self_altitude_m',
    'self_is_relay', 'peer_is_relay', 'tx_attempts_in_interval',
    'ack_successes_in_interval', 'ack_success_ratio_in_interval',
    'last_ack_received_power_dbm', 'last_ack_sinr_db',
    'seconds_since_last_ack', 'self_queue_packets',
]
EXTRA_FEATURES = [
    'self_position_x_m', 'self_position_y_m', 'inbound_successes_in_interval',
    'seconds_since_peer_state_timestamp', 'estimated_relative_x_m',
    'estimated_relative_y_m', 'last_received_relative_velocity_x_mps',
    'last_received_relative_velocity_y_mps', 'estimated_peer_distance_m',
    'estimated_distance_rate_mps', 'estimated_path_loss_now_db',
    'estimated_path_loss_after_5s_db', 'estimated_los_now',
    'estimated_los_after_5s', 'estimated_path_loss_change_5s_db',
    'self_distance_to_ge40m_building_m',
]
FEATURES = BASE_FEATURES + EXTRA_FEATURES


def source_hashes():
    names = ['swarm_sim/' + n + '.py' for n in ('config', 'city', 'radio', 'motion', 'mission', 'fast_network')]
    names += ['upgrade/map_setup.py', 'upgrade/collect.py']
    return {n: sha256_file(ROOT / n) for n in names}


def baseline_observations(xyz_start, xyz_end, received, sinr, attempts, successes, ack, queues, ns, dt):
    """원래 12개 입력을 벡터화합니다. 별도 회귀 검사로 원래 함수와 일치시킵니다."""
    nt, n, _ = attempts.shape
    x = np.zeros((nt, n, n, 12), np.float32)
    mask = np.zeros_like(x, bool)
    off = ~np.eye(n, dtype=bool)
    velocity = (xyz_end - xyz_start) / dt
    x[..., 0] = velocity[:, :, None, 0]
    x[..., 1] = velocity[:, :, None, 1]
    x[..., 2] = xyz_end[:, :, None, 2]
    x[..., 3] = (np.arange(n) >= ns)[None, :, None]
    x[..., 4] = (np.arange(n) >= ns)[None, None, :]
    x[..., 5] = attempts
    x[..., 6] = successes
    mask[..., :7] = off[None, ..., None]
    np.divide(successes, attempts, out=x[..., 7], where=attempts > 0)
    mask[..., 7] = (attempts > 0) & off
    index = np.maximum.accumulate(np.where(successes > 0, np.arange(nt)[:, None, None], -1), axis=0)
    when = np.maximum.accumulate(np.where(successes > 0, ack, -1.), axis=0)
    available = (index >= 0) & off
    last_rx = np.take_along_axis(received.swapaxes(1, 2), np.maximum(index, 0), axis=0)
    last_sinr = np.take_along_axis(sinr.swapaxes(1, 2), np.maximum(index, 0), axis=0)
    x[..., 8] = np.where(available, last_rx, 0)
    x[..., 9] = np.where(available, last_sinr, 0)
    x[..., 10] = np.where(available, np.maximum(0., (np.arange(nt)[:, None, None] + 1)*dt - when), 0)
    mask[..., 8:11] = available[..., None]
    x[..., 11] = queues[:, :, None]
    mask[..., 11] = off
    x[:, ~off] = 0
    return x, mask


def augment_observations(city, x, mask, xyz_start, xyz_end, dt, scenario, maximum_peer_age_s=10.):
    """과거에 실제 받은 peer 상태만 외삽합니다. 미래 궤적·채널은 인자가 아닙니다."""
    nt, n, _, _ = x.shape
    z = np.zeros((nt, n, n, len(EXTRA_FEATURES)), np.float32)
    valid = np.zeros_like(z, bool)
    off = ~np.eye(n, dtype=bool)
    velocity = (xyz_end - xyz_start) / dt
    z[..., 0] = xyz_end[:, :, None, 0]
    z[..., 1] = xyz_end[:, :, None, 1]
    # j→i 데이터가 도착해야 i가 j의 헤더에 든 위치·속도를 읽을 수 있습니다.
    inbound = x[..., 6].swapaxes(1, 2)
    z[..., 2] = inbound
    valid[..., :3] = off[None, ..., None]
    updates = inbound > 0
    last = np.maximum.accumulate(np.where(updates, np.arange(nt)[:, None, None], -1), axis=0)
    known = (last >= 0) & off
    age = (np.arange(nt)[:, None, None] + 1 - last) * dt
    z[..., 3] = np.where(known, age, 0)
    valid[..., 3] = known
    # 헤더의 위치·속도·시각은 해당 0.5초 구간 시작에 송신자가 기록한 값입니다.
    # 수신은 구간 끝까지 완료되어 있어, 예측 시점 이후 값은 포함되지 않습니다.
    peer = np.arange(n)[None, None, :]
    last_position = xyz_start[np.maximum(last, 0), peer]
    last_velocity = velocity[np.maximum(last, 0), peer]
    predicted_peer = last_position + last_velocity * age[..., None]
    predicted_peer[..., :2] = np.clip(predicted_peer[..., :2], -195, 195)
    relative = predicted_peer - xyz_end[:, :, None, :]
    relative_velocity = last_velocity - velocity[:, :, None, :]
    distance = np.linalg.norm(relative[..., :2], axis=-1)
    rate = np.sum(relative[..., :2] * relative_velocity[..., :2], axis=-1) / np.maximum(distance, 1.)
    fresh_enough = known & (age <= maximum_peer_age_s)
    z[..., 4:6] = relative[..., :2]
    z[..., 6:8] = relative_velocity[..., :2]
    z[..., 8] = distance
    z[..., 9] = rate
    valid[..., 4:15] = fresh_enough[..., None]
    where = np.where(fresh_enough)
    # 지도 기반 손실은 현재까지 수신한 상태의 추정입니다. 실제 미래 손실과 구분합니다.
    for first in range(0, len(where[0]), 4096):
        idx = tuple(a[first:first+4096] for a in where)
        a = xyz_end[idx[0], idx[1]].copy()
        b = predicted_peer[idx].copy()
        v_a = velocity[idx[0], idx[1]]
        v_b = last_velocity[idx]
        loss, los, _ = city.radio.propagation(a, b, scenario)
        a5, b5 = a + 5*v_a, b + 5*v_b
        a5[:, :2] = np.clip(a5[:, :2], -195, 195)
        b5[:, :2] = np.clip(b5[:, :2], -195, 195)
        loss5, los5, _ = city.radio.propagation(a5, b5, scenario)
        z[idx + (10,)] = loss
        z[idx + (11,)] = loss5
        z[idx + (12,)] = los
        z[idx + (13,)] = los5
        z[idx + (14,)] = loss5 - loss
    high = [g for g, h in zip(city.geometries, city.heights) if h >= 40]
    if high:
        geometries = np.asarray(high, object)
        tree = shapely.STRtree(geometries)
        points = shapely.points(xyz_end[..., :2].reshape(-1, 2))
        nearest = tree.nearest(points)
        distances = shapely.distance(points, geometries[nearest]).reshape(nt, n)
        z[..., 15] = distances[:, :, None]
    else:
        # 지도 대각선보다 큰 고정값으로 건물 없음이라는 알려진 상태를 표현합니다.
        z[..., 15] = math.sqrt(2)*400
    valid[..., 15] = off
    z[~valid] = 0
    if not np.isfinite(z).all():
        raise RuntimeError('보강 관측에 NaN/무한대가 있습니다.')
    return np.concatenate((x, z), axis=-1), np.concatenate((mask, valid), axis=-1)


def prepare_city_without_unused_radio_cache(path, scenario, cache_dir):
    """0.5초 수집에서 사용하지 않는 전체 격자 간 A2A 손실 행렬만 생략합니다."""
    city = City(path, scenario)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{city.key}_{city.sha256[:12]}_visibility.npz'
    if path.exists():
        with np.load(path) as data:
            city.visible = data['visible']
    else:
        city.visible = np.zeros((len(city.points), len(city.target_points)), bool)
        points = np.column_stack((city.points, np.full(len(city.points), 50.)))
        for i, p in enumerate(points):
            city.visible[i, city.sensing(p, scenario)] = True
        atomic_npz(path, visible=city.visible)
    city.observable = np.any(city.visible, axis=0)
    matrix = city.graph.toarray() > 0
    city.neighbors = np.full((len(city.points), 9), -1, np.int32)
    for i in range(len(city.points)):
        v = np.r_[i, np.flatnonzero(matrix[i])]
        city.neighbors[i, :len(v)] = v
    return city


def trajectory(city, design, scenario, cache_dir):
    """한 지역·방향의 전체 비행 및 실제 A2A 손실을 공유해 같은 계산을 반복하지 않습니다."""
    tag = digest({'map': city.sha256, 'design': asdict(design), 'scenario': asdict(scenario),
                  'source': source_hashes()})[:20]
    path = Path(cache_dir) / f'{city.key}_{scenario.direction}_{tag}.npz'
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        return arrays, read_json(meta_path)
    mission = full_mission(city, design, scenario)
    coarse = mission['nodes']
    segments = len(coarse) - 1
    k = np.repeat(np.arange(segments), 6)
    alpha = np.tile(np.arange(6)/6, segments)
    a, b = city.points[coarse[k]], city.points[coarse[k+1]]
    start = a*(1-alpha[:, None, None]) + b*alpha[:, None, None]
    end = a*(1-(alpha+1/6)[:, None, None]) + b*(alpha+1/6)[:, None, None]
    xyz = np.concatenate((start, np.full((*start.shape[:2], 1), 50.)), axis=-1)
    xyz_end = np.concatenate((end, np.full((*end.shape[:2], 1), 50.)), axis=-1)
    nt, n, _ = xyz.shape
    losses, los = np.empty((nt, n, n), np.float64), np.empty((nt, n, n), bool)
    for first in range(0, nt, 100):
        block = xyz[first:first+100]
        tx = np.repeat(block, n, axis=1).reshape(-1, 3)
        rx = np.tile(block, (1, n, 1)).reshape(-1, 3)
        aa, bb, _ = city.radio.propagation(tx, rx, scenario)
        losses[first:first+len(block)] = aa.reshape(-1, n, n)
        los[first:first+len(block)] = bb.reshape(-1, n, n)
    arrays = {'xyz_start': xyz, 'xyz_end': xyz_end, 'nodes': coarse[k], 'loss': losses, 'los': los}
    meta = {k: v for k, v in mission.items() if k not in ('nodes', 'desired', 'phase')}
    atomic_npz(path, **arrays)
    write_json(meta_path, meta)
    return arrays, meta


def collect_one(city, arrays, mission_meta, design, scenario, seed, kind, episode_dir, protocol, split):
    """전체 임무 기록입니다. 모델 입력과 정답용 잠재 채널을 다른 파일에 저장합니다."""
    before = time.perf_counter()
    xyz, xyz_end = arrays['xyz_start'], arrays['xyz_end']
    nt, n, _ = xyz.shape
    dt, duration = .5, nt*.5
    rng = np.random.default_rng(seed)
    jammer = np.r_[city.target_points[int(rng.integers(len(city.target_points)))], 2.]
    # 시작/종료 시각은 비행별로 달라지며 모델 입력에는 노출하지 않습니다.
    if kind == 'continuous':
        start, stop, power = 0., duration, float(rng.uniform(-3, 3))
    elif kind == 'intermittent':
        start = float(rng.uniform(.15, .45)*duration)
        stop = float(rng.uniform(.65, .90)*duration)
        power = float(rng.uniform(-3, 3))
    else:
        start, stop, power = 0., 0., 0.
    t = np.arange(nt)*dt
    active = (t >= start) & (t < stop)
    if kind == 'clear':
        interference = np.zeros((nt, n))
    else:
        jloss, _, _ = city.radio.propagation(jammer[None], xyz.reshape(-1, 3), scenario)
        interference = 10**((power-jloss.reshape(nt, n))/10)*active[:, None]
    received = np.empty((nt, n, n))
    sinr = np.empty_like(received)
    # 캐시도 배정밀도로 저장하여 원래 .5초 수집의 손실 값을 유지합니다.
    p = probabilities(arrays['loss'].astype(float), interference, dt, seed,
                      scenario.shadow_sigma_db, scenario.shadow_correlation_s, received, sinr)
    attempts = np.zeros((nt, n, n), np.int32)
    successes = np.zeros_like(attempts)
    ack = np.full((nt, n, n), -1.)
    queues = np.zeros((nt, n), np.int32)
    packet = packet_run(p, arrays['nodes'], city.visible, design.n_search, dt, seed+1000003,
                        False, attempts, successes, ack, queues)
    values, _, graph = packet[:3]
    if values[0] != sum(values[i] for i in (1, 5, 6, 7)):
        raise RuntimeError('패킷 보존식 위반')
    base, mask = baseline_observations(xyz, xyz_end, received, sinr, attempts, successes, ack, queues, design.n_search, dt)
    features, valid = augment_observations(city, base, mask, xyz, xyz_end, dt, scenario)
    available = (p >= .9) & (p.swapaxes(1, 2) >= .9)
    atomic_npz(episode_dir / 'observations.npz', features=features, mask=valid,
               time_end_s=t+dt, feature_names=np.asarray(FEATURES))
    atomic_npz(episode_dir / 'truth.npz', link_available=available, hop_success_probability=p.astype(np.float32),
               positions_start_m=xyz.astype(np.float32), search_connected=graph[:, 0].astype(bool),
               jammer_active=active, time_end_s=t+dt)
    metadata = {
        'format': 'forecast-upgrade-episode-v1', 'complete': True, 'region': city.key, 'split': split,
        'kind': kind, 'seed': seed, 'direction': scenario.direction, 'frames': nt, 'duration_s': duration,
        'recording_truncated': False, 'mission': mission_meta, 'map': city.summary(),
        'design': asdict(design), 'scenario': asdict(scenario),
        'jammer_xyz_m': jammer.tolist(), 'jammer_power_dbm': power, 'jammer_window_s': [start, stop],
        'protocol_sha256': protocol['protocol_sha256'], 'source_sha256': source_hashes(),
        'metrics': {'generated': int(values[0]), 'delivered': int(values[1]), 'pdr': float(values[1]/max(values[0], 1)),
                    'search_connected_fraction': float(graph[:, 0].mean())},
        'feature_names': FEATURES, 'peer_state_observation_fraction': float(valid[..., 16].mean()),
        'elapsed_s': time.perf_counter()-before,
        'files_sha256': {n: sha256_file(episode_dir/n) for n in ('observations.npz', 'truth.npz')},
        'limits': [
            'Map heights include explicitly marked regional-mean imputation; flat ground and fixed altitude 50m.',
            'Global ideal routing is retained; global channel information is not a prediction input.',
            'Peer telemetry is conservatively recorded only on successful inbound data+ACK exchanges.',
            'Data header 64B already in airtime accounting reserves 32B for source timestamp/position/velocity.',
            'No extra beacon/probe slots; sampled source state timestamp is the radio-interval start.',
            'Estimated future geometry uses constant-velocity extrapolation, not actual future positions.',
            'This is prediction data collection, not prediction-controlled repositioning.',
        ],
    }
    write_json(episode_dir / 'episode.json', metadata)
    return metadata


def collect_region(key, protocol_path, kinds=('clear', 'continuous', 'intermittent'), directions=('east_west', 'north_south')):
    protocol = read_json(protocol_path)
    entry = next(e for e in protocol['maps'] if e['key'] == key)
    design = Design(**protocol['design']).validate()
    cache = ROOT / 'cache' / 'upgrade'
    first_scenario = Scenario(duration_s=7200, motion_dt_s=3, direction=directions[0]).validate()
    started = time.perf_counter()
    city = prepare_city_without_unused_radio_cache(ROOT / entry['path'], first_scenario, cache)
    records = []
    for direction in directions:
        scenario = Scenario(duration_s=7200, motion_dt_s=3, direction=direction).validate()
        city.route = city.make_sweep(scenario)
        arrays, meta = trajectory(city, design, scenario, cache)
        for kind in kinds:
            seed = int(digest([protocol['seed'], key, direction, kind])[:8], 16) % (2**31-1000004)
            episode_id = f'{key}_{direction}_{kind}_s{seed}'
            target = ROOT / 'datasets' / 'upgrade45' / 'episodes' / episode_id
            done = target / 'episode.json'
            if done.exists():
                item = read_json(done)
                if (item['protocol_sha256'] != protocol['protocol_sha256'] or item['source_sha256'] != source_hashes()
                    or any(sha256_file(target/n) != h for n, h in item['files_sha256'].items())):
                    raise ValueError(f'{episode_id}: 이전 파일의 설정·코드·해시가 다릅니다.')
                reused = True
            else:
                target.mkdir(parents=True, exist_ok=True)
                item = collect_one(city, arrays, meta, design, scenario, seed, kind, target, protocol, entry['split'])
                reused = False
            records.append({'id': episode_id, 'region': key, 'split': entry['split'],
                            'path': target.relative_to(ROOT).as_posix(), 'metadata': item})
            print(json.dumps({'stage': 'episode', 'region': key, 'direction': direction, 'kind': kind,
                              'duration_s': item['duration_s'], 'reused': reused,
                              'elapsed_s': round(item['elapsed_s'], 2)}, ensure_ascii=False), flush=True)
    print(json.dumps({'stage': 'region_complete', 'region': key, 'episodes': len(records),
                      'elapsed_s': round(time.perf_counter()-started, 2)}, ensure_ascii=False), flush=True)
    return records


def collect_all(workers=3, regions=None, kinds=('clear', 'continuous', 'intermittent'), directions=('east_west', 'north_south')):
    path = ROOT / 'configs' / 'upgrade_protocol.json'
    protocol = read_json(path)
    keys = regions or protocol['splits']['train'] + protocol['splits']['val'] + protocol['splits']['test']
    records, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(collect_region, key, path, kinds, directions): key for key in keys}
        for future in as_completed(jobs):
            try:
                records.extend(future.result())
            except Exception as error:
                failures.append({'region': jobs[future], 'error': repr(error)})
                print(json.dumps({'stage': 'region_failed', **failures[-1]}, ensure_ascii=False), flush=True)
    result = {'format': 'upgrade-collection-v1', 'protocol_sha256': protocol['protocol_sha256'],
              'regions_requested': keys, 'kinds': list(kinds), 'directions': list(directions),
              'complete': not failures, 'records': sorted(records, key=lambda r: r['id']), 'failures': failures}
    write_json(ROOT / 'datasets' / 'upgrade45' / 'collection_manifest.json', result)
    return result
