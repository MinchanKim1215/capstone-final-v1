"""지역 단위 실험 계획, 중단 후 재개, 미래 정답과 학습/검증/시험 분리."""
from dataclasses import asdict
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import platform
import sys

import numpy as np

from .config import Design, Scenario, from_dict
from .forecast import collect_episode, save_episode, atomic_npz, FEATURES
from .mission import full_mission, prepared_city
from .regions import ROOT, DEFAULT_REGISTRY, KEY, load_regions, read_json, write_json, sha256_file

SPLITS = ('train', 'val', 'test')
ENV_FIELDS = {'direction', 'reference_speed_mps', 'shadow_sigma_db', 'shadow_correlation_s',
              'jammer_power_dbm', 'jammer_height_m', 'jammer_start_fraction', 'jammer_end_fraction'}


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def source_hashes():
    """문서·출력 경로가 아닌 실제 데이터 생성·라벨 코드 버전을 기록한다."""
    names = ['swarm_sim/' + name + '.py' for name in
             ('config', 'city', 'radio', 'motion', 'mission', 'fast_network', 'regions', 'forecast', 'forecast_data')]
    return {name: sha256_file(ROOT / name) for name in names}


def steps(seconds, dt, name):
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds) or seconds <= 0
        or not math.isclose(seconds / dt, round(seconds / dt), abs_tol=1e-8)):
        raise ValueError(f'{name}은 양수이고 채널 주기의 정수 배여야 합니다.')
    return int(round(seconds / dt))


def load_experiment(path, output=None):
    """지역 선택과 난수 분리는 데이터를 수집하기 전에 고정한다."""
    path = Path(path).resolve()
    config = read_json(path)
    required = {'format', 'registry', 'output', 'design', 'radio_dt_s', 'max_duration_s',
                'splits', 'environments', 'dataset'}
    if not isinstance(config, dict) or set(config) != required or config['format'] != 'forecast-experiment-v1':
        raise ValueError(f'forecast-experiment-v1 필드를 확인하십시오: {sorted(required)}')
    registry = (path.parent / config['registry']).resolve()
    regions = load_regions(registry)
    design = from_dict(Design, config['design'])
    dt = config['radio_dt_s']
    if isinstance(dt, bool) or dt not in (0.5, 3.0):
        raise ValueError('radio_dt_s는 0.5 또는 3입니다.')
    maximum = config['max_duration_s']
    if maximum is not None:
        steps(maximum, 3., 'max_duration_s')
    ds = config['dataset']
    if not isinstance(ds, dict) or set(ds) != {'lookback_s', 'horizon_s', 'min_outage_s', 'stride_s', 'min_observed_history_steps'}:
        raise ValueError('dataset 설정에 알 수 없거나 빠진 필드가 있습니다.')
    w, h, persist, stride = [steps(ds[key], dt, key) for key in ('lookback_s', 'horizon_s', 'min_outage_s', 'stride_s')]
    if persist > h:
        raise ValueError('min_outage_s는 horizon_s 이하여야 합니다.')
    minimum = ds['min_observed_history_steps']
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 0 <= minimum <= w:
        raise ValueError('min_observed_history_steps는 0~과거 구간 수 사이 정수입니다.')
    if maximum is not None and maximum / dt < w + h:
        raise ValueError('기록 상한이 과거 창과 미래 정답 구간을 담기에 부족합니다.')
    if set(config['splits']) != set(SPLITS):
        raise ValueError('train, val, test 분할을 모두 정하십시오.')
    for split in SPLITS:
        part = config['splits'][split]
        if set(part) != {'regions', 'seeds'} or not part['regions'] or not part['seeds']:
            raise ValueError(f'{split}: 비어 있지 않은 regions와 seeds가 필요합니다.')
        if len(set(part['regions'])) != len(part['regions']) or set(part['regions']) - set(regions):
            raise ValueError(f'{split}: 중복 또는 미등록 지역입니다.')
        if len(set(part['seeds'])) != len(part['seeds']) or any(
            isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**31-1000003 for x in part['seeds']):
            raise ValueError(f'{split}: 난수 시드는 중복 없는 0~2³¹−1000004 정수입니다.')
    train, val, test = [config['splits'][x] for x in SPLITS]
    if set(test['regions']) & (set(train['regions']) | set(val['regions'])):
        raise ValueError('시험 지역은 학습·검증 지역과 분리해야 합니다.')
    if (set(train['regions']) & set(val['regions'])) and (set(train['seeds']) & set(val['seeds'])):
        raise ValueError('같은 지역·같은 시드의 실행을 train과 val에 나눌 수 없습니다.')
    seen_geometries = {regions[k].audit['geometry_sha256'] for k in train['regions'] + val['regions']}
    if any(regions[k].audit['geometry_sha256'] in seen_geometries for k in test['regions']):
        raise ValueError('이름만 바꾼 동일 건물 기하가 시험 지역에 있습니다.')
    envs = config['environments']
    if not isinstance(envs, list) or not envs:
        raise ValueError('환경 변형이 하나 이상 필요합니다.')
    names = set()
    for env in envs:
        if not isinstance(env, dict) or set(env) != {'name', 'jammer_on', 'jammer_window_s', 'scenario'}:
            raise ValueError('환경에는 name, jammer_on, jammer_window_s, scenario가 필요합니다.')
        if not isinstance(env['name'], str) or not KEY.fullmatch(env['name']) or env['name'] in names:
            raise ValueError('환경 이름은 고유한 소문자 영문·숫자·밑줄 이름이어야 합니다.')
        names.add(env['name'])
        if not isinstance(env['jammer_on'], bool) or not isinstance(env['scenario'], dict) or set(env['scenario']) - ENV_FIELDS:
            raise ValueError('간섭 유무 또는 지원하지 않는 환경 변형입니다.')
        if any(isinstance(v, bool) for v in env['scenario'].values()):
            raise ValueError('수치 시나리오 설정에 true/false를 넣지 마십시오.')
        scenario = from_dict(Scenario, {'duration_s': 7200, 'motion_dt_s': 3, **env['scenario']})
        if not math.isfinite(scenario.jammer_height_m) or scenario.jammer_height_m <= 0:
            raise ValueError('간섭원 높이는 지면 위 양수 미터여야 합니다.')
        if env['jammer_on'] and scenario.jammer_power_dbm is None:
            raise ValueError('간섭 ON에 jammer_power_dbm이 필요합니다.')
        window = env['jammer_window_s']
        if window is not None and (not isinstance(window, list) or len(window) != 2
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in window)
            or not 0 <= window[0] < window[1]):
            raise ValueError('jammer_window_s는 null 또는 [시작초, 종료초]입니다.')
    folder = Path(output).resolve() if output else (path.parent / config['output']).resolve()
    used = set(k for part in config['splits'].values() for k in part['regions'])
    contract = {'format': 'forecast-collection-v1', 'config': config,
                'maps': {k: {name:regions[k].audit[name] for name in
                             ('sha256', 'geometry_sha256', 'name', 'buildings', 'imputed_buildings',
                              'center_lonlat', 'is_demo', 'warnings')} for k in sorted(used)},
                'source_sha256': source_hashes()}
    return config, regions, design, folder, contract


def plan_jobs(config, regions, design, contract):
    jobs = []
    physical = set()
    for split in SPLITS:
        part = config['splits'][split]
        for key, seed, env in product(part['regions'], part['seeds'], config['environments']):
            scenario = from_dict(Scenario, {'duration_s': 7200, 'motion_dt_s': 3, **env['scenario']})
            payload = {'map_sha256': regions[key].audit['sha256'], 'seed': seed,
                       'design': asdict(design), 'scenario': asdict(scenario),
                       'jammer_on': env['jammer_on'], 'jammer_window_s': env['jammer_window_s'],
                       'max_duration_s': config['max_duration_s'], 'radio_dt_s': config['radio_dt_s'],
                       'source_sha256': contract['source_sha256']}
            ident = digest(payload)
            if ident in physical:
                raise ValueError('이름만 다른 중복 실행 또는 분할 간 같은 실행이 있습니다.')
            physical.add(ident)
            jobs.append({'id': f"{key}_{env['name']}_s{seed}_{ident[:12]}", 'fingerprint': ident,
                         'split': split, 'region': key, 'seed': seed, 'environment': env['name'],
                         'scenario': scenario, 'environment_config': env})
    return jobs


def checked_episode(folder, job):
    meta = read_json(folder / 'episode.json')
    if meta.get('episode_fingerprint') != job['fingerprint'] or meta.get('split') != job['split']:
        raise ValueError('기존 실행의 지도·설정·코드·분할이 다릅니다. 새 출력 폴더를 쓰십시오.')
    for name, expected in meta['files_sha256'].items():
        if name not in ('observations.npz', 'truth.npz') or sha256_file(folder / name) != expected:
            raise ValueError('기존 실행 데이터가 손상되거나 바뀌었습니다.')
    return meta


def collect_single(region_key, folder, registry=DEFAULT_REGISTRY, seed=31,
                   jammer_on=True, max_duration_s=None, direction='east_west'):
    """등록한 한 지역을 바로 시험한다. 아직 데이터 분할에 넣지 않은 실행이다."""
    regions = load_regions(registry)
    if region_key not in regions:
        raise ValueError(f'미등록 지역: {region_key}')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31-1000003:
        raise ValueError('지원 범위의 정수 시드가 필요합니다.')
    scenario = Scenario(duration_s=7200, motion_dt_s=3, direction=direction).validate()
    design = Design(8, 5, 'v', 40).validate()
    region = regions[region_key]
    payload = {'map_sha256':region.audit['sha256'], 'seed':seed, 'jammer_on':jammer_on,
               'max_duration_s':max_duration_s, 'scenario':asdict(scenario),
               'design':asdict(design), 'source_sha256':source_hashes()}
    fingerprint = digest(payload)
    folder = Path(folder).resolve()
    if (folder/'collection_manifest.json').exists():
        raise ValueError('여러 실행의 수집 폴더입니다. 단일 실행은 새 하위 폴더를 사용하십시오.')
    if (folder/'episode.json').exists():
        return checked_episode(folder, {'fingerprint':fingerprint, 'split':'unassigned'})
    city = prepared_city(region.path, scenario, ROOT/'cache')
    obs, truth, meta = collect_episode(city,design,scenario,seed,jammer_on,.5,max_duration_s)
    meta.update({'episode_id':f'{region_key}_single_s{seed}_{fingerprint[:12]}',
                 'episode_fingerprint':fingerprint,'split':'unassigned',
                 'geometry_sha256':region.audit['geometry_sha256'],'source_sha256':payload['source_sha256']})
    return save_episode(folder,obs,truth,meta)


def collect(config_path, output=None):
    config, regions, design, folder, contract = load_experiment(config_path, output)
    jobs = plan_jobs(config, regions, design, contract)
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder / 'collection_plan.json'
    if plan_path.exists() and read_json(plan_path) != contract:
        raise ValueError('이 출력 폴더는 다른 지도·설정·코드의 자료입니다. 새 출력 폴더를 사용하십시오.')
    write_json(plan_path, contract)
    records, cities, missions = [], {}, {}
    newly_collected = 0
    for number, job in enumerate(jobs, 1):
        target = folder / 'episodes' / job['id']
        if (target / 'episode.json').exists():
            meta = checked_episode(target, job)
            reused = True
        else:
            newly_collected += 1
            scenario = job['scenario']
            ckey = (job['region'], scenario.direction)
            if ckey not in cities:
                cities[ckey] = prepared_city(regions[job['region']].path, scenario, ROOT / 'cache')
            city = cities[ckey]
            motion_key = (ckey, scenario.direction, scenario.reference_speed_mps)
            if motion_key not in missions:
                missions[motion_key] = full_mission(city, design, scenario)
            obs, truth, meta = collect_episode(city, design, scenario, job['seed'],
                job['environment_config']['jammer_on'], config['radio_dt_s'],
                config['max_duration_s'], job['environment_config']['jammer_window_s'], missions[motion_key])
            meta.update({'episode_id': job['id'], 'episode_fingerprint': job['fingerprint'],
                         'split': job['split'], 'environment': job['environment'],
                         'geometry_sha256': regions[job['region']].audit['geometry_sha256']})
            meta = save_episode(target, obs, truth, meta)
            reused = False
        records.append({'id': job['id'], 'split': job['split'], 'region': job['region'],
                        'path': 'episodes/' + job['id'], 'frames': meta['frames'],
                        'duration_s': meta['duration_s'], 'recording_truncated': meta['recording_truncated'],
                        'mission_completed': meta['mission_completed'], 'files_sha256': meta['files_sha256'],
                        'episode_json_sha256': sha256_file(target / 'episode.json')})
        print(json.dumps({'episode': number, 'total': len(jobs), 'region': job['region'],
                          'split': job['split'], 'reused': reused,
                          'duration_s': meta['duration_s'], 'truncated': meta['recording_truncated']}, ensure_ascii=False), flush=True)
    manifest_path = folder / 'collection_manifest.json'
    if newly_collected == 0 and manifest_path.exists():
        previous = read_json(manifest_path)
        if (previous.get('complete') and previous.get('contract_sha256') == digest(contract)
            and previous.get('episodes') == records):
            return previous
    manifest = {'format': 'forecast-collection-v1', 'complete': True, 'contract_sha256': digest(contract),
                'expected_episodes': len(jobs), 'episodes': records, 'config': config,
                'maps': contract['maps'], 'source_sha256': contract['source_sha256'],
                'python': sys.version, 'platform': platform.platform(),
                'model_trained': False, 'historical_design_selection_is_not_an_independent_holdout': True}
    write_json(manifest_path, manifest)
    return manifest


def window_labels(available, observed, lookback, horizon, persistence, stride=1, minimum=0):
    """입력 끝 k 뒤의 k+1..k+H만 사용한다. 미래가 모자란 창은 버린다.

    y=1: 향후 H구간 안에 연속 D구간 이상 링크가 불가용함.
    이미 끊겨 있던 링크의 지속도 양성이다. 단절 시작 시점만 예측하는 라벨은 아니다.
    """
    available = np.asarray(available, dtype=bool)
    nt, n, other = available.shape
    if n != other or observed.shape != available.shape or min(lookback, horizon, persistence, stride) < 1 or persistence > horizon:
        raise ValueError('라벨 배열·시간 창 설정 오류')
    bad = (~available).astype(np.int32)
    cumulative = np.concatenate([np.zeros((1, n, n), np.int32), np.cumsum(bad, axis=0)], axis=0)
    persistent = cumulative[persistence:] - cumulative[:-persistence] >= persistence
    prefix = np.concatenate([np.zeros((1, n, n), np.int32), np.cumsum(persistent, axis=0)], axis=0)
    seen = np.concatenate([np.zeros((1, n, n), np.int32), np.cumsum(observed, axis=0)], axis=0)
    indices, targets = [], []
    off_diagonal = ~np.eye(n, dtype=bool)
    for k in range(lookback - 1, nt - horizon, stride):
        y = prefix[k + horizon - persistence + 2] - prefix[k + 1] > 0
        eligible = (seen[k + 1] - seen[k - lookback + 1] >= minimum) & off_diagonal
        src, dst = np.where(eligible)
        if len(src):
            indices.append(np.column_stack([np.full(len(src), k), src, dst]).astype(np.int32))
            targets.append(y[src, dst].astype(np.uint8))
    return (np.concatenate(indices) if indices else np.zeros((0, 3), np.int32),
            np.concatenate(targets) if targets else np.zeros(0, np.uint8))


def build_dataset(folder, window_config=None, view=None):
    folder = Path(folder).resolve()
    manifest = read_json(folder / 'collection_manifest.json')
    contract = read_json(folder / 'collection_plan.json')
    if not manifest.get('complete') or manifest['contract_sha256'] != digest(contract):
        raise ValueError('완료된 자료 수집 manifest가 필요합니다.')
    dt = manifest['config']['radio_dt_s']
    ds = read_json(window_config) if window_config else manifest['config']['dataset']
    if set(ds) != {'lookback_s', 'horizon_s', 'min_outage_s', 'stride_s', 'min_observed_history_steps'}:
        raise ValueError('시간 창 설정 필드를 확인하십시오.')
    if window_config and not view:
        raise ValueError('다른 시간 창을 만들 때 --view 이름을 지정하십시오. 기존 자료를 덮어쓰지 않습니다.')
    if view is not None and (not isinstance(view, str) or not KEY.fullmatch(view)):
        raise ValueError('view는 영문 소문자로 시작하는 이름이어야 합니다.')
    destination = folder / 'views' / view if view else folder
    w, h, d, stride = [steps(ds[key], dt, key) for key in ('lookback_s', 'horizon_s', 'min_outage_s', 'stride_s')]
    if d > h or isinstance(ds['min_observed_history_steps'], bool) or not isinstance(ds['min_observed_history_steps'], int) or not 0 <= ds['min_observed_history_steps'] <= w:
        raise ValueError('지속 시간 또는 최소 관측 구간 수가 유효하지 않습니다.')
    episodes = manifest['episodes']
    rows = {key: [] for key in SPLITS}
    labels = {key: [] for key in SPLITS}
    count = np.zeros(len(FEATURES), np.int64)
    total = np.zeros(len(FEATURES), np.float64)
    squared = np.zeros(len(FEATURES), np.float64)
    train_ids = []
    for epno, ep in enumerate(episodes):
        path = folder / ep['path']
        if sha256_file(path / 'episode.json') != ep['episode_json_sha256']:
            raise ValueError('episode.json이 수집 후 바뀌었습니다.')
        for name, sha in ep['files_sha256'].items():
            if sha256_file(path / name) != sha:
                raise ValueError('수집된 배열의 해시가 다릅니다.')
        with np.load(path / 'observations.npz', allow_pickle=False) as obs, np.load(path / 'truth.npz', allow_pickle=False) as truth:
            if list(obs['feature_names']) != FEATURES or not np.array_equal(obs['time_end_s'], truth['time_end_s']):
                raise ValueError('입력 스키마 또는 시각이 일치하지 않습니다.')
            x, mask = obs['features'], obs['feature_mask']
            index, y = window_labels(truth['link_available'], x[..., 5] > 0, w, h, d, stride, ds['min_observed_history_steps'])
            if len(index):
                rows[ep['split']].append(np.column_stack([np.full(len(index), epno), index]).astype(np.int32))
                labels[ep['split']].append(y)
            if ep['split'] == 'train':
                train_ids.append(ep['id'])
                # 학습 지역/실행의 실제 입력으로 쓰이는 시각만 통계에 넣는다. val/test는 읽어도 적합하지 않는다.
                last_input = int(index[-1, 0]) + 1 if len(index) else 0
                v = x[:last_input].astype(np.float64)
                m = mask[:last_input]
                count += m.sum(axis=(0, 1, 2))
                total += np.where(m, v, 0).sum(axis=(0, 1, 2))
                squared += np.where(m, v*v, 0).sum(axis=(0, 1, 2))
    summaries = {}
    for split in SPLITS:
        index = np.concatenate(rows[split]) if rows[split] else np.zeros((0, 4), np.int32)
        y = np.concatenate(labels[split]) if labels[split] else np.zeros(0, np.uint8)
        if not len(index):
            raise ValueError(f'{split}에 학습 창이 없습니다. 실행 길이·관측 조건을 확인하십시오.')
        atomic_npz(destination / (split + '_windows.npz'), index=index, y=y)
        summaries[split] = {'samples': len(y), 'positive': int(y.sum()), 'negative': int(len(y)-y.sum()),
                            'positive_fraction': float(y.mean()),
                            'episodes': sum(ep['split'] == split for ep in episodes)}
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(0, squared/np.maximum(count, 1) - mean*mean))
    std[std < 1e-8] = 1.
    normalizer = {'feature_names': FEATURES, 'mean': mean.tolist(), 'std': std.tolist(),
                  'valid_count': count.tolist(), 'fit_split': 'train', 'fit_episode_ids': train_ids,
                  'fit_regions': sorted({ep['region'] for ep in episodes if ep['split'] == 'train'}),
                  'fit_geometry_sha256': sorted({manifest['maps'][ep['region']]['geometry_sha256']
                                                for ep in episodes if ep['split'] == 'train'}),
                  'scope': '학습 실행의 마지막 유효 입력 시각까지. 결측값·대각선은 통계에서 제외.'}
    write_json(destination / 'normalizer.json', normalizer)
    dataset = {'format': 'forecast-windows-v1', 'source_collection_sha256': sha256_file(folder / 'collection_manifest.json'),
               'collection_path': Path(__import__('os').path.relpath(folder / 'collection_manifest.json', destination)).as_posix(),
               'window_settings': ds,
               'feature_names': FEATURES, 'lookback_steps': w, 'horizon_steps': h,
               'minimum_outage_steps': d, 'stride_steps': stride, 'radio_dt_s': dt,
               'target': '향후 horizon_s 안에 min_outage_s 이상 연속으로 양방향 가용 링크 기준을 충족하지 못함',
               'target_includes_already_disconnected_links': True,
               'threshold_origin': '가용 홉 확률 0.9 및 미래 구간/지속 시간은 과제 설계값이며 논문의 보편 합격점이 아님.',
               'splits': summaries, 'model_trained': False,
               'files_sha256': {name: sha256_file(destination / name) for name in
                                ('train_windows.npz', 'val_windows.npz', 'test_windows.npz', 'normalizer.json')}}
    write_json(destination / 'dataset_manifest.json', dataset)
    return dataset
