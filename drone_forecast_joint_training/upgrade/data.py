"""지역 분리를 유지하는 과거 창과 학습 전용 전처리입니다.

truth는 정답/사후 평가 집단을 정할 때만 읽습니다. 모형 입력은 observations뿐입니다.
겹치는 시간 창을 무작위로 train/test에 나누지 않습니다.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from swarm_sim.forecast_data import window_labels
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from upgrade.map_setup import digest

DATA = ROOT / 'datasets' / 'upgrade45'
PREPARED = DATA / 'prepared'


def collection():
    manifest = read_json(DATA / 'collection_manifest.json')
    if not manifest['complete'] or manifest['failures']:
        raise ValueError('45개 지역의 수집을 먼저 완료하십시오.')
    protocol = read_json(ROOT / 'configs' / 'upgrade_protocol.json')
    if manifest['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('수집과 지도 분할 설정이 다릅니다.')
    if len(manifest['records']) != 270:
        raise ValueError('45개 지역 × 2방향 × 3간섭 조건 = 270개 전체 실행이 필요합니다.')
    return manifest, protocol


def checked_observations(record):
    folder = ROOT / record['path']
    meta = record['metadata']
    for name in ('observations.npz', 'truth.npz'):
        if sha256_file(folder / name) != meta['files_sha256'][name]:
            raise ValueError(f'{record["id"]}: {name} 해시가 다릅니다.')
    with np.load(folder / 'observations.npz', allow_pickle=False) as f:
        x, m = f['features'], f['mask']
    if x.shape != m.shape or not np.isfinite(x).all() or np.any(x[~m] != 0):
        raise ValueError('입력 크기, 유한성, 결측값 표현을 확인하십시오.')
    return x, m


def index_episode(record, protocol):
    """입력 끝 k의 다음 k+1..k+10에서 2연속 불가용 여부를 정답으로 만듭니다."""
    path = PREPARED / 'index' / (record['id'] + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if str(f['protocol_sha256']) != protocol['protocol_sha256']:
                raise ValueError('이전 인덱스의 설정이 다릅니다.')
            return f['index'], f['y'], f['current_available']
    x, _ = checked_observations(record)
    with np.load(ROOT / record['path'] / 'truth.npz', allow_pickle=False) as f:
        available = f['link_available']
    dt = protocol['radio_dt_s']
    w, h, persist, stride = [int(protocol[k]/dt) for k in ('lookback_s', 'horizon_s', 'min_outage_s', 'stride_s')]
    index, y = window_labels(available, x[..., 5] > 0, w, h, persist, stride,
                             protocol['minimum_observed_tx_intervals'])
    index = np.asarray(index, np.int32)
    current = available[tuple(index.T)]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, index=index, y=y.astype(np.uint8), current_available=current,
                        protocol_sha256=protocol['protocol_sha256'])
    return index, y, current


def build_indices():
    manifest, protocol = collection()
    rows = []
    for record in manifest['records']:
        index, y, current = index_episode(record, protocol)
        rows.append({'id': record['id'], 'region': record['region'], 'split': record['split'],
                     'all_observed_windows': len(y), 'currently_available_windows': int(current.sum()),
                     'future_outage_windows': int(y[current].sum()),
                     'positive_fraction': float(y[current].mean()) if current.any() else None,
                     'duration_s': record['metadata']['duration_s'],
                     'mission_completed': record['metadata']['mission'].get('mission_completed')})
    write_json(PREPARED / 'index_summary.json', {'protocol_sha256': protocol['protocol_sha256'], 'episodes': rows})
    return rows


def fit_normalizer(records, name):
    """결측을 제외한 학습 관측에서만 평균·표준편차를 계산합니다."""
    path = PREPARED / ('normalizer_' + name + '.json')
    identity = digest([(r['id'], r['metadata']['files_sha256']['observations.npz']) for r in records])
    if path.exists():
        item = read_json(path)
        if item['input_identity'] != identity:
            raise ValueError('전처리에 사용한 자료가 바뀌었습니다.')
        return item
    total, square, count = np.zeros(28), np.zeros(28), np.zeros(28, np.int64)
    for record in records:
        if record['split'] != 'train':
            raise ValueError('전처리에 학습 외 지역을 사용하지 마십시오.')
        x, m = checked_observations(record)
        flat, valid = x.reshape(-1, 28), m.reshape(-1, 28)
        for k in range(28):
            values = flat[valid[:, k], k].astype(np.float64)
            total[k] += values.sum()
            square[k] += np.dot(values, values)
            count[k] += len(values)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(square/np.maximum(count, 1)-mean**2, 0))
    std[std < 1e-6] = 1
    item = {'input_identity': identity, 'training_regions': sorted({r['region'] for r in records}),
            'episodes': len(records), 'mean': mean.tolist(), 'std': std.tolist(), 'observed_counts': count.tolist(),
            'clipped_standardized_absolute_value': 8, 'feature_names': records[0]['metadata']['feature_names']}
    write_json(path, item)
    return item


def select_bank(records, protocol, per_region, seed, current_only=True):
    """각 지역에서 동일 최대 개수를 정답과 무관하게 추출합니다. 양성 과표집은 없습니다."""
    rng = np.random.default_rng(seed)
    selections = []
    for region in sorted({r['region'] for r in records}):
        candidates = []
        for ep, record in enumerate(records):
            if record['region'] != region:
                continue
            index, y, up = index_episode(record, protocol)
            keep = np.flatnonzero(up) if current_only else np.arange(len(y))
            if len(keep):
                candidates.append(np.column_stack((np.full(len(keep), ep), index[keep], y[keep])))
        if not candidates:
            raise ValueError(f'{region}: 사용할 과거 관측이 없습니다.')
        candidates = np.concatenate(candidates).astype(np.int32)
        chosen = rng.choice(len(candidates), min(per_region, len(candidates)), replace=False)
        selections.append(candidates[chosen])
    return np.concatenate(selections)


def materialize_bank(name, records, protocol, per_region, seed, current_only=True):
    folder = PREPARED / name
    identity = digest({'ids': [r['id'] for r in records], 'protocol': protocol['protocol_sha256'],
                       'per_region': per_region, 'seed': seed, 'current_only': current_only})
    if (folder / 'bank.json').exists():
        meta = read_json(folder / 'bank.json')
        if meta['identity'] != identity:
            raise ValueError(f'{name}: 이전 표본 설정과 다릅니다.')
        return meta
    selected = select_bank(records, protocol, per_region, seed, current_only)
    folder.mkdir(parents=True, exist_ok=True)
    n, w = len(selected), int(protocol['lookback_s']/protocol['radio_dt_s'])
    x_out = np.lib.format.open_memmap(folder/'values.npy', mode='w+', dtype=np.float32, shape=(n, w, 28))
    m_out = np.lib.format.open_memmap(folder/'mask.npy', mode='w+', dtype=bool, shape=(n, w, 28))
    for ep in np.unique(selected[:, 0]):
        x, m = checked_observations(records[int(ep)])
        rows = np.flatnonzero(selected[:, 0] == ep)
        end, src, dst = selected[rows, 1:4].T
        t = end[:, None] + np.arange(-w+1, 1)
        x_out[rows] = x[t, src[:, None], dst[:, None]]
        m_out[rows] = m[t, src[:, None], dst[:, None]]
    x_out.flush(); m_out.flush()
    np.save(folder / 'selected.npy', selected)
    meta = {'identity': identity, 'name': name, 'samples': n, 'lookback_steps': w, 'features': 28,
            'positive_fraction': float(selected[:, 4].mean()), 'regions': sorted({r['region'] for r in records}),
            'record_ids': [r['id'] for r in records], 'records': records,
            'sampling': 'uniform without replacement per region; label independent; current availability cohort' if current_only
                         else 'uniform without replacement per region over observed history; labels unused in pretraining',
            'current_only': current_only, 'per_region_max': per_region, 'seed': seed}
    write_json(folder / 'bank.json', meta)
    return meta


def prepare():
    manifest, protocol = collection()
    build_indices()
    records = manifest['records']
    training = [r for r in records if r['split'] == 'train']
    original = [r for r in training if r['region'] in ('teheran', 'gangnam', 'hongdae')]
    validation = [r for r in records if r['split'] == 'val']
    fit_normalizer(training, 'train30')
    fit_normalizer(original, 'train3')
    banks = {}
    for name, part, count, seed, current in (
        ('train30', training, 2048, 1401, True),
        ('train3', original, 2048, 1401, True),
        ('val7', validation, 4096, 1402, True),
        ('pretrain30', training, 2048, 1403, False),
    ):
        banks[name] = materialize_bank(name, part, protocol, count, seed, current)
        print(json.dumps({'stage': 'bank', 'name': name, 'samples': banks[name]['samples'],
                          'positive_fraction': banks[name]['positive_fraction']}, ensure_ascii=False), flush=True)
    write_json(PREPARED / 'prepared.json', {'protocol_sha256': protocol['protocol_sha256'],
               'banks': {k: {q: v[q] for q in ('identity', 'samples', 'positive_fraction', 'regions')} for k, v in banks.items()},
               'test_predictions_computed': False})
    return banks


class Bank:
    def __init__(self, name, normalizer, features=28):
        self.meta = read_json(PREPARED / name / 'bank.json')
        self.values = np.load(PREPARED / name / 'values.npy', mmap_mode='r')
        self.mask = np.load(PREPARED / name / 'mask.npy', mmap_mode='r')
        self.selected = np.load(PREPARED / name / 'selected.npy', mmap_mode='r')
        self.features = features
        self.mean = np.asarray(normalizer['mean'][:features], np.float32)
        self.std = np.asarray(normalizer['std'][:features], np.float32)
        self.region = np.asarray([self.meta['records'][int(ep)]['region'] for ep in self.selected[:, 0]])

    def __len__(self):
        return len(self.selected)

    def batch(self, index):
        import torch
        x = self.values[index, :, :self.features].copy()
        mask = self.mask[index, :, :self.features].copy()
        x = np.where(mask, np.clip((x-self.mean)/self.std, -8, 8), 0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.as_tensor(self.selected[index, 4].astype(np.float32))
