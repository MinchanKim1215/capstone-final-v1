"""GRU/LSTM/Transformer 등에서 공통으로 쓸 수 있는 NumPy 시간 창 로더.

진실값 파일은 읽지 않는다. 원래 물리량과 결측 마스크를 보존하며,
표준화 통계는 학습 분할에서 적합한 normalizer.json만 사용한다.
"""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .forecast import FEATURES
from .regions import read_json, sha256_file


class ForecastDataset:
    def __init__(self, folder, split='train', normalize=True, cache_episodes=2, normalizer_file=None):
        self.folder = Path(folder).resolve()
        if split not in ('train', 'val', 'test'):
            raise ValueError('split은 train, val, test 중 하나입니다.')
        self.split = split
        self.normalize = normalize
        self.cache_episodes = max(1, int(cache_episodes))
        self.cache = OrderedDict()
        self.meta = read_json(self.folder / 'dataset_manifest.json')
        collection_path = (self.folder / self.meta.get('collection_path', 'collection_manifest.json')).resolve()
        self.collection_folder = collection_path.parent
        if self.meta['source_collection_sha256'] != sha256_file(collection_path):
            raise ValueError('데이터셋 생성 후 수집 manifest가 바뀌었습니다. build를 다시 실행하십시오.')
        self.collection = read_json(collection_path)
        for name, sha in self.meta['files_sha256'].items():
            if sha256_file(self.folder / name) != sha:
                raise ValueError('데이터셋 인덱스 또는 표준화 파일이 바뀌었습니다.')
        if self.meta['feature_names'] != FEATURES:
            raise ValueError('입력 피처 스키마가 현재 코드와 다릅니다.')
        with np.load(self.folder / (split + '_windows.npz'), allow_pickle=False) as z:
            self.index = z['index']
            self.y = z['y']
        if self.index.shape != (len(self.y), 4):
            raise ValueError('시간 창 인덱스 크기 오류')
        self.lookback = int(self.meta['lookback_steps'])
        self.horizon = int(self.meta['horizon_steps'])
        norm_path = Path(normalizer_file).resolve() if normalizer_file else self.folder / 'normalizer.json'
        normalizer = read_json(norm_path)
        self.normalizer_sha256 = sha256_file(norm_path)
        train_ids = {x['id'] for x in self.collection['episodes'] if x['split'] == 'train'}
        if normalizer['fit_split'] != 'train' or (normalizer_file is None and set(normalizer['fit_episode_ids']) != train_ids):
            raise ValueError('표준화 통계가 학습 실행에서만 적합된 것이 아닙니다.')
        if normalizer_file is not None:
            test_regions = self.collection['config']['splits']['test']['regions']
            test_geometry = {self.collection['maps'][key]['geometry_sha256'] for key in test_regions}
            if (set(normalizer.get('fit_regions', [])) & set(test_regions)
                or set(normalizer.get('fit_geometry_sha256', [])) & test_geometry):
                raise ValueError('고정 표준화 통계가 이번 시험 지역을 학습에 사용했습니다.')
            if not normalizer.get('fit_regions') or not normalizer.get('fit_geometry_sha256'):
                raise ValueError('이전 표준화 파일의 학습 지역 출처가 없어 누출 여부를 확인할 수 없습니다.')
        if normalizer['feature_names'] != FEATURES:
            raise ValueError('표준화 피처 순서 오류')
        self.mean = np.asarray(normalizer['mean'], np.float32)
        self.std = np.asarray(normalizer['std'], np.float32)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('표준화 통계 오류')
        for epno in np.unique(self.index[:, 0]):
            if not 0 <= epno < len(self.collection['episodes']) or self.collection['episodes'][epno]['split'] != split:
                raise ValueError('분할 경계를 넘는 실행 인덱스입니다.')

    def __len__(self):
        return len(self.y)

    def _episode(self, number):
        if number not in self.cache:
            ep = self.collection['episodes'][number]
            path = self.collection_folder / ep['path'] / 'observations.npz'
            if sha256_file(path) != ep['files_sha256']['observations.npz']:
                raise ValueError('학습 입력 파일의 해시가 다릅니다.')
            with np.load(path, allow_pickle=False) as z:
                if list(z['feature_names']) != FEATURES:
                    raise ValueError('실행의 피처 스키마 오류')
                self.cache[number] = (z['features'], z['feature_mask'], z['time_end_s'])
            while len(self.cache) > self.cache_episodes:
                self.cache.popitem(last=False)
        self.cache.move_to_end(number)
        return self.cache[number]

    def __getitem__(self, item):
        epno, end, source, destination = map(int, self.index[item])
        x, mask, times = self._episode(epno)
        if not self.lookback-1 <= end < len(times)-self.horizon or source == destination:
            raise ValueError('과거 창 또는 미래 정답 범위를 벗어난 인덱스입니다.')
        sl = slice(end-self.lookback+1, end+1)
        values = x[sl, source, destination].copy()
        valid = mask[sl, source, destination].copy()
        if self.normalize:
            values = (values-self.mean) / self.std
        # 결측값을 정상적인 0 측정으로 오해하지 않도록 마스크를 항상 함께 반환한다.
        values[~valid] = 0.
        ep = self.collection['episodes'][epno]
        return {'x': values.astype(np.float32), 'mask': valid, 'y': np.float32(self.y[item]),
                'episode_id': ep['id'], 'region': ep['region'], 'source': source,
                'destination': destination, 'input_end_s': float(times[end]),
                'label_start_s': float(times[end]),
                'label_end_s': float(times[end]+self.horizon*self.meta['radio_dt_s'])}

    def iter_batches(self, batch_size=64, shuffle=False, seed=0):
        if batch_size < 1:
            raise ValueError('batch_size는 양수여야 합니다.')
        order = np.arange(len(self))
        if shuffle:
            rng = np.random.default_rng(seed)
            blocks = []
            for epno in np.unique(self.index[:, 0]):
                selected = order[self.index[:, 0] == epno]
                rng.shuffle(selected)
                width = max(1024, batch_size * 8)
                blocks.extend(selected[k:k+width] for k in range(0, len(selected), width))
            rng.shuffle(blocks)
            order = np.concatenate(blocks)
        for start in range(0, len(order), batch_size):
            samples = [self[int(i)] for i in order[start:start+batch_size]]
            yield {'x': np.stack([s['x'] for s in samples]),
                   'mask': np.stack([s['mask'] for s in samples]),
                   'y': np.asarray([s['y'] for s in samples], np.float32),
                   'metadata': [{k:v for k,v in s.items() if k not in ('x', 'mask', 'y')} for s in samples]}


def model_input(batch):
    """선택 사항: 관측값과 결측 마스크를 피처 축으로 붙인다. [B,T,2F]."""
    return np.concatenate([batch['x'], batch['mask'].astype(np.float32)], axis=-1)
