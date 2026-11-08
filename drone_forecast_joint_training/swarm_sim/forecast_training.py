"""모델을 새로 만들지 않고 같은 객체를 이어 학습하는 프레임워크 독립 어댑터."""
import numpy as np


class TrainingSession:
    """model.fit_batch(x, mask, y), model.predict_batch(x, mask)를 연결한다.

    딥러닝 구조·손실·최적화기는 선택된 모델 쪽에서 구현한다.
    이 클래스는 가중치나 optimizer를 초기화하지 않는다.
    """
    def __init__(self, model):
        self.model = model
        self.context = None
        self.epochs_completed = 0

    def _check(self, dataset):
        context = (tuple(dataset.meta['feature_names']), dataset.lookback,
                   dataset.normalize, dataset.normalizer_sha256)
        if self.context is None:
            self.context = context
        elif context != self.context:
            raise ValueError('이어 학습할 입력 스키마·과거 창·표준화 기준이 달라졌습니다. 이전 normalizer를 명시해 고정하거나 새 모델로 비교하십시오.')

    def fit(self, dataset, epochs=1, batch_size=64, seed=0):
        if dataset.split != 'train':
            raise ValueError('val/test는 fit에 사용할 수 없습니다.')
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError('epochs는 양의 정수여야 합니다.')
        self._check(dataset)
        history = []
        for _ in range(epochs):
            total = 0.
            count = 0
            for batch in dataset.iter_batches(batch_size, shuffle=True, seed=seed+self.epochs_completed):
                # 같은 model 객체와 그 객체가 보관한 optimizer를 계속 사용한다.
                loss = float(self.model.fit_batch(batch['x'], batch['mask'], batch['y']))
                if not np.isfinite(loss):
                    raise ValueError('모델이 유한하지 않은 손실을 반환했습니다.')
                total += loss * len(batch['y'])
                count += len(batch['y'])
            self.epochs_completed += 1
            history.append({'epoch': self.epochs_completed, 'samples': count,
                            'mean_loss': total/max(1, count)})
        return history

    def predict(self, dataset, batch_size=64):
        self._check(dataset)
        predictions = []
        for batch in dataset.iter_batches(batch_size):
            p = np.asarray(self.model.predict_batch(batch['x'], batch['mask']), dtype=float).reshape(-1)
            if p.shape != batch['y'].shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('predict_batch는 [배치] 크기의 0~1 확률을 반환해야 합니다.')
            predictions.append(p)
        return np.concatenate(predictions), dataset.y.copy()
