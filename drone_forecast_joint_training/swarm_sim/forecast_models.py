"""같은 [배치, 시간, 관측값+마스크]를 받는 세 예측기.

아래 크기는 프로젝트의 비교 시작 설정이다. 원 논문 전체 재현이나
통신 데이터에서의 최적 구조를 주장하지 않는다.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn


MODEL_NAMES = ('gru', 'tcn', 'transformer')
DEFAULT_SPECS = {
    'gru': {'hidden_size': 64, 'dropout': 0.1},
    'tcn': {'channels': 32, 'kernel_size': 3, 'dilations': [1, 2, 4], 'dropout': 0.1},
    'transformer': {'d_model': 32, 'nhead': 4, 'layers': 2,
                    'feedforward': 64, 'dropout': 0.1},
}


def packed_input(values, mask):
    values, mask = np.asarray(values, np.float32), np.asarray(mask, bool)
    if values.ndim != 3 or values.shape != mask.shape:
        raise ValueError('입력과 마스크는 같은 [배치, 시간, 항목] 크기여야 합니다.')
    if not np.isfinite(values[mask]).all():
        raise ValueError('실제 관측값에 NaN 또는 무한대가 있습니다.')
    return np.concatenate([np.where(mask, values, 0.), mask.astype(np.float32)], axis=-1)


class GRUPredictor(nn.Module):
    def __init__(self, input_size, lookback, hidden_size=64, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        _, state = self.rnn(x)
        return self.head(state[-1]).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size-1)*dilation
        self.conv1 = nn.Conv1d(inputs, outputs, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(outputs, outputs, kernel_size, padding=pad, dilation=dilation)
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 오른쪽 패딩 출력을 제거해 각 시점이 자기보다 미래의 입력을 보지 못하게 한다.
        size = x.shape[-1]
        z = self.dropout(torch.relu(self.conv1(x)[..., :size]))
        z = self.dropout(torch.relu(self.conv2(z)[..., :size]))
        return torch.relu(z+self.skip(x))


class TCNPredictor(nn.Module):
    def __init__(self, input_size, lookback, channels=32, kernel_size=3,
                 dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        receptive_field = 1+2*(kernel_size-1)*sum(dilations)
        if receptive_field < lookback:
            raise ValueError(f'TCN 수용 영역 {receptive_field}가 입력 길이 {lookback}보다 작습니다.')
        self.receptive_field = receptive_field
        self.blocks = nn.Sequential(*[
            CausalBlock(input_size if k == 0 else channels, channels, kernel_size, d, dropout)
            for k, d in enumerate(dilations)
        ])
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        z = self.blocks(x.transpose(1, 2))
        return self.head(z[..., -1]).squeeze(-1)


class TransformerPredictor(nn.Module):
    def __init__(self, input_size, lookback, d_model=32, nhead=4, layers=2,
                 feedforward=64, dropout=0.1):
        super().__init__()
        if d_model % nhead:
            raise ValueError('Transformer d_model은 nhead의 배수여야 합니다.')
        self.lookback = lookback
        self.project = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.normal_(self.position, std=0.02)
        # 층별 독립 초기화. TST의 사전학습·BatchNorm·전체 시점 결합 구조와는 구분한다.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, feedforward, dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.shape[1] != self.lookback:
            raise ValueError('학습할 때와 다른 길이의 시간 창입니다.')
        z = self.project(x)+self.position
        # 창 전체가 현재까지의 과거이다. 과거 창 내부의 양방향 attention은 미래 정답 누출이 아니다.
        for layer in self.layers:
            z = layer(z)
        return self.head(self.norm(z[:, -1])).squeeze(-1)


def make_model(name, input_size, lookback, spec=None):
    if name not in MODEL_NAMES:
        raise ValueError(f'모델은 {MODEL_NAMES} 중 하나입니다.')
    options = deepcopy(DEFAULT_SPECS[name])
    if spec:
        unknown = set(spec)-set(options)
        if unknown:
            raise ValueError(f'지원하지 않는 {name} 설정: {sorted(unknown)}')
        options.update(spec)
    for key, value in options.items():
        if key == 'dropout':
            if not isinstance(value, (float, int)) or not 0 <= value < 1:
                raise ValueError('dropout은 0 이상 1 미만이어야 합니다.')
        elif key == 'dilations':
            if not value or any(not isinstance(v, int) or v < 1 for v in value):
                raise ValueError('dilations는 양의 정수 목록입니다.')
        elif not isinstance(value, int) or value < 1:
            raise ValueError(f'{key}는 양의 정수입니다.')
    constructor = {'gru': GRUPredictor, 'tcn': TCNPredictor,
                   'transformer': TransformerPredictor}[name]
    return constructor(input_size, lookback, **options), options


class TorchForecastAdapter:
    """기존 TrainingSession에서도 사용할 수 있는 학습/추론 어댑터."""
    def __init__(self, model, device='cpu', learning_rate=1e-3, weight_decay=1e-4,
                 gradient_clip=1.0):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                            weight_decay=weight_decay)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.gradient_clip = gradient_clip

    def fit_packed(self, x, y):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        tx = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        ty = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        loss = self.loss_function(self.model(tx), ty)
        if not torch.isfinite(loss):
            raise ValueError('손실이 유한하지 않습니다.')
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def fit_batch(self, x, mask, y):
        return self.fit_packed(packed_input(x, mask), y)

    @torch.inference_mode()
    def predict_packed(self, x):
        self.model.eval()
        tx = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return torch.sigmoid(self.model(tx)).cpu().numpy()

    def predict_batch(self, x, mask):
        return self.predict_packed(packed_input(x, mask))


class ForecastPredictor:
    """저장된 학습 전용 표준화 기준을 포함한, 향후 제어부용 공통 추론 API."""
    def __init__(self, checkpoint, device='cpu'):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload.get('format') != 'forecast-checkpoint-v1':
            raise ValueError('지원하지 않는 체크포인트입니다.')
        self.contract = payload['contract']
        self.model, _ = make_model(payload['model_name'], self.contract['input_size'],
                                    self.contract['lookback_steps'], payload['model_spec'])
        self.model.load_state_dict(payload['model_state'])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.mean = np.asarray(self.contract['normalizer']['mean'], np.float32)
        self.std = np.asarray(self.contract['normalizer']['std'], np.float32)
        self.threshold = float(payload['threshold'])

    @torch.inference_mode()
    def predict(self, values, mask, feature_names=None):
        """원래 단위의 [배치, 과거 시점, 12항목]과 마스크를 받는다."""
        values, mask = np.asarray(values, np.float32), np.asarray(mask, bool)
        wanted = (self.contract['lookback_steps'], len(self.contract['feature_names']))
        if values.ndim != 3 or values.shape[1:] != wanted or mask.shape != values.shape:
            raise ValueError(f'입력은 [배치, {wanted[0]}, {wanted[1]}]이어야 합니다.')
        if feature_names is not None and list(feature_names) != self.contract['feature_names']:
            raise ValueError('입력 항목의 순서가 학습 때와 다릅니다.')
        z = packed_input((values-self.mean)/self.std, mask)
        risk = torch.sigmoid(self.model(torch.as_tensor(z, device=self.device))).cpu().numpy()
        return {'risk': risk, 'alert': risk >= self.threshold,
                'horizon_s': self.contract['horizon_s'],
                'minimum_outage_s': self.contract['minimum_outage_s'],
                'observed_tx_intervals': np.sum(mask[..., 5] & (values[..., 5] > 0), axis=1)}
