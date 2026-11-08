"""향후 재배치 제어기가 사용할 예측 전용 인터페이스입니다.

현재는 위험 점수까지만 반환하며 이동 명령을 생성하지 않습니다.
truth/미래 궤적/실제 간섭원 정보는 이 인터페이스의 인자가 아닙니다.
"""
from collections import deque
from pathlib import Path

import numpy as np
import torch

from swarm_sim.regions import ROOT, read_json
from upgrade.collect import FEATURES
from upgrade.evaluate import load_run


class OnlineLinkPredictor:
    def __init__(self, run_folder=None, threads=2):
        torch.set_num_threads(threads)
        if run_folder is None:
            selected = read_json(ROOT/'results'/'upgrade'/'selection.json')
            run_folder = ROOT/'results'/'upgrade'/'runs'/f'{selected["selected_variant"]}_s11'
        self.folder = Path(run_folder)
        self.model, state, _ = load_run(self.folder)
        self.length, self.features = state['spec']['length'], state['spec']['features']
        self.mean, self.std = [np.asarray(state['normalizer'][k][:self.features], np.float32) for k in ('mean', 'std')]
        self.threshold = read_json(self.folder/'evaluation.json')['threshold']
        self.values, self.masks = deque(maxlen=self.length), deque(maxlen=self.length)
        self.previous_time = None
        self.dt = .5

    def reset(self):
        """비행 또는 지역이 바뀔 때 과거 창을 반드시 비웁니다."""
        self.values.clear(); self.masks.clear(); self.previous_time = None

    @torch.inference_mode()
    def predict_window(self, values, mask):
        """[과거 40구간, 드론 수, 드론 수, 28항목] → 방향별 위험도 행렬.

        반환 NaN은 아직 관측 창/송신 기록이 없다는 뜻이며 안전하다는 뜻이 아닙니다.
        오프라인 평가의 '현재 실제 가용' 조건은 여기서 요구하지 않습니다.
        """
        values, mask = np.asarray(values, np.float32), np.asarray(mask, bool)
        if values.shape != mask.shape or values.ndim != 4 or values.shape[0] != self.length or values.shape[-1] != 28:
            raise ValueError('입력은 [40, N, N, 28] 값과 같은 크기의 관측 마스크여야 합니다.')
        if values.shape[1] != values.shape[2] or not np.isfinite(values).all():
            raise ValueError('드론 수/유한값을 확인하십시오.')
        n = values.shape[1]
        eligible = np.any(values[..., 5] > 0, axis=0) & ~np.eye(n, dtype=bool)
        src, dst = np.where(eligible)
        risk = np.full((n, n), np.nan, np.float32)
        if len(src):
            x = values[:, src, dst, :self.features].transpose(1, 0, 2)
            observed = mask[:, src, dst, :self.features].transpose(1, 0, 2)
            x = np.where(observed, np.clip((x-self.mean)/self.std, -8, 8), 0).astype(np.float32)
            risk[src, dst] = torch.sigmoid(self.model(torch.from_numpy(x), torch.from_numpy(observed))).numpy()
        return {'risk': risk, 'warning': eligible & (risk.astype(float) >= self.threshold), 'observed_history': eligible,
                'threshold': self.threshold, 'horizon_s': 5, 'minimum_outage_s': 1}

    def observe(self, time_end_s, values, mask):
        """0.5초마다 현재까지의 관측 한 장을 추가합니다. 20초가 쌓인 후 예측합니다."""
        if self.previous_time is not None and not np.isclose(time_end_s-self.previous_time, self.dt, rtol=0, atol=1e-6):
            raise ValueError('관측 간격이 바뀌거나 시간이 건너뛰었습니다. 새 비행이면 reset을 호출하십시오.')
        if not np.isfinite(time_end_s):
            raise ValueError('시간이 유한해야 합니다.')
        self.previous_time = float(time_end_s)
        self.values.append(np.asarray(values).copy()); self.masks.append(np.asarray(mask).copy())
        return None if len(self.values) < self.length else self.predict_window(np.stack(self.values), np.stack(self.masks))
