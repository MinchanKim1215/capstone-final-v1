"""잠재 채널의 '현재 가용' 정답으로 추론을 거르지 않는 추가 운용 진단입니다.

주 평가의 사전 단절 예측 집단은 유지합니다. 이 파일은 학습/선택을 바꾸지 않고
실행 때 실제로 관측할 수 있는 최근 ACK 기준에서도 같은 모형을 확인합니다.
"""
import json
import time

import numpy as np
import torch

from swarm_sim.forecast_evaluation import binary_metrics
from swarm_sim.regions import ROOT, read_json, write_json
from upgrade.data import collection, index_episode, checked_observations
from upgrade.evaluate import RESULTS, load_run, grouped
from upgrade.inference import OnlineLinkPredictor
from upgrade.train import RUNS, configure


def run():
    selection = read_json(RESULTS/'selection.json')
    name = f'{selection["selected_variant"]}_s11'
    folder = RUNS/name
    metrics = read_json(folder/'evaluation.json')
    configure(11)
    model, state, _ = load_run(folder)
    features, window = state['spec']['features'], state['spec']['length']
    mean, std = [np.asarray(state['normalizer'][k][:features], np.float32) for k in ('mean', 'std')]
    manifest, protocol = collection()
    records = [r for r in manifest['records'] if r['split'] == 'test']
    out = RESULTS/'operational'
    out.mkdir(parents=True, exist_ok=True)
    ys, ps, groups, fresh, currently = [], [], [], [], []
    checked = False
    for record in records:
        index, y, current = index_episode(record, protocol)
        x, mask = checked_observations(record)
        k, i, j = index.T
        recent_ack = mask[k, i, j, 10] & (x[k, i, j, 10] <= 1.)
        path = out/(record['id']+'.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                if str(f['checkpoint_sha256']) != metrics['checkpoint_sha256']:
                    raise ValueError('다른 모형의 운용 진단 파일입니다.')
                p = f['probability']
        else:
            p = np.empty(len(index), np.float32)
            with np.load(RESULTS/'dense'/name/(record['id']+'.npz'), allow_pickle=False) as f:
                np.testing.assert_array_equal(f['index'], index[current])
                p[current] = f['probability']
            missing = np.flatnonzero(~current)
            with torch.inference_mode():
                for first in range(0, len(missing), 512):
                    rows = missing[first:first+512]
                    end, src, dst = index[rows].T
                    t = end[:, None]+np.arange(-window+1, 1)
                    observed = mask[t, src[:, None], dst[:, None], :features]
                    values = x[t, src[:, None], dst[:, None], :features]
                    values = np.where(observed, np.clip((values-mean)/std, -8, 8), 0).astype(np.float32)
                    p[rows] = torch.sigmoid(model(torch.from_numpy(values), torch.from_numpy(observed))).numpy()
            np.savez_compressed(path, probability=p, index=index, y=y, recent_ack=recent_ack,
                                checkpoint_sha256=metrics['checkpoint_sha256'])
        if not checked:
            # 전체 기록에서 만든 창과 후속 제어기용 API 출력이 같은지 확인합니다.
            predictor = OnlineLinkPredictor(folder)
            end = int(index[min(100, len(index)-1), 0])
            answer = predictor.predict_window(x[end-window+1:end+1], mask[end-window+1:end+1])
            rows = np.flatnonzero(index[:, 0] == end)
            np.testing.assert_allclose(answer['risk'][index[rows, 1], index[rows, 2]], p[rows], atol=2e-6, rtol=1e-5)
            warm = time.perf_counter()
            for _ in range(20):
                predictor.predict_window(x[end-window+1:end+1], mask[end-window+1:end+1])
            latency = (time.perf_counter()-warm)/20*1000
            checked = True
        ys.append(y); ps.append(p); groups.append(np.full(len(y), record['region']))
        fresh.append(recent_ack); currently.append(current)
    y, p, region = np.concatenate(ys), np.concatenate(ps), np.concatenate(groups)
    recent, up = np.concatenate(fresh), np.concatenate(currently)
    threshold = metrics['threshold']
    result = {'complete': True, 'run': name, 'checkpoint_sha256': metrics['checkpoint_sha256'],
              'training_or_model_selection_changed': False, 'threshold': threshold,
              'all_observed_pairs': grouped(y, p, region, threshold),
              'recent_ack_within_1s': grouped(y[recent], p[recent], region[recent], threshold),
              'currently_truly_available_reference': grouped(y[up], p[up], region[up], threshold),
              'recent_ack_fraction': float(recent.mean()), 'recent_ack_but_latently_unavailable_fraction': float(np.mean(~up[recent])),
              'online_interface_matches_offline_window_predictions': checked,
              'cpu_matrix_prediction_ms': latency, 'cpu_threads': 2,
              'latency_evaluated_pair_count': int(answer['observed_history'].sum()),
              'latency_scope': 'one 40 x 8 x 8 x 28 window to risk matrix; excludes map-feature extraction and simulator',
              'interpretation': 'All-pair/ACK-gated labels may include already ongoing outages. Their AP is not the primary new-outage forecasting AP.'}
    write_json(out/'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    run()
