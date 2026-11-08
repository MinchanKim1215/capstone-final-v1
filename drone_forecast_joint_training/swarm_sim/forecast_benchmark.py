"""세 모델의 공통 학습 조건·검증 전용 선정·독립 시험과 결과 저장."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import random
import time

import numpy as np
import sklearn
import torch

from .forecast_dataset import ForecastDataset
from .forecast_models import MODEL_NAMES, make_model, TorchForecastAdapter, packed_input
from .forecast_evaluation import evaluation_context, grouped_metrics, validation_threshold, event_metrics
from .regions import read_json, write_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FIELDS = {'format','dataset','output','stage','seeds','data_seed','device','threads',
    'max_epochs','patience','batch_size','learning_rate','weight_decay','gradient_clip',
    'max_train_windows','max_val_windows','models'}


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                           separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    if set(cfg) != CONFIG_FIELDS or cfg['format'] != 'forecast-benchmark-v1':
        raise ValueError(f'설정 필드를 확인하십시오: {sorted(CONFIG_FIELDS)}')
    if cfg['stage'] not in ('smoke','pilot','final'):
        raise ValueError('stage는 smoke/pilot/final입니다.')
    for key in ('data_seed','threads','max_epochs','patience','batch_size'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == 'data_seed' else 1):
            raise ValueError(f'{key} 설정 오류')
    for key in ('max_train_windows','max_val_windows'):
        if cfg[key] is not None and (type(cfg[key]) is not int or cfg[key] < 2):
            raise ValueError(f'{key}는 null 또는 2 이상의 정수입니다.')
        if cfg['stage'] != 'smoke' and cfg[key] is not None:
            raise ValueError('pilot/final에서는 일부 창만 뽑는 제한을 사용할 수 없습니다.')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']) or any(type(s) is not int or s < 0 for s in cfg['seeds']):
        raise ValueError('seeds는 중복 없는 0 이상 정수 목록입니다.')
    for key in ('learning_rate','gradient_clip'):
        if not isinstance(cfg[key], (int, float)) or not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key}는 양수입니다.')
    if not isinstance(cfg['weight_decay'], (int, float)) or not np.isfinite(cfg['weight_decay']) or cfg['weight_decay'] < 0:
        raise ValueError('weight_decay는 0 이상입니다.')
    if set(cfg['models']) != set(MODEL_NAMES):
        raise ValueError('models에는 gru/tcn/transformer 설정이 모두 필요합니다.')
    folder = (path.parent/cfg['dataset']).resolve()
    output = (path.parent/cfg['output']).resolve()
    if not (folder/'dataset_manifest.json').is_file():
        raise ValueError(f'예측 데이터셋이 없습니다: {folder}\n먼저 prediction.py prepare를 실행하십시오.')
    manifest = read_json(folder/'dataset_manifest.json')
    for model in MODEL_NAMES:
        make_model(model, len(manifest['feature_names'])*2,
                   manifest['lookback_steps'], cfg['models'][model])
    return cfg, folder, output


def choose_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA가 없습니다. 설정의 device를 cpu로 바꾸십시오.')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS가 없습니다. 설정의 device를 cpu로 바꾸십시오.')
    if name not in ('cpu','cuda','mps'):
        raise ValueError('device는 cpu/cuda/mps/auto입니다.')
    return name


def seed_everything(seed, threads):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_fingerprints():
    names = ['forecast_models.py','forecast_evaluation.py','forecast_benchmark.py',
             'forecast_reports.py','forecast_dataset.py']
    return {name:sha256_file(ROOT/'swarm_sim'/name) for name in names}


def experiment_contract(cfg, folder):
    meta = read_json(folder/'dataset_manifest.json')
    return {'configuration':cfg, 'dataset_manifest_sha256':sha256_file(folder/'dataset_manifest.json'),
            'normalizer_sha256':sha256_file(folder/'normalizer.json'),
            'source_sha256':source_fingerprints(), 'target':meta['target']}


def load_arrays(folder, split, cap=None, data_seed=2026):
    """모든 모델에 같은 창을 제공한다. smoke에서만 고정 난수로 창 수를 줄인다."""
    data = ForecastDataset(folder, split, cache_episodes=4)
    positions = np.arange(len(data))
    if cap is not None and len(positions) > cap:
        positions = np.sort(np.random.default_rng(data_seed).choice(positions, cap, replace=False))
    x = np.empty((len(positions), data.lookback, len(data.meta['feature_names'])*2), np.float32)
    for j, i in enumerate(positions):
        sample = data[int(i)]
        x[j] = packed_input(sample['x'][None], sample['mask'][None])[0]
    return {'data':data, 'positions':positions, 'x':x, 'y':data.y[positions].astype(np.float32)}


def predict_batches(adapter, x, batch_size):
    return np.concatenate([adapter.predict_packed(x[i:i+batch_size])
                           for i in range(0, len(x), batch_size)])


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_contract(data):
    meta = data.meta
    return {'feature_names':meta['feature_names'], 'input_size':len(meta['feature_names'])*2,
            'lookback_steps':data.lookback, 'horizon_s':meta['window_settings']['horizon_s'],
            'minimum_outage_s':meta['window_settings']['min_outage_s'],
            'radio_dt_s':meta['radio_dt_s'],
            'normalizer':read_json(data.folder/'normalizer.json'),
            'normalizer_sha256':data.normalizer_sha256,
            'target_includes_already_disconnected_links':True}


def prediction_rows(arrays, probability, context, threshold):
    data = arrays['data']
    rows = []
    for j, position in enumerate(arrays['positions']):
        epno, end, src, dst = map(int, data.index[position])
        ep = data.collection['episodes'][epno]
        # 입력 끝은 데이터셋의 절대 시각이며 미래/현재 가용성 열은 사후 평가용이다.
        input_end_s = float(data._episode(epno)[2][end])
        rows.append({'episode_id':ep['id'], 'region':ep['region'], 'source':src, 'destination':dst,
            'input_end_s':input_end_s, 'future_outage_label':int(arrays['y'][j]),
            'risk':float(probability[j]), 'alert':int(probability[j] >= threshold),
            'current_available_evaluation_only':int(context.current_available[j])})
    return rows


def save_evaluation(run_folder, split, arrays, probability, context, threshold, extra=None):
    metrics = grouped_metrics(arrays['y'], probability, context, threshold)
    events, rows = event_metrics(context, probability, threshold)
    metrics.update({'split':split, 'threshold':threshold, 'threshold_fit_split':'val',
        'threshold_method':'검증셋 현재 가용 링크 F1 최대화',
        'event_metrics':events,
        'current_availability_is_evaluation_only':True,
        'all_labels_include_existing_outages':True,
        'samples_are_overlapping_windows_not_independent_flights':True})
    if extra:
        metrics.update(extra)
    write_json(run_folder/f'{split}_metrics.json', metrics)
    write_csv(run_folder/f'{split}_predictions.csv', prediction_rows(arrays, probability, context, threshold))
    write_csv(run_folder/f'{split}_events.csv', rows)
    return metrics


def measure_latency(adapter, x):
    sample = x[:1]
    for _ in range(5):
        adapter.predict_packed(sample)
    if adapter.device.type == 'cuda':
        torch.cuda.synchronize()
    measured = []
    for _ in range(30):
        start = time.perf_counter()
        adapter.predict_packed(sample)
        if adapter.device.type == 'cuda':
            torch.cuda.synchronize()
        measured.append((time.perf_counter()-start)*1000)
    return {'single_window_median_ms':float(np.median(measured)),
            'single_window_p95_ms':float(np.percentile(measured, 95)),
            'timed_repetitions':30, 'warmup_repetitions':5,
            'scope':'학습 실행 장비에서 adapter의 단일 창 추론. 실제 드론 장비 성능이 아님.'}


def train_one(cfg, folder, output, name, seed, train, val, context, resume=False):
    seed_everything(seed, cfg['threads'])
    device = choose_device(cfg['device'])
    contract = experiment_contract(cfg, folder)
    token = fingerprint(contract)
    run_folder = output/name/f'seed_{seed}'
    run_folder.mkdir(parents=True, exist_ok=True)
    status_path = run_folder/'run.json'
    if status_path.exists():
        old = read_json(status_path)
        if old['experiment_sha256'] != token:
            raise ValueError(f'이전 실행과 코드/설정/데이터가 다릅니다. 새 output을 사용하십시오: {run_folder}')
        if old.get('complete'):
            print(f'{name} seed={seed}: 완료된 동일 실행을 재사용합니다.', flush=True)
            return old
        if not resume:
            raise ValueError('중단된 실행이 있습니다. 같은 설정에 --resume을 추가하십시오.')
    model, spec = make_model(name, train['x'].shape[-1], train['data'].lookback, cfg['models'][name])
    adapter = TorchForecastAdapter(model, device, cfg['learning_rate'], cfg['weight_decay'], cfg['gradient_clip'])
    status = {'format':'forecast-run-v1', 'complete':False, 'stage':cfg['stage'],
        'model':name, 'seed':seed, 'experiment_sha256':token, 'contract':contract,
        'parameters':sum(p.numel() for p in model.parameters()), 'model_spec':spec,
        'training_windows':len(train['y']), 'validation_windows':len(val['y']),
        'training_episode_count':train['data'].meta['splits']['train']['episodes'],
        'validation_episode_count':val['data'].meta['splits']['val']['episodes'],
        'model_input_size':list(train['x'].shape[1:]), 'test_used_for_training_or_selection':False,
        'checkpoint_selection':'validation macro average precision on currently available links',
        'checkpoint_selection_is_not_a_composite_score':True,
        'device':device, 'versions':{'torch':str(torch.__version__), 'numpy':np.__version__,
            'scikit_learn':sklearn.__version__, 'python':platform.python_version(), 'platform':platform.platform()},
        'started_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(status_path, status)
    history, best_score, best_epoch, stale, start_epoch = [], -1., 0, 0, 0
    elapsed_before = 0.
    if resume and (run_folder/'last.pt').exists():
        saved = torch.load(run_folder/'last.pt', map_location='cpu', weights_only=True)
        if saved['experiment_sha256'] != token:
            raise ValueError('재개 파일의 실험 조건이 다릅니다.')
        model.load_state_dict(saved['model_state'])
        adapter.optimizer.load_state_dict(saved['optimizer_state'])
        for state in adapter.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        torch.set_rng_state(saved['torch_rng_state'])
        if device == 'cuda' and saved.get('cuda_rng_states'):
            torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        history, best_score, best_epoch = saved['history'], saved['best_score'], saved['best_epoch']
        stale, start_epoch = saved['stale'], saved['epoch']
        elapsed_before = saved['elapsed_training_s']
    started = time.perf_counter()
    for epoch in range(start_epoch, cfg['max_epochs']):
        if stale >= cfg['patience']:
            break
        epoch_start = time.perf_counter()
        order = np.random.default_rng(seed+epoch).permutation(len(train['y']))
        weighted_loss = 0.
        for first in range(0, len(order), cfg['batch_size']):
            selected = order[first:first+cfg['batch_size']]
            weighted_loss += adapter.fit_packed(train['x'][selected], train['y'][selected])*len(selected)
        probability = predict_batches(adapter, val['x'], cfg['batch_size'])
        scores = grouped_metrics(val['y'], probability, context, .5)
        score = scores['macro_available_region_ap']
        if score is None:
            raise ValueError('검증셋의 현재 가용 링크에 AP를 계산할 양성·음성이 없습니다.')
        history.append({'epoch':epoch+1, 'train_bce':weighted_loss/len(order),
            'val_bce':scores['all']['log_loss'], 'val_ap_all':scores['all']['average_precision'],
            'val_ap_current_available':scores['currently_available']['average_precision'],
            'val_macro_available_region_ap':score,
            'valid_regions_for_macro':scores['macro_available_region_ap_regions'],
            'epoch_s':time.perf_counter()-epoch_start})
        if score > best_score+1e-6:
            best_score, best_epoch, stale = score, epoch+1, 0
            atomic_torch_save(run_folder/'best.pt', {'format':'forecast-checkpoint-v1',
                'model_name':name, 'model_spec':spec, 'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'contract':checkpoint_contract(train['data']), 'threshold':.5,
                'threshold_fit_split':'val', 'seed':seed, 'epoch':best_epoch,
                'experiment_sha256':token, 'stage':cfg['stage']})
        else:
            stale += 1
        atomic_torch_save(run_folder/'last.pt', {'format':'forecast-resume-v1',
            'experiment_sha256':token, 'model_state':model.state_dict(),
            'optimizer_state':adapter.optimizer.state_dict(), 'torch_rng_state':torch.get_rng_state(),
            'cuda_rng_states':torch.cuda.get_rng_state_all() if device == 'cuda' else [],
            'history':history, 'best_score':best_score, 'best_epoch':best_epoch,
            'stale':stale, 'epoch':epoch+1,
            'elapsed_training_s':elapsed_before+time.perf_counter()-started})
        write_csv(run_folder/'history.csv', history)
        print(f'{name} seed={seed} epoch={epoch+1}: train_BCE={history[-1]["train_bce"]:.4f}, '
              f'val_macro_AP(currently_available)={score:.4f}', flush=True)
    best = torch.load(run_folder/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state'])
    probability = predict_batches(adapter, val['x'], cfg['batch_size'])
    threshold = validation_threshold(val['y'], probability, context.current_available)
    best['threshold'] = threshold
    atomic_torch_save(run_folder/'best.pt', best)
    scores = save_evaluation(run_folder, 'val', val, probability, context, threshold)
    status.update({'complete':True, 'best_epoch':best_epoch, 'epochs_completed':len(history),
        'best_val_macro_available_region_ap':best_score, 'threshold':threshold,
        'threshold_fit_split':'val', 'elapsed_training_s':elapsed_before+time.perf_counter()-started,
        'latency':measure_latency(adapter, val['x']), 'checkpoint_sha256':sha256_file(run_folder/'best.pt'),
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'repositioning_evaluated':False, 'predictor_accuracy_measured':True,
        'result_scope':'실행 확인용 일부 창' if cfg['stage'] == 'smoke' else '예측 전용, 지정 데이터/지역/시드 범위',
        'validation_metric_file':'val_metrics.json'})
    write_json(status_path, status)
    return status


def run_training(config_path, models=MODEL_NAMES, seeds=None, resume=False):
    cfg, folder, output = read_config(config_path)
    selected_seeds = cfg['seeds'] if seeds is None else seeds
    if any(s not in cfg['seeds'] for s in selected_seeds):
        raise ValueError('--seed는 설정 파일 seeds에 포함된 값이어야 합니다.')
    seed_everything(cfg['data_seed'], cfg['threads'])
    print('학습/검증 창을 불러옵니다. 시험셋으로 모델이나 경고 기준을 정하지 않습니다.', flush=True)
    train = load_arrays(folder, 'train', cfg['max_train_windows'], cfg['data_seed'])
    val = load_arrays(folder, 'val', cfg['max_val_windows'], cfg['data_seed'])
    if len(set(train['y'])) < 2:
        raise ValueError('학습 정답에 양성·음성이 모두 필요합니다.')
    context = evaluation_context(val['data'], val['positions'])
    results = []
    for name in models:
        for seed in selected_seeds:
            results.append(train_one(cfg, folder, output, name, seed, train, val, context, resume))
    from .forecast_reports import comparison_report
    comparison_report(config_path)
    return {'output':str(output), 'completed_runs':len(results), 'test_evaluated':False}


def run_test(config_path, models=MODEL_NAMES, seeds=None):
    cfg, folder, output = read_config(config_path)
    if cfg['stage'] == 'smoke':
        raise ValueError('smoke는 실행 확인 전용입니다. 시험에는 pilot 또는 final 설정을 사용하십시오.')
    selected_seeds = cfg['seeds'] if seeds is None else seeds
    if any(s not in cfg['seeds'] for s in selected_seeds):
        raise ValueError('--seed는 설정 파일에 포함된 값이어야 합니다.')
    seed_everything(cfg['data_seed'], cfg['threads'])
    token = fingerprint(experiment_contract(cfg, folder))
    runs = []
    # 시험 데이터를 읽기 전에 모든 요청 모델의 학습 완료와 설정을 확인한다.
    for name in models:
        for seed in selected_seeds:
            run_folder = output/name/f'seed_{seed}'
            status = read_json(run_folder/'run.json')
            if not status.get('complete') or status['experiment_sha256'] != token:
                raise ValueError(f'완료된 동일 조건 학습이 필요합니다: {run_folder}')
            if sha256_file(run_folder/'best.pt') != status['checkpoint_sha256']:
                raise ValueError('검증 후 체크포인트가 바뀌었습니다.')
            runs.append((name, seed, run_folder, status))
    arrays = load_arrays(folder, 'test')
    context = evaluation_context(arrays['data'], arrays['positions'])
    for name, seed, run_folder, status in runs:
        checkpoint = torch.load(run_folder/'best.pt', map_location='cpu', weights_only=True)
        model, _ = make_model(name, arrays['x'].shape[-1], arrays['data'].lookback, checkpoint['model_spec'])
        model.load_state_dict(checkpoint['model_state'])
        adapter = TorchForecastAdapter(model, choose_device(cfg['device']))
        probability = predict_batches(adapter, arrays['x'], cfg['batch_size'])
        save_evaluation(run_folder, 'test', arrays, probability, context, checkpoint['threshold'],
            {'checkpoint_sha256':status['checkpoint_sha256'],
             'experiment_sha256':token, 'test_did_not_change_model_or_threshold':True})
        print(f'{name} seed={seed}: 고정 모델·고정 경고 기준으로 시험을 완료했습니다.', flush=True)
    from .forecast_reports import comparison_report
    comparison_report(config_path)
    return {'output':str(output), 'evaluated_runs':len(runs), 'model_or_threshold_updated':False}
