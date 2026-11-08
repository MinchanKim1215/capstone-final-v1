"""검증셋에서 선택을 끝낸 후 처음 보는 지역을 평가합니다.

공통 표본에서 모든 비교군을 평가하고, 선정 모형은 전체 시험 비행에서도 평가합니다.
중첩 시간 창을 독립 표본처럼 취급한 신뢰구간 대신 공간 집단 단위 비교를 씁니다.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from swarm_sim.forecast_evaluation import binary_metrics
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from upgrade.data import Bank, collection, index_episode, materialize_bank, checked_observations, PREPARED
from upgrade.model import LinkTransformer
from upgrade.train import RUNS, VARIANTS, configure, macro_ap, predict_bank
from upgrade.map_setup import digest

RESULTS = ROOT / 'results' / 'upgrade'
EVALUATION_VERSION = 'v2_float32_representable_threshold'


def threshold_at_fpr(y, p, budget=.05):
    """검증 음성 중 경고 허용 개수 이하가 되도록 정합니다. 동점도 보수적으로 처리합니다."""
    scores = np.asarray(p)
    precision = np.float32 if scores.dtype == np.float32 else np.float64
    negative = np.sort(scores.astype(np.float64)[np.asarray(y) == 0])
    if not len(negative):
        raise ValueError('음성 검증 표본이 필요합니다.')
    allowed = int(math.floor(budget*len(negative)))
    boundary = len(negative)-allowed-1
    # float32 점수와 비교할 때 Python float가 다시 float32로 반올림되어
    # 경계 동점을 포함하는 일이 없도록 같은 저장 정밀도에서 다음 값을 택합니다.
    return float(np.nextafter(precision(negative[boundary]), precision(np.inf))) if boundary >= 0 else 0.


def load_run(folder):
    metadata = read_json(folder/'run.json')
    if sha256_file(folder/'best.pt') != metadata['best_checkpoint_sha256']:
        raise ValueError('선택 모형의 체크포인트가 바뀌었습니다.')
    state = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    model = LinkTransformer(**state['spec'])
    model.load_state_dict(state['state_dict']); model.eval()
    return model, state, metadata


def freeze_selection():
    plan = read_json(ROOT/'configs'/'upgrade_training.json')
    entries = {k: read_json(RUNS/f'{k}_s11'/'run.json') for k in VARIANTS}
    if any(not v['complete'] or v['plan_sha256'] != plan['plan_sha256'] for v in entries.values()):
        raise ValueError('모든 비교군의 학습/검증을 먼저 완료하십시오.')
    selected = max(entries, key=lambda k: entries[k]['validation_macro_ap'])
    item = {'plan_sha256': plan['plan_sha256'], 'selected_variant': selected, 'representative_seed': 11,
            'replication_seeds': plan['training']['replication_seeds'],
            'criterion': plan['training']['variant_selection'], 'test_used_for_selection': False,
            'validation_macro_ap': {k: v['validation_macro_ap'] for k, v in entries.items()},
            'checkpoint_sha256': {k: v['best_checkpoint_sha256'] for k, v in entries.items()},
            'test_classification_sampling': {'per_region_max': 8192, 'seed': 1404, 'label_independent': True},
            'dense_evaluation': 'all eligible windows in all 48 test flights, five ablations and two selected-variant replications',
            'ensemble': False}
    item['selection_sha256'] = digest(item)
    path = RESULTS/'selection.json'
    if path.exists() and read_json(path) != item:
        raise ValueError('동결한 선택을 결과 확인 후 바꾸지 마십시오.')
    write_json(path, item)
    return item


def grouped(y, p, regions, threshold):
    by = {str(k): binary_metrics(y[regions == k], p[regions == k], threshold) for k in sorted(set(regions.tolist()))}
    valid = [v['average_precision'] for v in by.values() if v['average_precision'] is not None]
    return {'all': binary_metrics(y, p, threshold), 'by_region': by,
            'macro_region_ap': float(np.mean(valid)) if valid else None,
            'macro_regions_with_both_classes': len(valid)}


def slice_metrics(bank, y, p, threshold):
    result = {}
    kinds = np.asarray([bank.meta['records'][int(ep)]['metadata']['kind'] for ep in bank.selected[:, 0]])
    # 움직이는 구간과 정지/대기 구간을 함께 공개하여 긴 대기 시간의 효과를 숨기지 않습니다.
    speed = np.linalg.norm(bank.values[:, :, :2], axis=-1)
    moving = np.max(speed, axis=1) > .25
    peer_known = bank.mask[:, -1, 20]
    for label, take in [('moving_history', moving), ('stationary_history', ~moving),
                        ('peer_geometry_available', peer_known), ('peer_geometry_missing', ~peer_known)]:
        result[label] = binary_metrics(y[take], p[take], threshold)
    result['interference'] = {k: binary_metrics(y[kinds == k], p[kinds == k], threshold) for k in sorted(set(kinds.tolist()))}
    times = (bank.selected[:, 1]+1)*.5
    starts = np.asarray([bank.meta['records'][int(ep)]['metadata']['jammer_window_s'][0] for ep in bank.selected[:, 0]])
    onset = (kinds == 'intermittent') & (times < starts) & (starts <= times+5)
    result['unannounced_jammer_start_within_horizon'] = binary_metrics(y[onset], p[onset], threshold)
    result['other_times'] = binary_metrics(y[~onset], p[~onset], threshold)
    return result


def bootstrap_differences(results, protocol, baseline='A0_three_regions_s11', repeats=2000):
    base = results[baseline]['test']['by_region']
    regions = sorted(base)
    group_of = {m['key']: m['spatial_group'] for m in protocol['maps']}
    groups = sorted({group_of[k] for k in regions})
    rng = np.random.default_rng(8841)
    draws = rng.choice(groups, (repeats, len(groups)), replace=True)
    output = {}
    for name, result in results.items():
        if name == baseline:
            continue
        delta = {k: result['test']['by_region'][k]['average_precision']-base[k]['average_precision'] for k in regions
                 if base[k]['average_precision'] is not None and result['test']['by_region'][k]['average_precision'] is not None}
        estimates = []
        for draw in draws:
            values = [delta[k] for g in draw for k in regions if group_of[k] == g and k in delta]
            if values:
                estimates.append(np.mean(values))
        output[name] = {'baseline': baseline, 'macro_ap_difference': float(np.mean(list(delta.values()))),
                        'cluster_bootstrap_95_interval': np.percentile(estimates, [2.5, 97.5]).tolist(),
                        'spatial_groups': len(groups), 'repeats': repeats,
                        'limit': 'Six held-out spatial groups; descriptive uncertainty, not proof of universal superiority.'}
    return output


def evaluate_banks():
    selection = read_json(RESULTS/'selection.json')
    manifest, protocol = collection()
    selected = selection['selected_variant']
    folders = [RUNS/f'{k}_s11' for k in VARIANTS]
    folders += [RUNS/f'{selected}_s{s}' for s in selection['replication_seeds']]
    if any(not (f/'run.json').exists() for f in folders):
        raise ValueError('선정 모델의 추가 초기화 시드 학습을 먼저 완료하십시오.')
    test_records = [r for r in manifest['records'] if r['split'] == 'test']
    sample = selection['test_classification_sampling']
    materialize_bank('test8', test_records, protocol, sample['per_region_max'], sample['seed'], True)
    configure(11)
    results = {}
    for folder in folders:
        model, state, meta = load_run(folder)
        normalizer, features = state['normalizer'], state['spec']['features']
        val, test = Bank('val7', normalizer, features), Bank('test8', normalizer, features)
        val_p = predict_bank(model, val)
        threshold = threshold_at_fpr(val.selected[:, 4], val_p)
        before = time.perf_counter()
        test_p = predict_bank(model, test)
        elapsed = time.perf_counter()-before
        np.save(folder/'test_bank_probability.npy', test_p)
        output = {'run': folder.name, 'threshold': threshold, 'threshold_source': 'validation bank negative-score quantile, FPR<=5%',
                  'validation': grouped(val.selected[:, 4], val_p, val.region, threshold),
                  'test': grouped(test.selected[:, 4], test_p, test.region, threshold),
                  'test_slices': slice_metrics(test, test.selected[:, 4], test_p, threshold),
                  'sample_inference_elapsed_s_including_batch_loading': elapsed,
                  'test_samples_per_s': len(test)/elapsed, 'model_parameters': meta['parameters'],
                  'checkpoint_sha256': meta['best_checkpoint_sha256']}
        write_json(folder/'evaluation.json', output)
        results[folder.name] = output
        print(json.dumps({'stage': 'test_bank', 'run': folder.name, 'macro_ap': output['test']['macro_region_ap'],
                          'ap': output['test']['all']['average_precision'], 'fpr': output['test']['all']['false_positive_rate']}, ensure_ascii=False), flush=True)
    # 점수만 내는 가장 단순한 관측 기준입니다. 추가 딥러닝 모델을 도입하는 것이 아닙니다.
    reference = Bank('test8', read_json(PREPARED/'normalizer_train30.json'))
    validation = Bank('val7', read_json(PREPARED/'normalizer_train30.json'))
    baselines = {}
    for name, feature, sign in [('latest_ack_sinr', 9, -1), ('estimated_loss_change', 26, 1)]:
        def scores(bank):
            raw = sign*bank.values[:, -1, feature].copy()
            valid = bank.mask[:, -1, feature]
            # 결측은 위험 점수의 중립값으로 두고 값이 있다는 표시를 억지로 예측에 넣지 않습니다.
            raw[~valid] = sign*float(read_json(PREPARED/'normalizer_train30.json')['mean'][feature])
            # 같은 단조 변환으로 0~1 점수로 바꿉니다. 확률 보정값이라는 주장은 하지 않습니다.
            return 1/(1+np.exp(-np.clip(raw/10, -40, 40)))
        vp, tp = scores(validation), scores(reference)
        cut = threshold_at_fpr(validation.selected[:, 4], vp)
        baselines[name] = {'threshold': cut, 'test': grouped(reference.selected[:, 4], tp, reference.region, cut)}
    output = {'evaluation_version': EVALUATION_VERSION, 'selection': selection, 'classification_bank': read_json(PREPARED/'test8'/'bank.json') | {'records': None},
              'results': results, 'simple_observation_baselines': baselines,
              'paired_cluster_bootstrap': bootstrap_differences(results, protocol),
              'new_protocol_baseline_not_directly_comparable_to_legacy_ap_012': True,
              'causal_claim': 'Only prediction inputs available by observation time. No repositioning improvement measured.'}
    write_json(RESULTS/'comparison.json', output)
    return output


def event_rows(record, index, probability, threshold, protocol):
    """관측 불가능한 단절도 총 사건 수에 남깁니다. 양방향 관측 주체는 구분합니다."""
    with np.load(ROOT/record['path']/'truth.npz', allow_pickle=False) as f:
        available = f['link_available']
    dt = protocol['radio_dt_s']
    w, horizon, persistence = [int(protocol[k]/dt) for k in ('lookback_s', 'horizon_s', 'min_outage_s')]
    rows = []
    n = available.shape[1]
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            pair = (index[:, 1] == src) & (index[:, 2] == dst)
            anchors, scores = index[pair, 0], probability[pair].astype(np.float64)
            bad = ~available[:, src, dst]
            boundary = np.diff(np.r_[False, bad, False].astype(np.int8))
            for start, stop in zip(np.flatnonzero(boundary == 1), np.flatnonzero(boundary == -1)):
                # 이미 기록 시작에 끊긴 상태 및 과거 창이 생기기 전 사건은 새로운 장애가 아닙니다.
                if stop-start < persistence or start*dt < protocol['lookback_s'] or start+persistence > len(bad):
                    continue
                eligible = (anchors < start) & (start+persistence-1 <= anchors+horizon)
                leads = start*dt-(anchors[eligible]+1)*dt
                alerts = scores[eligible] >= threshold
                detected = leads[alerts]
                lead = float(np.max(detected)) if len(detected) else None
                rows.append({'region': record['region'], 'episode': record['id'], 'source': src, 'destination': dst,
                             'start_s': start*dt, 'duration_s': (stop-start)*dt,
                             'observable_warning_window': bool(np.any(eligible)),
                             'possible_3s_warning': bool(np.any(leads >= 3)),
                             'detected': bool(len(detected)), 'lead_time_s': lead,
                             'detected_1s': bool(len(detected) and lead >= 1), 'detected_3s': bool(len(detected) and lead >= 3)})
    return rows


def summarize_events(rows):
    visible = [r for r in rows if r['observable_warning_window']]
    leads = [r['lead_time_s'] for r in visible if r['detected']]
    denominator = max(len(visible), 1)
    return {'total_new_directed_outage_events_after_lookback': len(rows), 'observable_events': len(visible),
            'without_observed_warning_window': len(rows)-len(visible),
            'observability_fraction': len(visible)/len(rows) if rows else None,
            'event_recall': sum(r['detected'] for r in visible)/denominator,
            'event_recall_at_least_1s': sum(r['detected_1s'] for r in visible)/denominator,
            'event_recall_at_least_3s': sum(r['detected_3s'] for r in visible)/denominator,
            'all_event_recall_at_least_3s': sum(r['detected_3s'] for r in visible)/max(len(rows), 1),
            'maximum_possible_observable_3s_recall': sum(r['possible_3s_warning'] for r in visible)/denominator,
            'detected_event_lead_median_s': float(np.median(leads)) if leads else None,
            'detected_event_lead_p10_s': float(np.percentile(leads, 10)) if leads else None,
            'counting_unit': 'directed observer-link outage; two directions are not statistically independent'}


def evaluate_dense(run_name=None):
    selection = read_json(RESULTS/'selection.json')
    variant = selection['selected_variant']
    run_name = run_name or f'{variant}_s11'
    allowed = [f'{k}_s11' for k in VARIANTS] + [f'{variant}_s{s}' for s in selection['replication_seeds']]
    if run_name not in allowed:
        raise ValueError('동결한 비교군에 없는 실행입니다.')
    folder = RUNS/run_name
    bank_eval = read_json(folder/'evaluation.json')
    configure(11)
    model, state, _ = load_run(folder)
    manifest, protocol = collection()
    records = [r for r in manifest['records'] if r['split'] == 'test']
    mean, std = [np.asarray(state['normalizer'][k][:state['spec']['features']], np.float32) for k in ('mean', 'std')]
    f, w, threshold = state['spec']['features'], state['spec']['length'], bank_eval['threshold']
    output_folder = RESULTS/'dense'/run_name
    output_folder.mkdir(parents=True, exist_ok=True)
    all_y, all_p, all_regions, episodes, events = [], [], [], [], []
    started = time.perf_counter()
    for record in records:
        result_path = output_folder/(record['id']+'.npz')
        index, y, up = index_episode(record, protocol)
        index, y = index[up], y[up]
        if result_path.exists():
            with np.load(result_path, allow_pickle=False) as f_data:
                if str(f_data['checkpoint_sha256']) != bank_eval['checkpoint_sha256']:
                    raise ValueError('전체 평가에 다른 모형의 예측이 섞였습니다.')
                p = f_data['probability']
                np.testing.assert_array_equal(index, f_data['index'])
        else:
            x, mask = checked_observations(record)
            p = np.empty(len(index), np.float32)
            with torch.inference_mode():
                for first in range(0, len(index), 512):
                    part = index[first:first+512]
                    end, src, dst = part.T
                    t = end[:, None]+np.arange(-w+1, 1)
                    observed = mask[t, src[:, None], dst[:, None], :f]
                    values = x[t, src[:, None], dst[:, None], :f]
                    values = np.where(observed, np.clip((values-mean)/std, -8, 8), 0).astype(np.float32)
                    p[first:first+len(part)] = torch.sigmoid(model(torch.from_numpy(values), torch.from_numpy(observed))).numpy()
            np.savez_compressed(result_path, index=index, y=y, probability=p,
                                checkpoint_sha256=bank_eval['checkpoint_sha256'])
        episode_events = event_rows(record, index, p, threshold, protocol)
        events.extend(episode_events)
        episodes.append({'id': record['id'], 'region': record['region'], 'kind': record['metadata']['kind'],
                         'metrics': binary_metrics(y, p, threshold), 'events': summarize_events(episode_events)})
        all_y.append(y); all_p.append(p); all_regions.append(np.full(len(y), record['region']))
        print(json.dumps({'stage': 'dense_test', 'episode': record['id'], 'windows': len(y),
                          'completed': len(episodes), 'total': len(records), 'elapsed_s': round(time.perf_counter()-started, 1)}), flush=True)
    result = {'evaluation_version': EVALUATION_VERSION, 'run': folder.name, 'threshold': threshold, 'episodes': len(records), 'complete': True,
              'checkpoint_sha256': bank_eval['checkpoint_sha256'],
              'classification': grouped(np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_regions), threshold),
              'events': summarize_events(events), 'by_episode': episodes,
              'elapsed_s': time.perf_counter()-started, 'no_repositioning_test': True}
    write_json(output_folder/'summary.json', result)
    write_json(output_folder/'events.json', events)
    return result


def evaluate_all_dense():
    selection = read_json(RESULTS/'selection.json')
    names = [f'{k}_s11' for k in VARIANTS]
    names += [f'{selection["selected_variant"]}_s{s}' for s in selection['replication_seeds']]
    results = {}
    for name in names:
        path = RESULTS/'dense'/name/'summary.json'
        if path.exists():
            item = read_json(path)
            expected = read_json(RUNS/name/'run.json')['best_checkpoint_sha256']
            if not item['complete'] or item['checkpoint_sha256'] != expected:
                raise ValueError('전체 평가 요약의 체크포인트가 다릅니다.')
            new_threshold = read_json(RUNS/name/'evaluation.json')['threshold']
            if item.get('evaluation_version') != EVALUATION_VERSION or item['threshold'] != new_threshold:
                item = evaluate_dense(name)
        else:
            item = evaluate_dense(name)
        results[name] = item
    protocol = read_json(ROOT/'configs'/'upgrade_protocol.json')
    boot_input = {k: {'test': v['classification']} for k, v in results.items()}
    result = {'evaluation_version': EVALUATION_VERSION, 'complete': True, 'selection': selection, 'results': results,
              'paired_cluster_bootstrap': bootstrap_differences(boot_input, protocol),
              'primary_evaluation': 'all 1-second stride eligible windows, all 48 held-out flights; no window sampling',
              'test_used_for_tuning': False, 'no_repositioning_test': True}
    write_json(RESULTS/'dense_comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'banks', 'dense', 'dense-all'])
    args = parser.parse_args()
    result = {'freeze': freeze_selection, 'banks': evaluate_banks, 'dense': evaluate_dense, 'dense-all': evaluate_all_dense}[args.stage]()
    if args.stage == 'freeze':
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
