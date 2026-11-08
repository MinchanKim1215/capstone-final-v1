"""Read-only event/corpus audit and spatially cross-fitted calibration diagnostics.

Writes only a separate operations folder. No rollout, GPU inference, predictor
retraining, controller change, or final-test access is performed.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

OPS = Path(__file__).resolve().parent
PROJECT = OPS.parent / 'drone_forecast_joint_training'
REPLAY = OPS / 'prediction_validation_20261029_v2'
JOB = OPS / 'prediction_diagnostics_20261029_v2'
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from solo.metrics import binary_metrics, events_from_links, alert_metrics
from solo.targets import prediction_targets

HEADS = ['risk_5s', 'onset_3to4s', 'onset_3to7s']
MODELS = ['pretrain2800', 'joint176', 'joint190', 'lightgbm']


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers and OneDrive can briefly open a file without delete sharing.
    # Keep atomic JSON replacement, but tolerate those short sharing violations.
    for attempt in range(12):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.05*(attempt+1), .5))


def status(stage, **extra):
    data = dict(state='running', stage=stage, pid=os.getpid(), updated_at=now(),
                cpu_threads=1, cuda_used=False, **extra)
    write(JOB / 'status.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def prepare():
    JOB.mkdir(exist_ok=True)
    previous = load(REPLAY / 'protocol.json')
    if load(REPLAY / 'status.json')['state'] != 'completed':
        raise ValueError('Completed common-observation replay is required')
    pinned = {}
    for relative, expected in previous['pinned_sha256'].items():
        path = OPS / 'validate_prediction.py' if relative == '@validation_script' else PROJECT / relative
        if sha(path) != expected:
            raise ValueError('Replay source input changed: ' + relative)
        pinned[str(path)] = expected
    for name in ['protocol.json', 'fusion.json', 'classification.json', 'alarm_crossfit.json']:
        pinned[str(REPLAY / name)] = sha(REPLAY / name)
    for ep in previous['episodes']:
        path = REPLAY / 'replay' / (ep['region'] + '.npz')
        meta = load(path.with_suffix('.json'))
        if meta['protocol_sha256'] != sha(REPLAY / 'protocol.json') or sha(path) != meta['npz_sha256']:
            raise ValueError('Replay cache changed: ' + ep['region'])
        pinned[str(path)] = meta['npz_sha256']
        pinned[str(path.with_suffix('.json'))] = sha(path.with_suffix('.json'))
    for name in ['results/transformer/bootstrap/complete.json', 'configs/transformer_plan.json',
                 'solo/metrics.py', 'solo/targets.py']:
        pinned[str(PROJECT / name)] = sha(PROJECT / name)
    pinned[str(Path(__file__).resolve())] = sha(__file__)
    protocol = dict(created_at=now(), pinned_sha256=pinned,
        development_targets=dict(first_alert_precision=.9, event_recall_with_minimum_3s_lead=.8,
                                 false_alerts_per_drone_hour=1., minimum_alerts=10),
        target_status='Development proposal; not professor-mandated, achieved, or operationally approved.',
        prediction_meaning='Observed outage outcomes under the recorded controller; not counterfactual no-reposition risk.',
        observations=previous['episodes'], models=MODELS, common_heads=HEADS,
        calibration='Positive-slope affine logit calibration, unweighted Bernoulli NLL; fit six maps, evaluate two, four existing spatial folds.',
        calibration_selection='One prespecified method versus unchanged probabilities; no hyperparameter or method selection on held-out maps.',
        timing='RF frames 0.5 s; stored predictions/labels every 3 s; eligible events and censoring exactly follow solo.metrics.',
        training_steps=0, simulation_runs=0, gpu_used=False, cpu_threads=1,
        final_test_accessed=False, writes_within_main_experiment=False,
        limitations=[
            'The eight diagnostic maps have already been used for checkpoint selection; cross-fitting calibration does not remove this earlier selection bias.',
            'Existing LightGBM and Transformer have unequal historical training corpora; this is a frozen-system diagnostic, not a controlled architecture experiment.',
            'A perfect stored-label oracle is a label/cadence diagnostic, not a theoretical upper bound for all predictors.',
            'Lead traces use the preceding stored decision; their actual lead can be up to 2.5 s longer than the requested nominal lead.',
            'Calibration is evaluated for three common heads independently. Nesting violations are counted; no calibrated model is deployed.',
            'No new calibrated alert thresholds or controller rollout are evaluated in this diagnostic.',
            'One low-priority CPU thread avoids GPU competition but can still add some CPU or disk contention.',
        ])
    path = JOB / 'protocol.json'
    if path.exists():
        saved = load(path)
        if saved['pinned_sha256'] != pinned:
            raise ValueError('Diagnostic inputs changed after protocol was frozen')
        return saved
    write(path, protocol)
    return protocol


def coverage(episode, head, horizon, support_name):
    links = episode['links']
    times = np.arange(len(episode['target'])) * 3.
    decisions = (times >= 21) & (times <= len(links) * .5 - 8)
    support = episode[support_name]
    result = dict(eligible_events=0, with_decision=0, with_support=0,
                  with_valid_current_up_label=0, with_positive_stored_label=0)
    for (relay, peer), starts in events_from_links(links).items():
        eligible = starts[(starts >= 24) & (starts <= len(links) * .5 - 8)]
        for onset in eligible:
            indices = np.flatnonzero(decisions & (times >= onset-horizon) & (times <= onset-3))
            result['eligible_events'] += 1
            result['with_decision'] += int(len(indices) > 0)
            available = indices[support[indices, relay, peer]]
            result['with_support'] += int(len(available) > 0)
            usable = available[episode['valid'][available, relay, peer, head]]
            result['with_valid_current_up_label'] += int(len(usable) > 0)
            result['with_positive_stored_label'] += int(episode['target'][usable, relay, peer, head].any())
    return result


def calibrator_fit(probability, target):
    p = np.asarray(probability, np.float64)
    y = np.asarray(target, np.float64)
    p = np.clip(p, 1e-12, 1-1e-12)
    x = np.log(p) - np.log1p(-p)
    if not (0 < y.sum() < len(y)):
        raise ValueError('Calibration fold must contain both classes')

    def objective(theta):
        a = np.exp(theta[0])
        z = a*x + theta[1]
        residual = expit(z)-y
        loss = np.mean(np.logaddexp(0., z)-y*z)
        gradient = np.array([np.mean(residual*a*x), np.mean(residual)])
        return float(loss), gradient

    identity_loss = objective(np.zeros(2))[0]
    fitted = minimize(objective, np.zeros(2), method='L-BFGS-B', jac=True,
                      bounds=[(np.log(.05), np.log(20.)), (-15., 15.)],
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    if not fitted.success or fitted.fun > identity_loss + 1e-8:
        raise ValueError('Calibration optimization did not converge: ' + str(fitted.message))
    return dict(slope=float(np.exp(fitted.x[0])), intercept=float(fitted.x[1]),
                training_samples=len(y), training_positives=int(y.sum()),
                identity_nll=float(identity_loss), fitted_nll=float(fitted.fun),
                optimization_success=bool(fitted.success))


def calibrator_apply(probability, parameters):
    p = np.clip(np.asarray(probability, np.float64), 1e-12, 1-1e-12)
    return expit(parameters['slope']*(np.log(p)-np.log1p(-p))+parameters['intercept'])


def metrics(y, p):
    result = binary_metrics(y, p)
    ordered = np.argsort(p, kind='stable')
    bins = []
    for indices in np.array_split(ordered, 10):
        if len(indices):
            bins.append(dict(count=len(indices), mean_probability=float(p[indices].mean()),
                             actual_fraction=float(y[indices].mean())))
    result['equal_count_calibration_bins'] = bins
    result['ece_equal_count_10'] = float(sum(b['count']*abs(b['mean_probability']-b['actual_fraction']) for b in bins)/len(y))
    result['mean_probability'] = float(p.mean())
    return result


def self_check():
    for onset, expected5 in [(30., 1), (31.5, 0)]:
        links = np.ones((120, 8, 8), bool)
        index = int(onset*2)
        links[index:index+2, 3, 0] = False
        y, valid = prediction_targets(links, 20)
        ep = dict(links=links, target=y, valid=valid, common_support=np.ones(y.shape[:3], bool))
        for head, horizon, expected in [(1, 5, expected5), (2, 8, 1)]:
            result = coverage(ep, head, horizon, 'common_support')
            assert result['eligible_events'] == 1 and result['with_decision'] == 1
            assert result['with_positive_stored_label'] == expected, result
            oracle = dict(links=links, prediction=y, prediction_supported=ep['common_support'] & valid[..., head])
            actual = alert_metrics([oracle], head, .5, .5, horizon)
            assert actual['eligible_events'] == 1 and actual['event_hits_3s'] == expected, actual
    p = np.array([.01, .02, .1, .2, .5, .7, .9, .99])
    parameters = calibrator_fit(p, np.array([0, 0, 0, 1, 0, 1, 1, 1]))
    calibrated = calibrator_apply(p, parameters)
    assert np.all(np.diff(calibrated) >= 0) and np.isfinite(calibrated).all()
    return dict(synthetic_event_alignment=True, synthetic_label_oracle=True,
                monotone_calibration=True, calibration_optimizer=True)


def corpus_audit():
    bootstrap = load(PROJECT / 'results/transformer/bootstrap/complete.json')
    plan = load(PROJECT / 'configs/transformer_plan.json')
    maps = {row['key']: row for row in plan['maps']}
    regions, combinations = set(), set()
    rows, summaries = [], {}
    for i, (prefix, expected) in enumerate(zip(bootstrap['train_paths'], bootstrap['train_hashes'])):
        path = (PROJECT / prefix).with_suffix('.npz')
        meta_path = path.with_suffix('.json')
        meta = load(meta_path)
        if maps[meta['region']]['joint_split'] != 'train':
            raise ValueError('Non-training geography in supervised corpus')
        if meta['array_sha256'] != expected or sha(path) != expected:
            raise ValueError('Supervised training archive changed: ' + prefix)
        with np.load(path, allow_pickle=False) as source:
            target = source['prediction_targets']
            valid = source['prediction_valid']
            support = source['prediction_supported']
            links = source['links']
        if i in [0, len(bootstrap['train_paths'])-1]:
            expected_target, expected_valid = prediction_targets(links, len(target))
            assert np.array_equal(target, expected_target) and np.array_equal(valid, expected_valid)
        usable = valid & support[..., None]
        samples = usable.sum(axis=(0, 1, 2)).astype(int).tolist()
        positives = (usable & target.astype(bool)).sum(axis=(0, 1, 2)).astype(int).tolist()
        link_events = sum(len(starts) for starts in events_from_links(links).values())
        row = dict(region=meta['region'], spatial_group=maps[meta['region']]['joint_spatial_group'],
                   kind=meta['kind'], direction=meta['direction'], seed=meta['seed'],
                   prefix=prefix, samples=samples, positives=positives,
                   persistent_directed_link_events=link_events, archive_sha256=expected,
                   metadata_sha256=sha(meta_path))
        rows.append(row)
        regions.add(meta['region'])
        combinations.add((meta['region'], meta['direction'], meta['kind']))
        for label in ['all', 'kind:'+meta['kind'], 'direction:'+meta['direction']]:
            group = summaries.setdefault(label, dict(episodes=0, samples=[0]*4, positives=[0]*4,
                                                      persistent_directed_link_events=0))
            group['episodes'] += 1
            group['samples'] = [a+b for a, b in zip(group['samples'], samples)]
            group['positives'] = [a+b for a, b in zip(group['positives'], positives)]
            group['persistent_directed_link_events'] += link_events
        if (i+1) % 32 == 0:
            status('supervised_corpus_audit', complete=i+1, total=len(bootstrap['train_paths']))
    train_regions = {row['key'] for row in plan['maps'] if row['joint_split'] == 'train'}
    result = dict(episodes=len(rows), represented_regions=len(regions), planned_train_regions=len(train_regions),
                  missing_train_regions=sorted(train_regions-regions),
                  represented_region_direction_interference_combinations=len(combinations),
                  planned_combinations_if_2_directions_3_conditions=len(train_regions)*2*3,
                  heads=HEADS+['onset_0to7s'], summaries=summaries, episodes_detail=rows,
                  scope='Only the fixed supervised bootstrap corpus; subsequent PPO flights are additional experience and are not counted here.',
                  missing_audits=['Raw RF precursors, beacon age, building corners and jammer transition attribution require an additional feature-level audit.'])
    write(JOB / 'corpus_audit.json', result)
    return result


def event_audit(episodes):
    aggregate, regional, traces = {}, [], {}
    for ep in episodes:
        spec = ep['spec']
        per_region = dict(region=spec['region'], kind=spec['kind'], direction=spec['direction'], horizons={})
        for label, head, horizon in [('5s', 1, 5), ('8s', 2, 8)]:
            row = {name: coverage(ep, head, horizon, name) for name in ['native_support', 'common_support']}
            prediction = np.where(ep['valid'], ep['target'], 0.)
            oracle = dict(prediction=prediction, prediction_supported=ep['common_support'] & ep['valid'][..., head], links=ep['links'])
            row['stored_label_oracle'] = alert_metrics([oracle], head, .5, .5, horizon)
            per_region['horizons'][label] = row
            for name in ['native_support', 'common_support']:
                sums = aggregate.setdefault(label, {}).setdefault(name, {k:0 for k in row[name]})
                for key, value in row[name].items():
                    sums[key] += value
        regional.append(per_region)
        times = np.arange(len(ep['target']))*3.
        duration = len(ep['links'])*.5
        for (relay, peer), starts in events_from_links(ep['links']).items():
            for onset in starts[(starts >= 24) & (starts <= duration-8)]:
                for nominal in [10., 8., 5., 3., 1.]:
                    k = int(np.floor((onset-nominal)/3.))
                    if k < 0 or times[k] < 21 or times[k] > duration-8:
                        continue
                    for model in MODELS:
                        for head in [0, 2]:
                            key = (model, HEADS[head], nominal)
                            bucket = traces.setdefault(key, dict(eligible=0, supported=0, scores=[], actual_leads=[]))
                            bucket['eligible'] += 1
                            p = float(ep[model][k, relay, peer, head])
                            if not ep['common_support'][k, relay, peer] or not np.isfinite(p):
                                continue
                            bucket['supported'] += 1
                            bucket['scores'].append(p)
                            bucket['actual_leads'].append(float(onset-times[k]))
    trace_rows = []
    for (model, head, nominal), bucket in traces.items():
        scores = np.asarray(bucket.pop('scores'))
        actual = np.asarray(bucket.pop('actual_leads'))
        trace_rows.append(dict(model=model, head=head, nominal_lead_s=nominal, **bucket,
                               probability_quantiles={str(q):float(np.quantile(scores, q)) for q in [.1, .5, .9]},
                               actual_lead_min_s=float(actual.min()), actual_lead_max_s=float(actual.max()),
                               actual_lead_mean_s=float(actual.mean())))
    oracle = {}
    for label in ['5s', '8s']:
        fields = ['first_alerts', 'correct_first_alerts', 'false_or_late_first_alerts', 'event_hits_3s', 'eligible_events', 'drone_hours']
        row = {key:sum(ep['horizons'][label]['stored_label_oracle'][key] for ep in regional) for key in fields}
        row.update(first_alert_precision=row['correct_first_alerts']/max(1, row['first_alerts']),
                   first_alert_event_recall_3s=row['event_hits_3s']/max(1, row['eligible_events']))
        oracle[label] = row
        assert row['eligible_events'] == aggregate[label]['common_support']['eligible_events']
    result = dict(coverage=aggregate, stored_label_oracle=oracle, per_region=regional,
                  event_conditioned_lead_trace=trace_rows,
                  interpretation='Only events are included in lead traces. High event-conditioned scores do not establish precision or absence of false alerts.',
                  oracle_warning='Future labels are used deliberately for this offline diagnostic only; this is not a usable predictor or a universal recall upper bound.')
    write(JOB / 'event_alignment.json', result)
    return result


def calibration_audit(episodes):
    previous_folds = load(REPLAY / 'fusion.json')['crossfit_folds']
    region_to_index = {ep['spec']['region']: i for i, ep in enumerate(episodes)}
    seen = []
    result = {}
    reference = load(REPLAY / 'classification.json')['common']
    for model in MODELS:
        calibrated = [np.full(ep[model][..., :3].shape, np.nan, np.float64) for ep in episodes]
        constants = [np.full(ep[model][..., :3].shape, np.nan, np.float64) for ep in episodes]
        folds = []
        for fold in previous_folds:
            train = [region_to_index[name] for name in fold['tune_regions']]
            test = [region_to_index[name] for name in fold['evaluation_regions']]
            assert not set(train) & set(test) and len(train) == 6 and len(test) == 2
            train_groups = {episodes[i]['spec']['spatial_group'] for i in train}
            test_groups = {episodes[i]['spec']['spatial_group'] for i in test}
            assert not train_groups & test_groups
            if model == MODELS[0]:
                seen.extend(test)
            parameters = {}
            for head, name in enumerate(HEADS):
                ys, ps = [], []
                for i in train:
                    keep = episodes[i]['common_support'] & episodes[i]['valid'][..., head]
                    ys.append(episodes[i]['target'][..., head][keep])
                    ps.append(episodes[i][model][..., head][keep])
                y, p = np.concatenate(ys), np.concatenate(ps)
                fitted = calibrator_fit(p, y)
                parameters[name] = fitted
                for i in test:
                    finite = np.isfinite(episodes[i][model][..., head])
                    calibrated[i][..., head][finite] = calibrator_apply(episodes[i][model][..., head][finite], fitted)
                    constants[i][..., head][finite] = float(y.mean())
            folds.append(dict(fold=fold['fold'], calibration_regions=fold['tune_regions'],
                              evaluation_regions=fold['evaluation_regions'], parameters=parameters))
            status('calibration_crossfit', model=model, fold=fold['fold']+1, total_folds=4)
        rows = {}
        for head, name in enumerate(HEADS):
            ys, raw, after, constant = [], [], [], []
            for i, ep in enumerate(episodes):
                keep = ep['common_support'] & ep['valid'][..., head]
                ys.append(ep['target'][..., head][keep])
                raw.append(ep[model][..., head][keep])
                after.append(calibrated[i][..., head][keep])
                constant.append(constants[i][..., head][keep])
            y, raw_p, new_p, prior = map(np.concatenate, [ys, raw, after, constant])
            before, after = metrics(y, raw_p), metrics(y, new_p)
            assert before['samples'] == reference[model][name]['samples']
            assert before['positives'] == reference[model][name]['positives']
            assert abs(before['average_precision']-reference[model][name]['average_precision']) < 1e-10
            rows[name] = dict(before=before, spatial_crossfit_after=after,
                              crossfit_prevalence_constant_brier=float(np.mean((prior-y)**2)),
                              brier_change=after['brier']-before['brier'],
                              ece_change=after['ece_equal_count_10']-before['ece_equal_count_10'])
        nesting = dict(samples=0, raw_violations=0, calibrated_violations=0)
        for ep, predicted in zip(episodes, calibrated):
            keep = ep['common_support'] & ep['valid'][..., :3].all(-1)
            raw = ep[model][..., :3][keep]
            new = predicted[keep]
            nesting['samples'] += len(raw)
            nesting['raw_violations'] += int((raw[:, 1] > np.minimum(raw[:, 0], raw[:, 2])+1e-7).sum())
            nesting['calibrated_violations'] += int((new[:, 1] > np.minimum(new[:, 0], new[:, 2])+1e-7).sum())
        result[model] = dict(heads=rows, folds=folds, common_head_nesting=nesting)
        write(JOB / 'calibration_crossfit.json', result)
    assert sorted(seen) == list(range(len(episodes)))
    return result


def report(protocol, corpus, events, calibration):
    lines = ['**예측 개선 1~6단계: 사건·학습 자료·확률 보정 추가 진단**', '', now(), '',
             '기존 GPU 학습을 유지하고, 별도 CPU 1개 스레드에서 이미 저장된 자료만 분석했다. 진행 중인 학습의 소스·보상·체크포인트를 바꾸지 않았다.', '',
             '**적용 순서**', '',
             '1. 신뢰 기준: 경보 정밀도 90% 이상, 최소 3초 선행 사건 재현율 80% 이상, 오경보 1회/드론·시간 이하를 개발 목표로 기록한다. 현재 달성값이나 교수 지정값은 아니다.',
             '2. 실패 진단: 같은 관측의 사전학습/공동학습/LightGBM 비교에 사건 시점·지원 범위·라벨 정렬을 추가한다.',
             '3. 학습 자료: 훈련 지역의 방향·간섭 조합과 head별 양성 사건을 확인한 뒤 부족한 조건을 보강한다.',
             '4. 학습 목적: 예측기 고정+PPO와 공동학습을 별도 실험으로 비교한다. 동일 자료의 비교에서 예측이 나빠지는지 확인한 뒤 손실·학습률을 조절한다.',
             '5. 확률 보정: 공간 그룹을 나눠 보정 전후를 평가한다. 사건 경보 통과 여부와 확률 보정 성능을 구분한다.',
             '6. 모델 크기: 입력·라벨·자료와 학습 조건을 고정한 뒤 현재 Transformer와 제한된 크기 후보, LightGBM을 비교한다.', '',
             '1~5단계의 공통 개선을 두 예측기에 적용한 뒤 개선 전/후 각각 비교한다. Transformer만 유리한 입력이나 정답을 주지 않는다. 현재 이미 고정 모델의 첫 비교는 완료했고, 이번 분석은 그 진단을 확장한다.', '',
             '**사건 시점과 3초 판단 주기**', '',
             '| 경보 | 전체 사건 | 선행 판단 시점 있음 | 공통 관측 지원 | 현재 연결·유효 라벨 있음 | 해당 head 양성 라벨 있음 | 완벽한 저장 라벨의 실제 경보 적중 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for label in ['5s', '8s']:
        row = events['coverage'][label]['common_support']
        oracle = events['stored_label_oracle'][label]
        lines.append(f"| {label} | {row['eligible_events']} | {row['with_decision']} | {row['with_support']} | {row['with_valid_current_up_label']} | {row['with_positive_stored_label']} | {oracle['event_hits_3s']} |")
    lines += ['', '양성 라벨 기회는 경보 허용 구간 안에 저장된 정답이 양성인 판단 시점이 있었는지를 뜻한다. 마지막 열은 그 정답을 그대로 경보로 사용한 가상 진단이며 최초 경보·hysteresis의 영향을 포함한다. 미래 정답을 쓴 진단이므로 실사용 모델 성능이 아니며, 다른 예측기의 이론적 최대 성능으로 해석할 수 없다.', '',
              '사건 전 10/8/5/3/1초 점수는 event_alignment.json에 저장했다. 직전 3초 격자의 예측을 쓰므로 실제 선행시간은 표기보다 최대 2.5초 길다. 모든 사건에 대한 점수만 모은 결과로 오경보나 정밀도를 판단할 수 없다.', '',
              '**별도 지도학습 자료 범위**', '',
              f"고정 사전학습 corpus {corpus['episodes']}회는 계획 훈련 지역 {corpus['planned_train_regions']}개 중 {corpus['represented_regions']}개를 포함한다. 지역×방향×간섭 조합은 {corpus['represented_region_direction_interference_combinations']}개다. 두 방향·세 간섭 조건을 모두 한 번씩 포함하면 {corpus['planned_combinations_if_2_directions_3_conditions']}개가 된다. 이후 PPO가 새로 수집한 비행은 추가 경험이며 이 숫자에 포함하지 않았다.", '',
              '| 조건 | 비행 수 | 5초 위험 양성/유효 표본 | 3~4초 시작 양성/유효 표본 | 3~7초 시작 양성/유효 표본 | 지속 링크 단절 시작 수 |',
              '| --- | --- | --- | --- | --- | --- |']
    for label, row in corpus['summaries'].items():
        lines.append('| '+label+f" | {row['episodes']} | "+' | '.join(f"{row['positives'][h]}/{row['samples'][h]}" for h in range(3))+f" | {row['persistent_directed_link_events']} |")
    lines += ['', '양성 표본은 같은 사건의 여러 인접 시점이 포함될 수 있다. 링크별 단절 시작 수는 전체 네트워크 단절 수와 다른 지표다. 건물 모서리·재머 전환·비콘 노후화별 원인 분석은 아직 수행하지 않았다.', '',
              '**공간 교차 적합 확률 보정**', '',
              '각 fold에서 6개 지역으로 보정기를 맞추고 다른 2개 지역으로 평가했다. 원래 확률 logit에 양수 기울기와 절편을 맞추는 한 방법을 미리 정해 비교했다. 아래 값은 낮을수록 좋다.', '',
              '| 모델 | 예측 head | Brier 보정 전 → 후 | ECE 보정 전 → 후 |',
              '| --- | --- | --- | --- |']
    for model, value in calibration.items():
        for head in HEADS:
            row = value['heads'][head]
            before, after = row['before'], row['spatial_crossfit_after']
            lines.append(f"| {model} | {head} | {before['brier']:.6f} → {after['brier']:.6f} | {before['ece_equal_count_10']:.6f} → {after['ece_equal_count_10']:.6f} |")
    lines += ['', 'ECE는 표본 수가 같은 10개 구간의 진단값이며 희귀 사건에서는 작은 숫자만으로 충분히 보정됐다고 판단할 수 없다. 각 구간의 실제 발생률과 표본 수를 함께 저장했다. 보정은 기존 점수의 순서를 각 fold 안에서 단조 변환한다. fold마다 변환이 달라 전체를 합친 AP는 달라질 수 있으나 새로운 전조를 배웠다는 뜻은 아니다.', '',
              '| 모델 | 포함 관계 검사 표본 | 원래 위반 | 독립 head 보정 후 위반 |',
              '| --- | --- | --- | --- |']
    for model, value in calibration.items():
        row = value['common_head_nesting']
        lines.append(f"| {model} | {row['samples']} | {row['raw_violations']} | {row['calibrated_violations']} |")
    lines += ['', '포함 관계는 p(3~4초 시작) ≤ min(p(5초 위험), p(3~7초 시작))를 검사했다. head를 따로 보정하면 이 제약이 깨질 수 있다. 이번 보정기는 배포하지 않았고, 보정 뒤 경보 임계값이나 재배치 결과도 아직 재평가하지 않았다.', '',
              '**남은 실험**', '',
              '- 부족한 조건과 놓친 시작 사건을 훈련 지역에서 보강하고, head·사건 단위 추출과 어려운 정상 사례를 비교한다.',
              '- 동일 사전학습에서 예측기를 고정한 PPO와 공동 PPO를 비교한다. 예측 기준과 제어 기준의 best를 각각 보관한다.',
              '- 현재 Transformer(hidden 32/2층), hidden 64/2층, hidden 128/3층 후보와 LightGBM을 같은 원천 자료·정답·분할·평가 시점으로 학습한다. 작은 탐색으로 유효한 후보부터 좁히고 복수 seed로 확인한다.',
              '- 예측 자체의 비교 뒤 동일한 제어 연결 방식으로 연결율·PDR·단절·이동 비용을 비교한다. 예측기만 교체하면 입력 분포가 바뀔 수 있어 제어 성능은 별도 검증한다.',
              '- 최종 모델·보정·경보 규칙을 고정한 뒤 전체 검증과 미사용 테스트를 평가한다. 현 8개 맵 결과를 최종 일반화 성능으로 보고하지 않는다.', '',
              '현재 큰 모델 재학습이나 자동 후속 학습 예약은 하지 않았다. 기존 GPU run은 그대로 진행되며, 이번 작업은 CPU 진단과 재현 가능한 실행 계획까지다.', '',
              '**검증과 제한**', '',
              '합성 사건 두 개로 3초 판단 주기·저장 라벨·경보 집계 정렬을 검사했다. 보정기 단조성·최적화 수렴, fold 공간 분리, 기존 AP 재현, 모든 읽기 입력의 SHA와 훈련 192개 archive SHA를 확인했다. 완료 시 원본 입력과 소스 해시를 다시 검사했다.', '']
    lines += ['- '+item for item in protocol['limitations']]
    (JOB / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    JOB.mkdir(exist_ok=True)
    checks = self_check()
    if args.prepare_only:
        protocol = prepare()
        print(json.dumps(dict(protocol=str(JOB / 'protocol.json'), checks=checks,
                              validation_maps=len(protocol['observations']), final_test_accessed=False), ensure_ascii=False))
        return
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError('Could not set below-normal diagnostic priority')
    import msvcrt
    with (JOB / 'job.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if (JOB / 'results.json').exists():
            print('Diagnostics already completed; saved results retained.', flush=True)
            return
        try:
            status('preflight', checks=checks)
            protocol = prepare()
            corpus = corpus_audit()
            episodes = []
            for spec in protocol['observations']:
                with np.load(REPLAY / 'replay' / (spec['region']+'.npz'), allow_pickle=False) as source:
                    ep = {key: source[key] for key in source.files}
                ep['spec'] = spec
                episodes.append(ep)
            status('event_alignment')
            events = event_audit(episodes)
            calibration = calibration_audit(episodes)
            for path, expected in protocol['pinned_sha256'].items():
                if sha(path) != expected:
                    raise ValueError('Pinned input changed during diagnostic: '+path)
            report(protocol, corpus, events, calibration)
            write(JOB / 'results.json', dict(completed_at=now(), checks=checks,
                  protocol_sha256=sha(JOB / 'protocol.json'), immutable_inputs_verified=True,
                  corpus=corpus, events=events, calibration=calibration))
            write(JOB / 'status.json', dict(state='completed', stage='completed', pid=os.getpid(),
                  updated_at=now(), cpu_threads=1, cuda_used=False, report=str(JOB / 'report.md')))
            print('DIAGNOSTICS COMPLETE', flush=True)
        except BaseException as exc:
            write(JOB / 'status.json', dict(state='failed', stage='failed', pid=os.getpid(),
                  updated_at=now(), error=repr(exc)))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
