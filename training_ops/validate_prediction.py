"""Isolated CPU replay audit. Reads the experiment and writes only this job directory."""
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
JOB = OPS / 'prediction_validation_20261029_v2'
JOB.mkdir(exist_ok=True)
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['NUMBA_CACHE_DIR'] = str(JOB / 'numba_cache')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def status(stage, **extra):
    data = dict(state='running', stage=stage, pid=os.getpid(), updated_at=now(), cpu_threads=1, cuda_used=False, **extra)
    write(JOB / 'status.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def prepare():
    path = JOB / 'protocol.json'
    if path.exists():
        protocol = load(path)
        for name, expected in protocol['pinned_sha256'].items():
            actual = sha(PROJECT / name) if name != '@validation_script' else sha(__file__)
            if actual != expected:
                raise ValueError('Pinned validation input changed: ' + name)
        return protocol
    bootstrap = load(PROJECT / 'results/transformer/bootstrap/complete.json')
    plan = load(PROJECT / 'configs/transformer_plan.json')
    entries = {r['key']: r for r in plan['maps']}
    old = load(PROJECT / 'configs/usable_protocol.json')
    old_regions = {r['key'] for r in old['maps']}
    models = {
        'pretrain1200': 'results/transformer/pretrain/checkpoints/s0001200.pt',
        'pretrain2800': 'results/transformer/pretrain/checkpoints/s0002800.pt',
        'joint176': 'results/transformer/train/transformer_main/checkpoints/s0000176.pt',
        'joint190': 'results/transformer/train/transformer_main/checkpoints/s0000190.pt',
        'lightgbm': 'model/advance/selected/best.pt',
    }
    pinned = {name: sha(PROJECT / name) for name in models.values()}
    for name in ['model/advance/selected/calibration.json', 'model/advance/selected/alarm_policy.json',
                 'configs/transformer_plan.json', 'configs/usable_protocol.json']:
        pinned[name] = sha(PROJECT / name)
    pinned.update(bootstrap['source_sha256'])
    # These analysis helpers are not all covered by the training source manifest.
    for name in ['solo/evaluate.py', 'solo/metrics.py', 'solo/alarms.py', 'solo/checks.py']:
        pinned[name] = sha(PROJECT / name)
    for name, expected in pinned.items():
        if sha(PROJECT / name) != expected:
            raise ValueError('Frozen source mismatch: ' + name)
    episodes = []
    for prefix, digest in zip(bootstrap['validation_paths'], bootstrap['validation_hashes']):
        meta_path = (PROJECT / prefix).with_suffix('.json')
        array_path = (PROJECT / prefix).with_suffix('.npz')
        row = load(meta_path)
        entry = entries[row['region']]
        if entry['joint_split'] != 'validation' or entry['previous_role'] != 'fresh':
            raise ValueError('Only fresh validation geography allowed')
        if row['region'] in old_regions:
            raise ValueError('New validation map appears in historical protocol')
        if sha(array_path) != digest or row['array_sha256'] != digest:
            raise ValueError('Validation array hash mismatch')
        for item in [meta_path, array_path, PROJECT / entry['path']]:
            pinned[item.relative_to(PROJECT).as_posix()] = sha(item)
        episodes.append(dict(prefix=prefix, array_sha256=digest, region=row['region'],
                             spatial_group=entry['joint_spatial_group'], map_path=entry['path'],
                             direction=row['direction'], kind=row['kind'], seed=row['seed']))
    if len(episodes) != 8 or len({r['spatial_group'] for r in episodes}) != 8:
        raise ValueError('Expected eight spatially distinct diagnostic episodes')
    pinned['@validation_script'] = sha(__file__)
    protocol = dict(created_at=now(), models=models, episodes=episodes, pinned_sha256=pinned,
                    device='cpu', threads=1, simulation_runs=0, training_steps=0,
                    common_heads=['risk_5s', 'onset_3to4s', 'onset_3to7s'],
                    blend_weights=[0., .25, .5, .75, 1.], blend_weight_meaning='LightGBM fraction; remaining fraction is joint176',
                    outer_spatial_folds=4, fold_seed=2026102909,
                    profiles={'legacy': dict(precision=.8, false_alerts_per_drone_hour=5., minimum_alerts=10, event_recall=.8),
                              'trust_goal': dict(precision=.9, false_alerts_per_drone_hour=1., minimum_alerts=10, event_recall=.8)},
                    limitations=[
                        'These eight maps were already used to select checkpoints; cross-fitting blend weights and alarm thresholds does not remove earlier checkpoint-selection bias.',
                        'Diagnostic only: not full 96-case validation, final test, operational alarm authorization, or a controller rollout comparison.',
                        'Historical LightGBM training bank is not included; training-only checkpoint metadata and fresh-map protocol are checked, but raw bank provenance cannot be independently reconstructed.',
                        'No probability calibrator is fitted in this first audit; existing probabilities and calibration diagnostics are reported.',
                    ])
    write(path, protocol)
    return protocol


def replay(protocol):
    import numpy as np
    import torch
    from solo.model import RiskTransformer
    from solo.data import read_episode, PRED_KEYS, link_batch
    from cooperative.prediction import ControlPredictor
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('Validation must not see a CUDA device')
    cache = JOB / 'replay'
    cache.mkdir(exist_ok=True)
    models = {}
    for name, relative in protocol['models'].items():
        if name == 'lightgbm':
            continue
        state = torch.load(PROJECT / relative, map_location='cpu', weights_only=False)
        model = RiskTransformer()
        weights = state['model'] if name.startswith('pretrain') else {k[5:]:v for k,v in state['model'].items() if k.startswith('risk.')}
        model.load_state_dict(weights)
        model.eval()
        models[name] = model
    signature = sha(JOB / 'protocol.json')
    for i, ep in enumerate(protocol['episodes']):
        destination = cache / (ep['region'] + '.npz')
        metadata = destination.with_suffix('.json')
        if destination.exists() and metadata.exists():
            saved = load(metadata)
            if saved['protocol_sha256'] != signature or saved['npz_sha256'] != sha(destination):
                raise ValueError('Cached validation replay changed')
            status('replay_cached', complete=i+1, total=len(protocol['episodes']), region=ep['region'])
            continue
        began = time.perf_counter()
        status('replay_transformer', complete=i, total=8, region=ep['region'])
        data, _ = read_episode(PROJECT / ep['prefix'], 'validation', PRED_KEYS)
        rows = np.argwhere(data['prediction_supported'])
        result = dict(target=data['prediction_targets'], valid=data['prediction_valid'],
                      native_support=data['prediction_supported'])
        with np.load((PROJECT / ep['prefix']).with_suffix('.npz'), allow_pickle=False) as source:
            result['links'] = source['links']
        for name, model in models.items():
            output = np.full((*data['prediction_supported'].shape, 4), np.nan, np.float32)
            with torch.inference_mode():
                for first in range(0, len(rows), 256):
                    part = rows[first:first+256]
                    batch = link_batch(data, part, 'cpu')
                    _, prob = model.encode_links(batch['values'], batch['masks'], batch['intents'], batch['geometry'])
                    t, relay, peer = part.T
                    output[t, relay, peer] = prob.numpy()
            result[name] = output
        status('replay_lightgbm', complete=i, total=8, region=ep['region'])
        predictor = ControlPredictor(PROJECT / protocol['models']['lightgbm'], PROJECT / ep['map_path'], threads=1, observers=range(3,8))
        output = np.full((*data['prediction_supported'].shape, 3), np.nan, np.float32)
        last_heartbeat = time.monotonic()
        for tick in range(len(data['rf_values'])):
            x = np.zeros((8,8,48), np.float32)
            mask = np.zeros_like(x, bool)
            intent = np.zeros((8,8,9), np.float32)
            x[3:8] = data['rf_values'][tick]
            mask[3:8] = data['rf_masks'][tick]
            intent[3:8] = data['rf_intents'][tick]
            observed = predictor.observe((tick+1)*.5, x, mask, intent)
            decision = (tick+1)//6
            if (tick+1)%6 == 0 and decision < len(output) and observed['status'] == 'ready':
                for row in observed['links']:
                    relay, peer = row['source']-3, row['peer']
                    if 0 <= relay < 5:
                        for head, keyname in enumerate(['risk_5s', 'risk_onset_3to4s', 'risk_onset_3to7s']):
                            if row.get(keyname) is not None:
                                output[decision, relay, peer, head] = row[keyname]
            if time.monotonic()-last_heartbeat > 30:
                status('replay_lightgbm', complete=i, total=8, region=ep['region'], rf_frames=tick+1, total_rf_frames=len(data['rf_values']))
                last_heartbeat = time.monotonic()
        result['lightgbm'] = output
        result['common_support'] = data['prediction_supported'] & np.isfinite(output).all(-1)
        temporary = destination.with_suffix('.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f, **result)
        os.replace(temporary, destination)
        write(metadata, dict(protocol_sha256=signature, npz_sha256=sha(destination),
                             region=ep['region'], wall_s=time.perf_counter()-began))
        status('replay_committed', complete=i+1, total=8, region=ep['region'], wall_s=time.perf_counter()-began)
        del predictor, data, result


def analyze(protocol):
    import numpy as np
    from solo.metrics import binary_metrics, alarm_trials, alert_metrics
    from sklearn.metrics import precision_recall_curve
    cache = JOB / 'replay'
    episodes = []
    for spec in protocol['episodes']:
        with np.load(cache / (spec['region'] + '.npz'), allow_pickle=False) as z:
            episodes.append(dict(spec=spec, **{k:z[k] for k in z.files}))
    heads = protocol['common_heads']
    base_names = list(protocol['models'])

    def probability(ep, name):
        if isinstance(name, float):
            return name*ep['lightgbm']+(1-name)*ep['joint176'][...,:3]
        return ep[name][...,:3]

    def vectors(indices, name, head, common=True):
        ys, ps = [], []
        for i in indices:
            ep = episodes[i]
            pred = probability(ep, name)[...,head]
            mask = ep['common_support'] if common else ep['native_support']
            keep = mask & ep['valid'][...,head] & np.isfinite(pred)
            ys.append(ep['target'][...,head][keep])
            ps.append(pred[keep])
        return np.concatenate(ys), np.concatenate(ps)

    def metrics(indices, name, head, common=True):
        y, p = vectors(indices, name, head, common)
        result = binary_metrics(y,p)
        precision, recall, thresholds = precision_recall_curve(y,p)
        result['recall_at_precision'] = {}
        for requirement in [.8,.9]:
            select = (precision[:-1] >= requirement) & (recall[:-1]*y.sum()/np.maximum(precision[:-1],1e-15)>=10-1e-7)
            result['recall_at_precision'][str(requirement)] = float(recall[:-1][select].max()) if select.any() else 0.
        ordered = np.argsort(p, kind='stable')
        bins = []
        for ix in np.array_split(ordered,10):
            if len(ix):
                bins.append(dict(count=len(ix), mean_probability=float(p[ix].mean()), actual_fraction=float(y[ix].mean())))
        result['calibration_bins'] = bins
        result['ece_equal_count_10'] = float(sum(r['count']*abs(r['mean_probability']-r['actual_fraction']) for r in bins)/len(p))
        return result

    all_indices = list(range(len(episodes)))
    classification = {name:{head:metrics(all_indices,name,h) for h,head in enumerate(heads)} for name in base_names}
    native = {name:{head:metrics(all_indices,name,h,False) for h,head in enumerate(heads)} for name in base_names if name != 'lightgbm'}
    prior = load(PROJECT/'results/transformer/lightgbm_reference.json')
    parity = {}
    for name, prior_key in [('pretrain1200','standalone_transformer'),('lightgbm','metrics')]:
        parity[name] = {h:dict(sample_count_equal=classification[name][h]['samples']==prior[prior_key][h]['samples'],
                             positives_equal=classification[name][h]['positives']==prior[prior_key][h]['positives'],
                             ap_error=abs(classification[name][h]['average_precision']-prior[prior_key][h]['average_precision'])) for h in heads}
    if not all(v['sample_count_equal'] and v['positives_equal'] and v['ap_error']<1e-4 for rows in parity.values() for v in rows.values()):
        write(JOB/'replay_parity_failure.json',parity)
        raise ValueError('Replay did not reproduce prior same-corpus benchmark; stop before blending')
    write(JOB/'classification.json',dict(common=classification,native_transformer=native,prior_replay_check=parity))
    status('classification_complete', common_risk5_AP={name:classification[name]['risk_5s']['average_precision'] for name in base_names})

    per_region = []
    for i, ep in enumerate(episodes):
        rows = {}
        for name in base_names:
            rows[name] = {h:binary_metrics(*vectors([i],name,j)) for j,h in enumerate(heads)}
        per_region.append(dict(**ep['spec'], metrics=rows))
    coverage = []
    for ep in episodes:
        coverage.append(dict(region=ep['spec']['region'], native_supported=int(ep['native_support'].sum()),
                             common_supported=int(ep['common_support'].sum()),
                             eligible_positive_risk5=int((ep['native_support']&ep['valid'][...,0]&ep['target'][...,0].astype(bool)).sum()),
                             common_positive_risk5=int((ep['common_support']&ep['valid'][...,0]&ep['target'][...,0].astype(bool)).sum())))
    weights = protocol['blend_weights']
    descriptive_blends = {str(w):{h:metrics(all_indices,float(w),j) for j,h in enumerate(heads)} for w in weights}
    # Cross-fit only the fusion weight. All source models remain frozen and no RF rollout is repeated.
    rng = np.random.default_rng(protocol['fold_seed'])
    order = rng.permutation(len(episodes))
    heldout = [sorted(x.tolist()) for x in np.array_split(order,4)]
    fold_rows, crossfit_p = [], [None]*len(episodes)
    for fold, test in enumerate(heldout):
        tune = [i for i in all_indices if i not in test]
        scores = {}
        for w in weights:
            a = metrics(tune,float(w),0)['average_precision']
            b = metrics(tune,float(w),2)['average_precision']
            scores[w] = .6*a+.4*b
        w = max(weights,key=lambda q:(scores[q],-q))
        for i in test:
            crossfit_p[i] = probability(episodes[i],float(w))
        fold_rows.append(dict(fold=fold,tune_regions=[episodes[i]['spec']['region'] for i in tune],
                              evaluation_regions=[episodes[i]['spec']['region'] for i in test],
                              selected_lightgbm_weight=w,tuning_scores=scores))
    for ep,p in zip(episodes,crossfit_p):
        ep['blend_crossfit'] = p
    crossfit_classification = {h:metrics(all_indices,'blend_crossfit',j) for j,h in enumerate(heads)}
    write(JOB/'fusion.json',dict(descriptive_all_map_weight_sweep=descriptive_blends,crossfit_folds=fold_rows,
                                crossfit_classification=crossfit_classification,selection_score='.6 AP risk5 + .4 AP onset37',
                                warning='All-map weight sweeps are tuning diagnostics; use cross-fit result for a less optimistic weight comparison. Earlier checkpoint-selection bias remains.'))
    status('fusion_classification_complete',crossfit_risk5_AP=crossfit_classification['risk_5s']['average_precision'])

    def event_episodes(indices,name):
        return [dict(prediction=probability(episodes[i],name),prediction_supported=episodes[i]['common_support'],links=episodes[i]['links']) for i in indices]

    def trials_for(items,head,horizon):
        vals=[]
        for ep in items:
            t=np.arange(len(ep['prediction']))*3.
            use=(t>=21)&(t<=len(ep['links'])*.5-8)
            p=ep['prediction'][...,head]
            keep=ep['prediction_supported']&use[:,None,None]&np.isfinite(p)
            vals.append(p[keep])
        scores=np.concatenate(vals)
        quantiles=np.quantile(scores,np.unique(np.r_[np.linspace(0,1,49),np.linspace(.9,1,51)]))
        enter=np.unique(np.clip(np.r_[np.linspace(.01,.99,35),quantiles,.7674193548387096,.7451612903225806],0,1))
        return alarm_trials(items,head,np.repeat(enter,3),np.repeat(enter,3)*np.tile([.25,.5,1.],len(enter)),horizon)

    def choose(trials,profile):
        eligible=[r for r in trials if r['first_alerts']>=profile['minimum_alerts'] and r['first_alert_precision']>=profile['precision'] and r['false_alerts_per_drone_hour']<=profile['false_alerts_per_drone_hour']]
        if not eligible:
            return None
        return max(eligible,key=lambda r:(r['first_alert_event_recall_3s'],r['first_alert_precision'],-r['false_alerts_per_drone_hour']))

    def merge(rows):
        fields=['first_alerts','correct_first_alerts','false_or_late_first_alerts','eligible_events','event_hits_3s','drone_hours']
        result={k:sum(r[k] for r in rows) for k in fields}
        result.update(first_alert_precision=result['correct_first_alerts']/max(1,result['first_alerts']),
                      false_alerts_per_drone_hour=result['false_or_late_first_alerts']/max(1e-12,result['drone_hours']),
                      first_alert_event_recall_3s=result['event_hits_3s']/max(1,result['eligible_events']))
        return result

    alarms={}
    alarm_names=['pretrain2800','joint176','joint190','lightgbm','blend_crossfit']
    for name in alarm_names:
        alarms[name]={}
        for label,head,horizon in [('5s',1,5),('8s',2,8)]:
            records={p:[] for p in protocol['profiles']}
            fold_rules=[]
            for fold,test in enumerate(heldout):
                tune=[i for i in all_indices if i not in test]
                # For fusion, both weight selection and threshold fitting use tuning maps only.
                if name=='blend_crossfit':
                    name_at_fold=float(fold_rows[fold]['selected_lightgbm_weight'])
                else:
                    name_at_fold=name
                fitted=trials_for(event_episodes(tune,name_at_fold),head,horizon)
                fold_detail=dict(fold=fold,rules={})
                for profile_name,profile in protocol['profiles'].items():
                    rule=choose(fitted,profile)
                    # No admissible tuning rule means disabled: score its missed events rather than omitting the maps.
                    if rule is None:
                        score=alert_metrics(event_episodes(test,name_at_fold),head,2.,2.,horizon)
                    else:
                        score=alert_metrics(event_episodes(test,name_at_fold),head,rule['threshold'],rule['exit_threshold'],horizon)
                    records[profile_name].append(score)
                    fold_detail['rules'][profile_name]=dict(enabled=rule is not None,tuning_rule=rule,evaluation=score)
                fold_rules.append(fold_detail)
                status('alarm_crossfit',model=name,horizon=label,fold=fold+1,total_folds=4)
            profiles={}
            for profile_name,rows in records.items():
                aggregate=merge(rows)
                p=protocol['profiles'][profile_name]
                aggregate['meets_all_development_targets']=bool(aggregate['first_alerts']>=p['minimum_alerts'] and aggregate['first_alert_precision']>=p['precision'] and aggregate['false_alerts_per_drone_hour']<=p['false_alerts_per_drone_hour'] and aggregate['first_alert_event_recall_3s']>=p['event_recall'])
                profiles[profile_name]=aggregate
            alarms[name][label]=dict(crossfit_profiles=profiles,fold_rules=fold_rules)
        write(JOB/'alarm_crossfit.json',alarms)

    complement={}
    for j,head in enumerate(heads):
        y,a=vectors(all_indices,'joint176',j)
        _,b=vectors(all_indices,'lightgbm',j)
        positive=y.astype(bool)
        complement[head]=dict(positive_samples=int(positive.sum()),
                              both_detect_at_half=int((positive&(a>=.5)&(b>=.5)).sum()),
                              tree_only_detect_at_half=int((positive&(a<.5)&(b>=.5)).sum()),
                              transformer_only_detect_at_half=int((positive&(a>=.5)&(b<.5)).sum()),
                              both_miss_at_half=int((positive&(a<.5)&(b<.5)).sum()))
    final=dict(protocol_sha256=sha(JOB/'protocol.json'),completed_at=now(),classification=classification,
               native_transformer=native,fusion_crossfit=crossfit_classification,folds=fold_rows,
               alarms_crossfit=alarms,coverage=coverage,per_region=per_region,
               error_complementarity=complement,prior_replay_check=parity,limitations=protocol['limitations'])
    write(JOB/'results.json',final)
    report(protocol,final)


def report(protocol,result):
    lines=['**예측기 동일 관측 예비 검증**','',result['completed_at'],'',
           '기존 CUDA 본 학습을 변경하지 않고 고정된 검증 관측 8회를 CPU 1개 스레드로 재생했다. 새 시뮬레이션·학습·최종 테스트는 수행하지 않았다.','',
           '이 자료는 이미 PPO checkpoint 선택에 사용한 맵이다. 공간 그룹 교차 적합은 혼합 비율과 경보 문턱의 직접 과적합을 줄일 뿐, 과거 모델 선택 편향까지 제거하지 않는다.','',
           '**동일 링크·시점·정답에서의 AP**','','| 모델 | 5초 위험 AP | 3–4초 시작 AP | 3–7초 시작 AP |','| --- | --- | --- | --- |']
    all_metrics=dict(result['classification'],blend_crossfit=result['fusion_crossfit'])
    for name,metrics in all_metrics.items():
        lines.append('| '+name+' | '+' | '.join(f"{metrics[h]['average_precision']:.4f}" for h in protocol['common_heads'])+' |')
    lines.extend(['','LightGBM이 확률을 낼 수 있는 공통 지원 표본으로 비교했다. Transformer 원래 지원 범위의 결과와 제외 표본 수는 classification.json/results.json에 별도 저장했다.','',
                  '**공간 그룹 교차 적합 사건 경보**','','| 모델 | 경보 | 조건 | 경보 정밀도 | 오경보/드론·시간 | 3초 선행 사건 재현율 | 탐지/사건 | 목표 전부 충족 |','| --- | --- | --- | --- | --- | --- | --- | --- |'])
    for name, horizons in result['alarms_crossfit'].items():
        for horizon,data in horizons.items():
            for profile,row in data['crossfit_profiles'].items():
                lines.append(f"| {name} | {horizon} | {profile} | {row['first_alert_precision']*100:.2f}% | {row['false_alerts_per_drone_hour']:.3f} | {row['first_alert_event_recall_3s']*100:.2f}% | {row['event_hits_3s']}/{row['eligible_events']} | {row['meets_all_development_targets']} |")
    lines.extend(['','legacy는 문턱 선택 시 정밀도 80%·오경보 5회/h·경보 10건, trust_goal은 90%·1회/h·10건을 요구한다. 목표 전부 충족 여부에는 최소 3초 선행 사건 재현율 80%도 포함했다. 숫자는 개발 목표이며 운용 승인 기준이나 이미 달성한 성능으로 정하지 않았다.','',
                  '튜닝 맵에서 허용 가능한 규칙이 없으면 그 fold의 경보를 끄고, 평가 맵의 놓친 사건은 분모에 그대로 남겼다. 최초 경보 정밀도는 horizon 내 사건 유무, 사건 재현율은 3초 이상 선행한 별도 정의다.','',
                  '**혼합 비율**',''])
    for fold in result['folds']:
        lines.append(f"- fold {fold['fold']+1}: LightGBM 비중 {fold['selected_lightgbm_weight']:.2f}; 평가 지역 {', '.join(fold['evaluation_regions'])}")
    lines.extend(['','혼합은 176 Transformer와 LightGBM의 확률 가중 평균이다. 비율은 각 fold의 나머지 6개 맵에서 AP 점수로 선택했다. 복잡한 stacking이나 새 확률 보정기는 학습하지 않았다.','',
                  '**확인할 다음 항목**','',
                  '1. 모델별 AP 개선과 사건 경보 개선이 같은지 확인한다. AP만 오른 혼합을 신뢰 가능한 경보로 채택하지 않는다.',
                  '2. calibration_bins, recall_at_precision, error_complementarity와 지역별 결과에서 누락 사건/오경보 집중 조건을 확인한다.',
                  '3. 이 결과로 최종 테스트를 열거나 기존 학습에 혼합을 주입하지 않는다. 전체 검증과 독립된 튜닝/평가 설계가 다음 단계다.',
                  '4. 별도 모델 재학습, 손실 계수 조정, 모델 크기 변경은 아직 수행하지 않았다.','',
                  '**제한**','']+['- '+x for x in protocol['limitations']]+[''])
    (JOB/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if os.name=='nt':
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000):
            raise OSError('Could not set below-normal validation priority')
    import msvcrt
    with (JOB/'job.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0');lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            raise SystemExit('Prediction validation already running')
        protocol=prepare()
        if args.prepare_only:
            print(json.dumps(dict(protocol=str(JOB/'protocol.json'),episodes=len(protocol['episodes']),models=list(protocol['models']),final_test_accessed=False),ensure_ascii=False))
            return
        if (JOB/'results.json').exists():
            print('Validation already completed; saved results retained.',flush=True)
            return
        try:
            status('starting',started_at=now())
            replay(protocol)
            analyze(protocol)
            # Prove immutable inputs and all project sources still match after the audit.
            prepare()
            write(JOB/'status.json',dict(state='completed',stage='completed',pid=os.getpid(),updated_at=now(),
                                         cpu_threads=1,cuda_used=False,report=str(JOB/'report.md')))
            print('VALIDATION COMPLETE',flush=True)
        except BaseException as exc:
            write(JOB/'status.json',dict(state='failed',stage='failed',pid=os.getpid(),updated_at=now(),error=repr(exc)))
            traceback.print_exc()
            raise


if __name__=='__main__':
    main()
