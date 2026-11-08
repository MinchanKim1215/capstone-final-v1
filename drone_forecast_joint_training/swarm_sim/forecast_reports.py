"""모델별·지역별 지표를 보존하는 예측 비교표. 종합 점수나 자동 최적 선언은 하지 않는다."""
from pathlib import Path
from html import escape
import csv
import json
import numpy as np
from .regions import read_json, write_json
from .forecast_benchmark import read_config, experiment_contract, fingerprint, write_csv
from .forecast_models import MODEL_NAMES


def _number(value, digits=4):
    if value is None:
        return '—'
    if isinstance(value, int):
        return str(value)
    return f'{value:.{digits}f}' if isinstance(value, float) else escape(str(value))


def _table(rows, columns):
    if not rows:
        return '<p class="muted">아직 실행된 결과가 없습니다.</p>'
    heads = ''.join(f'<th>{escape(label)}</th>' for key,label in columns)
    body = ''.join('<tr>'+''.join(f'<td>{_number(row.get(key))}</td>' for key,label in columns)+'</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>'


def _curve(history, title):
    if not history:
        return ''
    width, height = 360, 150
    def path(key, ymax):
        points = []
        for i, row in enumerate(history):
            value = row.get(key)
            if value is not None:
                x = 30+i/max(1, len(history)-1)*(width-45)
                y = height-25-min(1., value/max(ymax, 1e-8))*(height-45)
                points.append(f'{x:.1f},{y:.1f}')
        return ' '.join(points)
    return f'''<div class="curve"><b>{escape(title)}</b>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="검증 지역 평균 AP 학습 곡선">
    <line x1="30" y1="125" x2="345" y2="125" stroke="#ccd4df"/>
    <line x1="30" y1="20" x2="30" y2="125" stroke="#ccd4df"/>
    <text x="5" y="25" font-size="10">1.0</text><text x="12" y="128" font-size="10">0</text>
    <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{path('val_macro_available_region_ap', 1.)}"/>
    <text x="260" y="145" font-size="10">epoch 1–{len(history)}</text></svg>
    <span class="muted">현재 가용 링크의 검증 지역별 AP 평균</span></div>'''


def comparison_report(config_path):
    cfg, folder, output = read_config(config_path)
    token = fingerprint(experiment_contract(cfg, folder))
    runs, detail_rows, region_rows, curves = [], [], [], []
    missing = []
    for model in MODEL_NAMES:
        for seed in cfg['seeds']:
            run_folder = output/model/f'seed_{seed}'
            if not (run_folder/'run.json').exists():
                missing.append(f'{model}/seed_{seed}')
                continue
            run = read_json(run_folder/'run.json')
            if run['experiment_sha256'] != token:
                raise ValueError('서로 다른 코드·데이터·설정의 결과를 한 비교표에 섞을 수 없습니다.')
            if not run.get('complete'):
                missing.append(f'{model}/seed_{seed} (미완료)')
                continue
            runs.append(run)
            if (run_folder/'history.csv').exists():
                with (run_folder/'history.csv').open(encoding='utf-8-sig') as f:
                    history = [{k:float(v) if v else None for k,v in row.items()} for row in csv.DictReader(f)]
                curves.append(_curve(history, f'{model} / seed {seed}'))
            for split in ('val','test'):
                path = run_folder/f'{split}_metrics.json'
                if not path.exists():
                    continue
                metrics = read_json(path)
                if split == 'test' and metrics.get('checkpoint_sha256') != run['checkpoint_sha256']:
                    raise ValueError('시험 결과와 현재 체크포인트가 다릅니다.')
                up, all_ = metrics['currently_available'], metrics['all']
                event = metrics['event_metrics']
                detail_rows.append({'model':model, 'seed':seed, 'split':split, 'stage':cfg['stage'],
                    'parameters':run['parameters'], 'best_epoch':run['best_epoch'],
                    'threshold':metrics['threshold'], 'samples_all':all_['samples'],
                    'samples_currently_available':up['samples'],
                    'available_positive_fraction':up['positive_fraction'],
                    'macro_available_region_ap':metrics['macro_available_region_ap'],
                    'macro_valid_regions':metrics['macro_available_region_ap_regions'],
                    'macro_excluded_regions':metrics['macro_available_region_ap_excluded_regions'],
                    'available_ap':up['average_precision'], 'available_auroc':up['auroc'],
                    'available_recall':up['recall'], 'available_precision':up['precision'],
                    'available_f1':up['f1'], 'available_fpr':up['false_positive_rate'],
                    'available_brier':up['brier'], 'all_ap':all_['average_precision'],
                    'all_auroc':all_['auroc'], 'all_accuracy':all_['accuracy'],
                    'all_recall':all_['recall'], 'all_precision':all_['precision'],
                    'all_f1':all_['f1'], 'all_fpr':all_['false_positive_rate'],
                    'all_brier':all_['brier'], 'all_log_loss':all_['log_loss'],
                    'eligible_events':event['eligible_events'], 'event_recall':event['event_recall'],
                    'strictly_early_recall':event['strictly_early_recall'],
                    'lead_time_median_s':event['lead_time_median_s'],
                    'lead_time_p10_s':event['lead_time_p10_s'],
                    'inference_p95_ms':run['latency']['single_window_p95_ms']})
                for region, values in metrics['by_region'].items():
                    for cohort, m in values.items():
                        region_rows.append({'model':model, 'seed':seed, 'split':split,
                                            'region':region, 'cohort':cohort, **m})
    aggregates = []
    fields = ['macro_available_region_ap','available_ap','available_auroc','available_recall',
              'available_precision','available_f1','available_fpr','available_brier',
              'event_recall','strictly_early_recall','lead_time_median_s','inference_p95_ms']
    for split in ('val','test'):
        for model in MODEL_NAMES:
            selected = [r for r in detail_rows if r['model'] == model and r['split'] == split]
            if not selected:
                continue
            aggregate = {'model':model, 'split':split, 'seeds_completed':len(selected),
                         'seeds_planned':len(cfg['seeds']), 'stage':cfg['stage']}
            for key in fields:
                values = [r[key] for r in selected if r[key] is not None]
                aggregate[key+'_mean'] = float(np.mean(values)) if values else None
                aggregate[key+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                aggregate[key+'_min'] = float(np.min(values)) if values else None
                aggregate[key+'_max'] = float(np.max(values)) if values else None
                aggregate[key+'_valid_seeds'] = len(values)
            aggregates.append(aggregate)
    meta = read_json(folder/'dataset_manifest.json')
    summary = {'format':'forecast-comparison-v1', 'stage':cfg['stage'],
        'experiment_sha256':token, 'completed_runs':len(runs), 'planned_runs':3*len(cfg['seeds']),
        'missing_runs':missing, 'dataset_splits':meta['splits'],
        'lookback_s':meta['window_settings']['lookback_s'],
        'horizon_s':meta['window_settings']['horizon_s'],
        'minimum_outage_s':meta['window_settings']['min_outage_s'],
        'individual_runs':detail_rows, 'aggregates':aggregates,
        'final_model_selected':False, 'repositioning_evaluated':False,
        'ap_definition':'sklearn average_precision_score; 계단 가중 AP이며 사다리꼴 PR-AUC와 다름',
        'seed_std_scope':'같은 수집 데이터에서 모델 초기화/학습 순서 시드 간 편차. 독립 도시 일반화 신뢰구간이 아님.'}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'comparison.json', summary)
    write_csv(output/'comparison_by_seed.csv', detail_rows)
    write_csv(output/'comparison_by_region.csv', region_rows)
    write_csv(output/'comparison_mean_std.csv', aggregates)
    stage_text = {'smoke':'실행 확인용 · 일부 시간 창 · 최종 성능/모델 선정에 사용하지 않음',
                  'pilot':'예비 비교 · 제공된 8회 비행 데이터 · 추가 환경 검증 필요',
                  'final':'확장 데이터의 예측 비교 · 수집 시나리오 범위 안에서 해석'}[cfg['stage']]
    columns = [('model','모델'),('seed','학습 시드'),('parameters','파라미터 수'),('best_epoch','선택 epoch'),
        ('macro_available_region_ap','지역 평균 AP'),('available_ap','가용 링크 AP'),
        ('available_recall','재현율'),('available_fpr','오경보율 FPR'),
        ('strictly_early_recall','사전 경고 사건 재현율'),('lead_time_median_s','탐지 사건 중앙 선행시간(s)'),
        ('inference_p95_ms','단일 창 추론 p95(ms)')]
    region_columns = [('model','모델'),('seed','시드'),('split','분할'),('region','지역'),
        ('samples','창 수'),('positives','양성 수'),('average_precision','AP'),
        ('auroc','AUROC'),('recall','재현율'),('precision','정밀도'),('false_positive_rate','FPR')]
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>GRU · TCN · Transformer 예측 비교</title><style>
    body{{font-family:"Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",sans-serif;background:#f4f6fa;color:#17233b;margin:0}}
    main{{max-width:1280px;margin:30px auto;padding:24px;background:white;border-radius:14px}}
    h1{{font-size:27px}}h2{{font-size:20px;margin-top:32px}}p,li{{line-height:1.8}}.note{{padding:14px 18px;background:#fff5d6;border-left:4px solid #d7a11b}}
    .muted{{color:#626e82;font-size:13px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}
    th,td{{padding:10px 12px;border-bottom:1px solid #e0e6ef;text-align:right}}th{{background:#eaf0fa;color:#203e6b}}th:first-child,td:first-child{{text-align:left}}
    .curves{{display:flex;flex-wrap:wrap;gap:16px}}.curve{{width:360px;border:1px solid #e0e6ef;border-radius:10px;padding:12px}}svg{{width:100%}}a{{color:#1c56a0}}
    </style><main><p class="muted">통신 장애 예측 전용 비교 · 재배치 성능은 평가하지 않음</p>
    <h1>GRU · TCN · Transformer</h1><p class="note">{stage_text}</p>
    <p>과거 {summary['lookback_s']:g}초 → 향후 {summary['horizon_s']:g}초 안의 연속 {summary['minimum_outage_s']:g}초 이상 링크 불가용 위험.
    학습 완료 {len(runs)}/{summary['planned_runs']}회. 세 모델의 입력·정답·분할·표준화 기준은 같습니다.</p>
    <p class="muted">완료되지 않은 실행: {escape(', '.join(missing)) if missing else '없음'}.
    서로 다른 지표를 합친 종합 점수와 자동 최적 모델 선언은 제공하지 않습니다.</p>
    <h2>검증 결과</h2>{_table([r for r in detail_rows if r['split']=='val'], columns)}
    <h2>고정 모델의 시험 결과</h2>{_table([r for r in detail_rows if r['split']=='test'], columns)}
    <h2>지역별 결과: 현재 가용 링크의 미래 단절</h2>
    {_table([r for r in region_rows if r['cohort']=='currently_available'], region_columns)}
    <h2>학습 중 검증 AP</h2><div class="curves">{''.join(curves)}</div>
    <h2>해석 기준</h2><ul>
    <li>AP는 average precision입니다. FPR은 FP/(FP+TN)이며, 잘못된 경고의 비율 FP/(TP+FP)과 다릅니다.</li>
    <li>현재 가용성은 시뮬레이터 진실값으로 평가 집단만 나눈 정보입니다. 신경망 입력이나 실제 운용의 경고 필터로 주지 않습니다.</li>
    <li>전체 창 결과에는 이미 끊긴 링크의 지속도 포함됩니다. 세부 CSV/JSON에 전체 창과 현재 가용 링크 결과를 함께 보존합니다.</li>
    <li>사건 재현율은 예측 가능한 과거 창이 존재한 방향별 지속 단절 사건에서 계산합니다. 누락 사건 수도 각 metrics.json에 기록합니다.</li>
    <li>사전 경고 사건 재현율은 선행시간이 0초보다 큰 탐지만 셉니다. 중앙 선행시간은 탐지한 사건에 한정하며 미탐지는 별도입니다.</li>
    <li>AP/AUROC를 정의하기 어려운 단일 클래스 집단은 빈 값으로 표시합니다. 제외 지역 수를 CSV에 기록합니다.</li>
    <li>모델/epoch는 검증 지역 평균 AP로, 경고 기준은 검증 가용 링크 F1으로 정합니다. 시험 결과로 다시 조정하지 않습니다.</li>
    <li>시간 창은 서로 겹칩니다. 모델 시드 간 표준편차를 독립 비행이나 도시 일반화의 신뢰구간으로 해석하지 않습니다.</li>
    <li>기존 5개 지역은 배치 선정에 사용했습니다. 여기서 두 지역 보류는 예측기 학습에 대한 보류입니다.</li>
    <li>연결 유지율·PDR·재배치 이동량 개선은 이 실험의 결과가 아닙니다. 현재 모델 확률의 실세계 보정도 검증하지 않았습니다.</li>
    </ul><p><a href="comparison_by_seed.csv">시드별 모든 지표 CSV</a> · <a href="comparison_by_region.csv">지역별 CSV</a> ·
    <a href="comparison_mean_std.csv">모델 시드 평균·표준편차 CSV</a> · <a href="comparison.json">전체 JSON</a></p>
    <p class="muted">GRU: Chung et al., arXiv:1412.3555 · TCN: Bai et al., arXiv:1803.01271 ·
    Transformer 분류 참고: Zerveas et al., KDD 2021. 구현 차이는 docs/예측모델_3종_테스트안내.md에 명시했습니다.</p></main></html>'''
    (output/'comparison.html').write_text(html, encoding='utf-8')
    return {'html':str(output/'comparison.html'), 'completed_runs':len(runs),
            'test_rows':sum(r['split']=='test' for r in detail_rows)}
