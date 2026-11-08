"""실제 저장 수치만 읽어 독립 실행 HTML 설명서와 CSV를 생성한다."""
from __future__ import annotations

from pathlib import Path
import csv
import html
import json
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from swarm_sim.regions import ROOT, read_json, write_json
from upgrade.collect import BASE_FEATURES, EXTRA_FEATURES
from upgrade.evaluate import EVALUATION_VERSION
from upgrade.train import RUNS, VARIANTS

RESULTS = ROOT/'results'/'upgrade'
DOCS = ROOT/'docs'
LABELS = {
    'A0_three_regions': ('A0', '3개 지역 · 기본 12항목'),
    'A1_more_maps': ('A1', '30개 지역 · 기본 12항목'),
    'A2_observed_geometry': ('A2', '30개 지역 · 관측 보강 28항목'),
    'A3_masked_pretraining': ('A3', '보강 입력 · 가림 복원 사전학습'),
    'A4_matched_compute': ('A4', '보강 입력 · 더 긴 지도학습 예산'),
}
KOREAN_FEATURES = [
    '자신의 x축 속도', '자신의 y축 속도', '자신의 고도', '자신의 중계 역할 여부', '상대의 중계 역할 여부',
    '직전 구간 송신 시도 수', '직전 구간 ACK 성공 수', '직전 구간 ACK 성공 비율',
    '마지막 ACK 수신 전력', '마지막 ACK SINR', '마지막 ACK 이후 경과 시간', '자신의 큐 길이',
    '자신의 x 위치', '자신의 y 위치', '직전 구간 수신 성공 수', '상대 상태의 시각 이후 경과 시간',
    '추정 상대 x 거리', '추정 상대 y 거리', '마지막 수신 상태 기반 상대 x 속도', '마지막 수신 상태 기반 상대 y 속도',
    '추정 상대 거리', '추정 거리 변화율', '현재 지도 기반 추정 손실', '5초 뒤 지도 기반 추정 손실',
    '현재 추정 LoS 여부', '5초 뒤 추정 LoS 여부', '5초 동안의 추정 손실 변화', '높이 40m 이상 건물까지의 거리',
]


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def percent(value, digits=2):
    return '—' if value is None else f'{100*value:.{digits}f}%'


def table(headers, rows, cls=''):
    return '<div class="table-scroll"><table class="'+cls+'"><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def save_figure(fig, name):
    assets = DOCS/'figures'; assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets/(name+'.svg'), bbox_inches='tight')
    fig.savefig(assets/(name+'.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    text = (assets/(name+'.svg')).read_text()
    return text[text.index('<svg'):]


def predictions(name):
    ys, ps = [], []
    for path in sorted((RESULTS/'dense'/name).glob('*.npz')):
        with np.load(path, allow_pickle=False) as f:
            ys.append(f['y']); ps.append(f['probability'])
    return np.concatenate(ys), np.concatenate(ps)


def embedded_fonts():
    """배포용 Noto 부분 글꼴이 있으면 외부 접속 없이 같은 한글 서체를 사용한다."""
    faces = []
    for weight, name in [(400, 'CapstoneNoto-Regular.woff'), (700, 'CapstoneNoto-Bold.woff')]:
        path = DOCS/'assets'/name
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode('ascii')
            faces.append(f"@font-face{{font-family:CapstoneSans;font-style:normal;font-weight:{weight};font-display:swap;src:url(data:font/woff;base64,{data}) format('woff')}}")
    return ''.join(faces)


def figures(dense, selected, protocol):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'path', 'axes.titleweight': 'bold'})
    names = [k+'_s11' for k in VARIANTS]
    codes = [LABELS[k][0] for k in VARIANTS]
    colors = ['#94a3b8', '#5985ad', '#138b83', '#c77945', '#555a91']
    results = dense['results']
    output = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.7), layout='constrained')
    for ax, key, title in [(axes[0], 'ap', 'Unseen-region macro AP'), (axes[1], 'event', 'Outages warned at least 3 seconds early')]:
        values = [results[n]['classification']['macro_region_ap'] if key == 'ap' else results[n]['events']['event_recall_at_least_3s'] for n in names]
        ax.bar(codes, values, color=colors, width=.64)
        ax.set_ylim(0, min(1., max(values)*1.3+.02)); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True); ax.set_title(title, pad=14)
        for k, value in enumerate(values):
            ax.text(k, value+.008, f'{value:.3f}', ha='center', weight='bold')
        ax.set_ylabel('AP' if key == 'ap' else 'Event recall')
    output['ablation'] = save_figure(fig, '01_ablation')
    fig, ax = plt.subplots(figsize=(11.4, 3.7), layout='constrained')
    for name, code, color in zip(names, codes, colors):
        history = read_json(RUNS/name/'history.json')
        steps, scores = [r['step'] for r in history], [r['validation_macro_ap'] for r in history]
        ax.plot(steps, scores, marker='o', ms=3, color=color, label=code, lw=2)
        best = int(np.argmax(scores)); ax.scatter(steps[best], scores[best], s=75, facecolors='none', edgecolors=color, lw=2)
    ax.set_title('Validation learning curves — circles mark selected checkpoints', pad=13)
    ax.set_xlabel('Supervised optimizer updates (pretraining shown separately)'); ax.set_ylabel('Mean AP across 7 validation regions')
    ax.grid(alpha=.18); ax.legend(ncol=5, loc='lower right')
    output['learning'] = save_figure(fig, '02_validation_learning')
    regions = protocol['splits']['test']
    values = np.asarray([[results[n]['classification']['by_region'][r]['average_precision'] or 0 for r in regions] for n in names])
    fig, ax = plt.subplots(figsize=(11.4, 3.6), layout='constrained')
    image = ax.imshow(values, cmap='YlGnBu', aspect='auto', vmin=0, vmax=max(.5, values.max()))
    ax.set_xticks(range(len(regions)), [f'R{i+1}' for i in range(len(regions))]); ax.set_yticks(range(len(names)), codes)
    for i in range(len(names)):
        for j in range(len(regions)):
            ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', color='white' if values[i,j] > max(.5, values.max())*.65 else '#142336', weight='bold')
    ax.set_title('AP by unseen region — regional variation remains visible', pad=14)
    fig.colorbar(image, ax=ax, label='AP', fraction=.035)
    output['regions'] = save_figure(fig, '03_region_ap')
    fig, ax = plt.subplots(figsize=(11.4, 3.8), layout='constrained')
    for name, code, color in zip(names, codes, colors):
        metric = results[name]['classification']['all']
        ax.scatter(100*metric['false_positive_rate'], 100*metric['recall'], s=100, color=color)
        ax.annotate(code, (100*metric['false_positive_rate'], 100*metric['recall']), xytext=(6, 6), textcoords='offset points', weight='bold')
    ax.axvline(5, ls='--', color='#d48b32', label='5% budget on validation, not a test guarantee')
    ax.set_xlabel('Actual false positive rate on test (%)'); ax.set_ylabel('Recall on test (%)')
    ax.set_title('One validation-selected alert threshold per run', pad=14); ax.grid(alpha=.18); ax.legend(loc='lower right')
    output['fpr'] = save_figure(fig, '04_fpr_recall')
    fig, ax = plt.subplots(figsize=(11.4, 4.0), layout='constrained')
    baseline = None
    for name, code, color in zip(names, codes, colors):
        y, p = predictions(name)
        precision, recall, _ = precision_recall_curve(y, p)
        take = np.unique(np.linspace(0, len(precision)-1, min(len(precision), 800)).astype(int))
        ax.plot(recall[take], precision[take], lw=2, color=color,
                label=f'{code} pooled AP={results[name]["classification"]["all"]["average_precision"]:.3f}')
        baseline = y.mean()
    ax.axhline(baseline, color='#8d8d8d', ls=':', label=f'Random ranking baseline={baseline:.3f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision–recall curves, full held-out flights', pad=14); ax.legend(loc='upper right'); ax.grid(alpha=.18)
    output['pr'] = save_figure(fig, '05_precision_recall')
    y, p = predictions(selected+'_s11')
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.5), layout='constrained')
    bins = np.minimum((p*10).astype(int), 9)
    xx, yy, count = [], [], []
    for k in range(10):
        take = bins == k
        if np.any(take):
            xx.append(p[take].mean()); yy.append(y[take].mean()); count.append(int(take.sum()))
    axes[0].plot(xx, yy, marker='o', color='#138b83', lw=2); axes[0].plot([0,1], [0,1], ls=':', color='#8792a2')
    axes[0].set(xlim=(0,1), ylim=(0,1), xlabel='Mean predicted risk', ylabel='Actual outage fraction', title='Selected model: calibration diagnostic')
    axes[0].grid(alpha=.18)
    run_names = [selected+'_s'+str(s) for s in (11,22,33)]
    scores = [results[n]['classification']['macro_region_ap'] for n in run_names]
    axes[1].bar(['seed 11','seed 22','seed 33'], scores, color=['#138b83','#5ba9a0','#9bcfc9'], width=.6)
    axes[1].set(ylim=(0, min(1,max(scores)*1.3+.02)), ylabel='Unseen-region macro AP', title='Initialization sensitivity (no ensemble)')
    for k, value in enumerate(scores): axes[1].text(k, value+.008, f'{value:.3f}', ha='center', weight='bold')
    axes[1].grid(axis='y', alpha=.18); axes[1].set_axisbelow(True)
    output['robustness'] = save_figure(fig, '06_calibration_and_seeds')
    return output


def write_csv(dense, protocol):
    path = DOCS/'tables'; path.mkdir(parents=True, exist_ok=True)
    rows, regional = [], []
    for name, result in dense['results'].items():
        m, e = result['classification']['all'], result['events']
        row = {'run': name, 'macro_region_ap': result['classification']['macro_region_ap'],
               'pooled_ap': m['average_precision'], 'positive_fraction': m['positive_fraction'],
               'false_positive_rate': m['false_positive_rate'], 'precision': m['precision'], 'recall': m['recall'],
               'event_recall_1s': e['event_recall_at_least_1s'], 'event_recall_3s': e['event_recall_at_least_3s'],
               'event_observability': e['observability_fraction'], 'brier': m['brier'], 'threshold': result['threshold'],
               'windows': m['samples'], 'outage_events': e['total_new_directed_outage_events_after_lookback']}
        rows.append(row)
        for key, value in result['classification']['by_region'].items():
            regional.append({'run': name, 'region': key, **{k: value[k] for k in ('samples','positives','positive_fraction','average_precision','auroc','false_positive_rate','precision','recall','brier')}})
    for name, data in [('model_comparison.csv', rows), ('region_comparison.csv', regional)]:
        with (path/name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    return rows


def create():
    dense = read_json(RESULTS/'dense_comparison.json')
    bank = read_json(RESULTS/'comparison.json')
    operational = read_json(RESULTS/'operational'/'summary.json')
    audit = read_json(RESULTS/'integrity_audit.json')
    protocol = read_json(ROOT/'configs'/'upgrade_protocol.json')
    training = read_json(ROOT/'configs'/'upgrade_training.json')
    collection = read_json(ROOT/'datasets'/'upgrade45'/'collection_manifest.json')
    posthoc = read_json(RESULTS/'posthoc_diagnostics.json')
    motion_quality = read_json(RESULTS/'motion_diagnostics.json')
    if not dense['complete'] or dense.get('evaluation_version') != EVALUATION_VERSION or bank.get('evaluation_version') != EVALUATION_VERSION:
        raise ValueError('최신 수치 검사로 전체 평가를 완료한 뒤 보고서를 생성하십시오.')
    selected = dense['selection']['selected_variant']
    names = [k+'_s11' for k in VARIANTS]
    chosen = dense['results'][selected+'_s11']
    reference = dense['results']['A0_three_regions_s11']
    metric, events = chosen['classification']['all'], chosen['events']
    baseline_ap, selected_ap = reference['classification']['macro_region_ap'], chosen['classification']['macro_region_ap']
    seed_scores = [dense['results'][selected+'_s'+str(s)]['classification']['macro_region_ap'] for s in (11,22,33)]
    improvement = selected_ap-baseline_ap
    old_pool, new_pool = reference['classification']['all']['average_precision'], metric['average_precision']
    worst_key = min(chosen['classification']['by_region'], key=lambda k: chosen['classification']['by_region'][k]['average_precision'])
    worst_metric = chosen['classification']['by_region'][worst_key]
    worst_name = next(m['name'] for m in protocol['maps'] if m['key'] == worst_key)
    improved_regions = sum(chosen['classification']['by_region'][k]['average_precision'] > reference['classification']['by_region'][k]['average_precision']
                           for k in chosen['classification']['by_region'])
    pooled_note = (f'통합 AP: {old_pool:.3f} → {new_pool:.3f}. 지역 평균 AP와 달리 개선 없음. '
                   '지역·집계 방식 전체의 일관된 개선 미확인. 지역 내부 순위와 지역 간 위험 점수 비교의 구분 필요.'
                   if new_pool <= old_pool else
                   f'통합 AP: {old_pool:.3f} → {new_pool:.3f}. 지역별 편차·오경보율·선행시간의 병행 평가 필요.')
    DOCS.mkdir(parents=True, exist_ok=True)
    charts = figures(dense, selected, protocol)
    csv_rows = write_csv(dense, protocol)
    choice_reason = {
        'A0_three_regions': '3개 지역 기준 설정의 검증 평균 AP 최고. 자료 확대의 일관된 이점 미확인.',
        'A1_more_maps': '지도·고유 기록 확대 설정의 검증 평균 AP 최고. 입력 보강·사전학습의 추가 이점 미확인.',
        'A2_observed_geometry': '위치·속도·지도 기반 추정치를 추가한 지도학습의 검증 평균 AP 최고. 사전학습 미채택.',
        'A3_masked_pretraining': '가림 복원 사전학습 후 미세조정 설정의 검증 평균 AP 최고. 추가 지도학습 대조군과의 비교 결과 반영.',
        'A4_matched_compute': '보강 입력·최대 지도학습 예산 확대 설정의 검증 평균 AP 최고. 가림 복원 사전학습의 추가 이점 미확인에 따른 미채택.',
    }[selected]
    summary = {'selected_variant': selected, 'selected_by': 'validation macro AP, not test', 'primary_cohort': protocol['target_definition'],
               'baseline_macro_ap': baseline_ap, 'selected_macro_ap': selected_ap, 'macro_ap_difference': improvement,
               'selected_seed_macro_ap_mean': float(np.mean(seed_scores)), 'selected_seed_macro_ap_sd': float(np.std(seed_scores, ddof=1)),
               'test_windows': metric['samples'], 'positive_fraction': metric['positive_fraction'],
               'test_actual_fpr': metric['false_positive_rate'], 'test_recall': metric['recall'],
               'baseline_pooled_ap': old_pool, 'selected_pooled_ap': new_pool, 'regions_with_ap_improvement': improved_regions,
               'worst_region': worst_key, 'worst_region_ap': worst_metric['average_precision'], 'worst_region_fpr': worst_metric['false_positive_rate'],
               'event_recall_3s': events['event_recall_at_least_3s'], 'source': 'results/upgrade/dense_comparison.json',
               'not_directly_comparable_to_legacy_ap012': True, 'repositioning_not_executed': True}
    write_json(DOCS/'report_numbers.json', summary)
    rows = []
    for variant in VARIANTS:
        name = variant+'_s11'; r = dense['results'][name]; m = r['classification']['all']; e = r['events']
        run = read_json(RUNS/name/'run.json')
        badge = '<strong class="selected">선정</strong> ' if variant == selected else ''
        rows.append([badge+LABELS[variant][0], esc(LABELS[variant][1]), number(run['validation_macro_ap']),
                     '<b>'+number(r['classification']['macro_region_ap'])+'</b>', number(m['average_precision']),
                     percent(m['false_positive_rate']), percent(m['precision']), percent(m['recall']), percent(e['event_recall_at_least_3s'])])
    comparison_table = table(['설정','비교 목적','검증 지역 평균 AP','시험 지역 평균 AP','시험 통합 AP','실제 FPR','정밀도','재현율','3초 사건 재현율'], rows)
    region_rows = []
    map_by_key = {m['key']: m for m in protocol['maps']}
    for i, key in enumerate(protocol['splits']['test']):
        m = chosen['classification']['by_region'][key]
        region_rows.append([f'R{i+1}', esc(map_by_key[key]['name']), '<code>'+esc(key)+'</code>', str(map_by_key[key]['spatial_group']),
                            percent(m['positive_fraction']), number(m['average_precision']), percent(m['false_positive_rate']), percent(m['recall'])])
    regions_table = table(['표기','지역','키','공간 집단','장애 비율','선정 모델 AP','실제 FPR','재현율'], region_rows)
    record_lookup = {r['id']: r['metadata'] for r in collection['records']}
    weak_rows = []
    for ep in chosen['by_episode']:
        if ep['region'] != worst_key:
            continue
        meta, m = record_lookup[ep['id']], ep['metrics']
        weak_rows.append([esc(meta['direction']), esc(meta['kind']), f'{meta["duration_s"]:.0f}초',
                          '완료' if meta['mission']['mission_completed'] else '제한 시간 / 미완료',
                          f'{m["samples"]:,}', number(m['average_precision']), percent(m['false_positive_rate'])])
    weak_table = table(['비행 방향','간섭','기록 길이','임무','평가 시점','통합 AP','실제 FPR'], weak_rows)
    weak_motion = [p for p in motion_quality['paths'] if p['region'] == worst_key and not p['mission_completed']]
    weak_motion_text = ' '.join(f'{esc(p["direction"])} 방향: 마지막 {p["terminal_window_s"]}초의 3초 갱신 시점에서 군집 배치 {p["unique_joint_configurations_in_terminal_window"]}개 반복.' for p in weak_motion)
    diagnostic_rows = []
    for key, label in [('moving_observer_history','자신이 과거 20초 중 움직인 구간'),
                       ('stationary_observer_history','자신이 과거 20초 내내 정지한 구간'),
                       ('completed_mission_records','완료된 비행 기록'), ('deadline_mission_records','제한 시간 미완료 비행 기록')]:
        a, b = [posthoc['results'][n][key]['all'] for n in ('A0_three_regions_s11', selected+'_s11')]
        diagnostic_rows.append([label, f'{b["samples"]:,}', f'{b["positives"]:,}', percent(b['positive_fraction'],4), number(a['average_precision']), number(b['average_precision']),
                                percent(b['false_positive_rate']), percent(b['recall'])])
    diagnostic_table = table(['사후 진단 집단','시점 수','장애 시점','장애 비율','A0 통합 AP','선정 통합 AP','선정 실제 FPR','선정 재현율'], diagnostic_rows)
    fp_distribution = posthoc['results'][selected+'_s11']['false_alarm_distribution']
    training_rows = []
    for variant in VARIANTS:
        r = read_json(RUNS/(variant+'_s11')/'run.json')
        training_rows.append([LABELS[variant][0], str(r['parameters']), str(len(r['training_regions'])), f'{r["training_samples_per_pass"]:,}',
                              str(r['pretraining_updates']), str(r['supervised_updates']), str(r['best_step']), number(r['validation_macro_ap'])])
    training_table = table(['설정','파라미터 수','학습 지역','한 순회 표본','사전학습 갱신','실제 지도학습 갱신','선택된 갱신','최고 검증 AP'], training_rows)
    seed_table = table(['초기화 시드','시험 지역 평균 AP','시험 실제 FPR','3초 선행 사건 재현율'],
        [[str(s), number(dense['results'][selected+'_s'+str(s)]['classification']['macro_region_ap']),
          percent(dense['results'][selected+'_s'+str(s)]['classification']['all']['false_positive_rate']),
          percent(dense['results'][selected+'_s'+str(s)]['events']['event_recall_at_least_3s'])] for s in (11,22,33)])
    ci_rows = []
    for variant in list(VARIANTS)[1:]:
        ci = dense['paired_cluster_bootstrap'][variant+'_s11']
        ci_rows.append([LABELS[variant][0], number(ci['macro_ap_difference']),
                        '['+', '.join(number(v) for v in ci['cluster_bootstrap_95_interval'])+']'])
    ci_table = table(['A0 대비','지역 평균 AP 차이','공간 집단 재표집 95% 구간'], ci_rows)
    simple_rows = []
    for key, label in [('latest_ack_sinr','마지막 ACK SINR만 이용한 위험 순위'), ('estimated_loss_change','등속 외삽한 지도 손실 변화만 이용한 위험 순위')]:
        m = bank['simple_observation_baselines'][key]['test']
        simple_rows.append([label, number(m['macro_region_ap']), number(m['all']['average_precision']), percent(m['all']['false_positive_rate'])])
    m = bank['results'][selected+'_s11']['test']
    simple_rows.append(['선정 Transformer / 같은 공통 표본', number(m['macro_region_ap']), number(m['all']['average_precision']), percent(m['all']['false_positive_rate'])])
    simple_table = table(['단순 관측 기준과 비교','표본 지역 평균 AP','표본 통합 AP','실제 FPR'], simple_rows)
    feature_rows = []
    for i, (key, korean) in enumerate(zip(BASE_FEATURES+EXTRA_FEATURES, KOREAN_FEATURES)):
        source = ('자기 상태/설정 또는 실제 ACK 관측' if i < 12 else '자기 위치/설정 또는 성공적으로 받은 이웃 데이터와 지도')
        feature_rows.append([str(i+1), esc(korean), '<code>'+key+'</code>', esc(source)])
    feature_table = table(['번호','입력 항목','코드 이름','정보 출처'], feature_rows)
    all_maps_rows = [[esc(m['key']), esc(m['name']), esc(m['split']), str(m['spatial_group']), str(m['buildings']),
                      f'{m["known_heights"]}/{m["buildings"]}', str(m['imputed_heights'])] for m in protocol['maps']]
    map_table = table(['지도 키','지역','분할','공간 집단','건물','실제 높이 자료','높이 보간'], all_maps_rows)
    slice_rows = []
    for key, value in bank['results'][selected+'_s11']['test_slices'].items():
        if 'samples' in value:
            slice_rows.append([esc(key), f'{value["samples"]:,}', percent(value['positive_fraction']), number(value['average_precision']),
                               percent(value['false_positive_rate']), percent(value['recall'])])
    slices_table = table(['구간','고정 표본 수','장애 비율','AP','실제 FPR','재현율'], slice_rows)
    operational_rows = []
    for key, label in [('currently_truly_available_reference','주 평가: 잠재 채널 기준 현재 가용'),
                       ('all_observed_pairs','운용 진단: 관측 이력이 있는 모든 쌍'),
                       ('recent_ack_within_1s','운용 진단: 최근 1초 ACK 성공이 있는 쌍')]:
        q = operational[key]['all']
        operational_rows.append([label, f'{q["samples"]:,}', percent(q['positive_fraction']), number(q['average_precision']),
                                 percent(q['false_positive_rate']), percent(q['recall'])])
    operational_table = table(['평가 집단','시점 수','장애 비율','통합 AP','실제 FPR','재현율'], operational_rows)
    event_table = table(['항목','실제 결과'], [
        ['20초 이력 이후 새로 발생한 방향별 지속 단절', f'{events["total_new_directed_outage_events_after_lookback"]:,}건'],
        ['사전 경고를 평가할 관측 창이 있는 사건', f'{events["observable_events"]:,}건'],
        ['관측 가능한 경고 창이 없는 사건', f'{events["without_observed_warning_window"]:,}건'],
        ['경고 창 관측 가능 비율', percent(events['observability_fraction'])],
        ['1초 이상 미리 경고한 사건 비율 / 관측 가능 사건', percent(events['event_recall_at_least_1s'])],
        ['3초 이상 미리 경고한 사건 비율 / 관측 가능 사건', percent(events['event_recall_at_least_3s'])],
        ['3초 이상 미리 경고한 사건 비율 / 모든 새 사건', percent(events['all_event_recall_at_least_3s'])],
        ['3초 경고가 원천적으로 가능한 관측 사건 비율', percent(events['maximum_possible_observable_3s_recall'])],
        ['감지한 사건의 선행시간 중앙값', number(events['detected_event_lead_median_s'],2)+'초'],
    ])
    total_hours = sum(r['metadata']['duration_s'] for r in collection['records'])/3600
    payload = {'summary': summary, 'comparison': csv_rows, 'selected': selected, 'labels': LABELS,
               'region_metrics': {n: r['classification']['by_region'] for n, r in dense['results'].items()},
               'map_names': {k: v['name'] for k,v in map_by_key.items()}}
    payload_json = json.dumps(payload, ensure_ascii=False).replace('<', r'\u003c')
    font_css = embedded_fonts()
    font_license_path = DOCS/'assets'/'OFL-NotoSansKR.txt'
    font_license = ('<script id="font-license" type="application/json">'+json.dumps(font_license_path.read_text(), ensure_ascii=False).replace('<', r'\u003c')+'</script>') if font_license_path.exists() else ''
    body = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>45개 지도 Transformer 예측 개선 실험</title>
<style>
{font_css}
:root{{--navy:#142e46;--teal:#0f827b;--paper:#f4f7fa;--line:#dce5ec;--muted:#5a6b7b;--orange:#a45a1b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:#203347;font-family:CapstoneSans,Pretendard,'Noto Sans KR','Malgun Gothic',system-ui,sans-serif;font-size:15px;line-height:1.8}}
header{{background:var(--navy);color:white;padding:54px max(24px,calc((100vw - 1180px)/2)) 44px}}.eyebrow{{font-size:12px;letter-spacing:2px;color:#87d1ca}}h1{{font-size:34px;line-height:1.4;margin:12px 0 16px;letter-spacing:-1px}}header p{{color:#d0dce5;max-width:960px;margin:0}}.pill{{display:inline-block;border:1px solid #7795a8;border-radius:30px;padding:3px 12px;font-size:12px;margin:18px 5px 0 0}}
nav{{position:sticky;top:0;background:#ffffffed;backdrop-filter:blur(12px);z-index:10;border-bottom:1px solid var(--line);padding:10px 24px;display:flex;justify-content:center;gap:20px;overflow:auto}}nav a{{color:var(--navy);white-space:nowrap;text-decoration:none;font-size:13px;font-weight:700}}
main{{max-width:1236px;padding:26px 28px 80px;margin:auto}}section{{scroll-margin-top:66px;background:white;border:1px solid var(--line);border-radius:14px;padding:28px;margin:22px 0;box-shadow:0 4px 16px #183e5c05}}h2{{font-size:23px;color:var(--navy);letter-spacing:-.5px;margin:0 0 18px}}h3{{font-size:17px;margin:25px 0 10px}}p{{margin:10px 0 14px}}.lead{{font-size:17px;line-height:1.9}}.muted,figcaption{{color:var(--muted);font-size:13px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.card{{padding:18px;border-radius:9px;background:#eef5f7;border:1px solid #d4e7e8}}.card .value{{display:block;font-size:29px;font-weight:800;color:var(--teal);line-height:1.3}}.card .label{{font-size:12px;color:var(--muted)}}.note{{border-left:4px solid var(--teal);background:#eff8f6;padding:13px 17px;border-radius:0 7px 7px 0;margin:16px 0}}.caution{{border-left-color:#c58c48;background:#fff8ee}}.selected{{background:#d9f1e9;color:#176d55;padding:3px 7px;border-radius:4px;font-size:12px}}.table-scroll{{overflow:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;min-width:650px;font-size:13px;line-height:1.6}}th{{text-align:left;background:#eef3f7;color:#304c62;font-size:12px;white-space:nowrap;padding:12px 11px;border-bottom:2px solid #cddbe5}}td{{padding:12px 11px;vertical-align:top;border-bottom:1px solid #e6edf2}}tr:hover td{{background:#f6fafb}}td code{{font-size:11px;overflow-wrap:anywhere}}figure{{margin:22px 0;padding:16px;border:1px solid #e1e9ee;border-radius:9px}}figure svg{{max-width:100%;height:auto;display:block}}figcaption{{margin-top:8px}}code,pre{{font-family:Consolas,'DejaVu Sans Mono',monospace}}code{{font-size:13px}}pre{{background:#122a40;color:#e2eff5;padding:18px;overflow:auto;border-radius:8px;font-size:12px;line-height:1.8}}details{{border:1px solid var(--line);border-radius:7px;padding:13px 16px;margin:14px 0}}summary{{cursor:pointer;font-weight:700;color:var(--navy)}}select{{padding:9px 12px;border:1px solid #b9ccd9;border-radius:6px;font:inherit;background:white}}.small{{font-size:13px}}.steps{{counter-reset:steps;list-style:none;padding:0}}.steps li{{counter-increment:steps;position:relative;padding:10px 0 10px 40px}}.steps li::before{{content:counter(steps);position:absolute;left:0;background:#e2f0ee;color:#126d65;border-radius:50%;width:27px;height:27px;text-align:center;line-height:27px;font-weight:700}}a{{color:#126e93}}.qa dt{{font-weight:800;color:var(--navy);margin-top:16px}}.qa dd{{margin:4px 0 14px}}footer{{text-align:center;color:var(--muted);font-size:12px;padding:25px}}@media(max-width:760px){{header{{padding:32px 22px}}h1{{font-size:26px}}main{{padding:12px}}section{{padding:19px}}.cards{{grid-template-columns:repeat(2,1fr)}}nav{{justify-content:flex-start;gap:15px}}}}@media print{{nav{{display:none}}body{{background:white;font-size:11px}}header{{padding:25px}}main{{padding:0}}section{{box-shadow:none;border-radius:0;break-inside:auto}}h2,h3{{break-after:avoid}}figure,tr,.card{{break-inside:avoid}}details{{display:block}}table{{min-width:0;font-size:9px}}}}
</style></head><body>
<header><div class="eyebrow">CAPSTONE · PREDICTION IMPROVEMENT</div><h1>45개 지도 기반<br>Transformer 통신 장애 예측 실험</h1><p>사전 검증: 지도·관측·정답.<br>비교 항목: 데이터 확대·입력 보강·사전학습.<br>평가 범위: 예측 성능. 재배치 성능 미포함.</p><span class="pill">45개 지도 / 270개 비행</span><span class="pill">학습 30 · 검증 7 · 시험 8</span><span class="pill">예측 모델 계열 1개</span><span class="pill">지휘소 없는 드론 간 통신</span></header>
<nav><a href="#decision">선정 결과</a><a href="#data">지도 사용 방법</a><a href="#method">예측 방법</a><a href="#results">평가 결과</a><a href="#events">선행 경고</a><a href="#operation">운용 진단</a><a href="#limits">한계·향후 계획</a><a href="#guide">기타 설명</a></nav><main>

<section id="decision"><h2>1. 모델 선정 결과</h2>
<p class="lead"><strong>선정 설정: {LABELS[selected][0]} — {esc(LABELS[selected][1])}</strong><br>선정 기준: 검증 지역 평균 AP.<br>{choice_reason}</p>
<div class="cards"><div class="card"><span class="value">{selected_ap:.3f}</span><span class="label">시험 지역 평균 AP</span></div><div class="card"><span class="value">{improvement:+.3f}</span><span class="label">동일 시험 A0 대비 AP 차이</span></div><div class="card"><span class="value">{percent(metric['false_positive_rate'])}</span><span class="label">실제 시험 오경보율</span></div><div class="card"><span class="value">{percent(events['event_recall_at_least_3s'])}</span><span class="label">관측 가능 사건의 3초 선행 경고율</span></div></div>
<p>A0 시험 지역 평균 AP: <b>{baseline_ap:.3f}</b>.<br>선정 설정의 초기화 시드 3개 평균: <b>{np.mean(seed_scores):.3f} ± {np.std(seed_scores,ddof=1):.3f}</b>(표본 표준편차).<br>시드별 단일 모델 평가. 앙상블 미적용.</p>
<p><b>{pooled_note}</b></p>
<div class="note caution"><b>미해결 사항</b><br>A0 대비 AP 개선 지도: 8개 중 {improved_regions}개.<br>최저 AP 지역: {esc(worst_name)}(<code>{esc(worst_key)}</code>).<br>해당 지역 AP: {worst_metric['average_precision']:.3f}. 실제 FPR: {percent(worst_metric['false_positive_rate'])}.<br>전체 경고 정밀도: {percent(metric['precision'])}. 전 지역의 안정적 운용 보장 불가.</div>
<div class="note caution"><b>비교 기준</b><br>기존 약 0.12와 지도 분할·관측 창·평가 집단 상이. 직접 개선율 비교 제외.<br>비교 대상: 동일 시험 기록의 A0.<br>AP와 정확도의 구분 필요.</div></section>

<section id="data"><h2>2. 지도 사용 방법</h2>
<p>지도 수: 기존 5개 + 추가 ZIP 두 묶음의 40개 = 총 <b>45개</b>.<br>사용 정보: 로컬 미터 좌표, 건물 폴리곤·높이, 비행 금지 구역.<br>원본 형식·기하·보간 표시 변경 없음.</p>
{table(['항목','확정한 조건'], [['공간·배치','400m × 400m, 평탄 지면 기준 고도 50m, 전체 8대·수색 3대·중계 5대·수색 V 대형·간격 40m'],['학습/검증/시험','30 / 7 / 8개 지도. 기존 결과를 이미 본 원래 5개 지도는 모두 학습 집단.'],['공간 누출 방지','공유 건물 ID, 양의 면적 중첩, 경계 간격 100m 미만을 같은 집단으로 묶음. 시험 8개 지도는 6개 공간 집단.'],['비행 변형','동서/남북 두 방향 × 간섭 없음/연속/간헐 세 조건 = 지도당 6개 기록.'],['기록 시간',f'총 {total_hours:.2f}시간의 시뮬레이션 시간. 임무 완료 또는 7200초까지. 실제 계산 시간과 다름.'],['미완료',f'90개 고유 경로 중 {audit["completed_paths"]}개 완료, {audit["mission_deadline_paths"]}개 제한 시간 도달. 간섭 조건별 복제 기록에서 57개 비행이 미완료.'],['실제 수행한 검사',f'45개 지도·270개 기록 해시 일치, 독립 정답 재계산 {audit["independent_label_checks"]:,}개 일치, 경로 장애물 교차 0, 연속 이동 최소 간격 {audit["minimum_continuous_separation_m"]:.2f}m.']])}
<p>표본 추출: 지역별 동일 최대 개수의 시간 창. 정답과 무관한 추출.<br>학습: 지도당 최대 2,048개, 총 61,440개.<br>검증: 지도당 최대 4,096개, 총 28,672개.<br>최종 주 시험: 평가 가능한 시간 창 <b>{metric['samples']:,}개</b> 전체 사용. 표본 추출 없음.<br>전처리 평균·표준편차: 학습 관측에서만 산출.</p>
<details><summary>45개 지도 분할 및 높이 자료</summary>{map_table}<p class="small">높이 원자료 누락 건물: 지역 평균 기반 보간.<br>실제 지형 고도·실측 전파 지도 미포함.</p></details></section>

<section id="method"><h2>3. 예측 문제 및 개선 방법</h2>
<p>평가 대상: <b>현재 실제 가용한 링크</b>.<br>입력: 과거 20초, 0.5초 간격의 40구간.<br>예측 대상: <b>향후 5초 내 1초 이상 지속되는 양방향 불가용</b>.<br>정답 생성 범위: 입력 종료 다음 구간부터.</p>
<p>링크 가용 조건: 데이터·역방향 ACK를 포함한 홉 성공 확률이 양방향 모두 0.9 이상.<br>시간 구간의 패킷 전달률 90% 및 전체 군집 연결률 90%와 구분.<br>현재 가용 정답의 사용 범위: 오프라인 학습·평가 집단 구성에 한정. 신경망 입력 미포함.</p>
{table(['단계','도입한 이유','구현과 비교'], [['A0 → A1: 자료 확대','기존 소수 지역의 전파·이동 패턴에만 맞춰지는 위험을 줄이기 위해','같은 Transformer에서 3개 → 30개 지역. 지도와 고유 기록의 확대 효과를 함께 비교.'],['A1 → A2: 관측 보강','신호 값만으로는 왜 악화되는지 알기 어려워 위치·상대 운동·차폐 추세를 제공','기본 12항목 → 28항목. 실제 수신한 이웃 상태와 알려진 지도만 사용.'],['A2 → A3: 가림 복원 사전학습','라벨을 사용하기 전에 과거 기록의 상관관계를 학습하는 접근을 검증','TST의 원리를 변형 적용. 관측값 15%, 평균 3구간 가림. 원래 관측됐던 가림 값에만 복원 MSE.'],['A3 ↔ A4: 추가 계산 통제','사전학습과 단순히 학습을 더 수행한 효과를 구분','A3 최대 500+2000회, A4 최대 2500회. 실제 횟수는 동일한 검증 중단 규칙에 따라 달라짐.']])}
<p>공통 모델 구조: <b>은닉 64·헤드 4·인코더 2층·피드포워드 128·드롭아웃 0.1</b>.<br>구성 요소: 입력 투영, 관측·가림 마스크, 시간 위치 표현.<br>출력 구성: 마지막 시점과 평균 인코딩 결합.<br>원 TST 구조의 변형 적용.</p>
<div class="note"><b>미래 정보 유출 방지</b><br>이웃 위치·속도: 수신 성공 데이터의 기존 64B 헤더 중 32B에서 취득.<br>ACK 단독 수신에 의한 상대 위치 취득 없음.<br>마지막 수신 상태: 최대 10초 외삽.<br>5초 후 손실: <em>등속 추정치</em> 사용.<br>실제 미래 위치·채널 확률·간섭 시작 시각: 입력 제외.</div>
<details><summary>28개 입력 항목 및 정보 출처</summary>{feature_table}</details>
{training_table}
<p class="muted">최대 예산·검증 규칙 동일 적용. 조기 종료에 따른 실제 갱신 횟수 차이 존재.<br>A3: 지도학습 외 사전학습 500회 포함.<br>A2·A4: 최대 갱신 수와 코사인 학습률 감쇠 일정 동시 변경. 갱신 횟수의 단독 효과 해석 불가.</p>
<figure>{charts['learning']}<figcaption>체크포인트 선택 기준: 검증 지역 평균 AP. 검증 성능 정체 시 조기 종료 적용.</figcaption></figure></section>

<section id="results"><h2>4. 미학습 8개 지역 평가 결과</h2>
<p>비교 조건: 48개 시험 비행의 단절 정의·시간 창 동일 적용.<br>경고 임계값: 검증 음성 표본의 FPR 5% 이하 조건으로 설정. 시험 중 변경 없음.</p>
{comparison_table}
<div class="note caution"><b>FPR 기준</b><br>검증 경고 예산: 프로젝트 설정값 5%. 논문·통신 표준의 보편적 성공 기준 아님.<br>시험 지역의 실제 FPR 5% 보장 없음.<br>시험 장애 비율 및 무작위 위험 순위의 AP 기준: {percent(metric['positive_fraction'])}.</div>
<figure>{charts['ablation']}<figcaption>AP와 3초 선행 경고의 개별 평가. 종합 점수 미사용.</figcaption></figure>
<figure>{charts['regions']}<figcaption>R1~R8의 지역·장애 비율: 하단 표 참조. 지역 평균 AP: 지역별 AP의 단순 평균. 통합 AP: 전체 시간 창 기준 산출.</figcaption></figure>
{regions_table}
<h3>최저 성능 지역의 비행별 분석</h3>
<p>분석 지역: {esc(worst_name)}.<br>분석 단위: 6개 비행 기록.<br>저성능 기록도 주 시험에 포함. 기록 제외에 의한 점수 보정 없음.</p>
{weak_table}
<div class="note caution"><b>경로 반복 검사</b><br>{weak_motion_text}<br>전체 미완료 경로 19개 중 {motion_quality['incomplete_terminal_one_or_two_state_repeats']}개: 마지막 1200초 동안 1~2개 배치 반복.<br>후속 점검 대상: 반복 경로의 등속 외삽 오차 및 이동 엔진.<br>경로 반복과 오경보의 인과관계: 미검증.</div>
<details><summary>이동·대기·임무 미완료별 사후 진단</summary>
<p>분석 시점: 주 평가 이후.<br>평가 집단·가중치·경고 임계값 변경 없음.<br>임무 미완료와 오경보의 인과관계: 미검증.</p>
{diagnostic_table}
<p>자기 드론의 20초 정지 집단: 장애 정답 6개. 극소수 양성에 따른 AP 변동 민감성 존재.<br>선정 모델 오경보 중 자기 드론의 20초 정지 구간 비율: {percent(fp_distribution['fp_on_stationary_observer_history_fraction'])}.<br>선정 모델 오경보 중 제한 시간 미완료 기록 비율: {percent(fp_distribution['fp_on_deadline_missions_fraction'])}.<br>우선 점검 대상: 경로 반복·대기 분포 및 신호·지도 입력의 불일치.</p></details>
<figure>{charts['fpr']}<figcaption>실제 FPR·장애 재현율 병행 비교. AP 단독의 경고 운용 성능 판단 제외.</figcaption></figure>
<figure>{charts['pr']}<figcaption>전체 예측의 통합 AP 기준 곡선. 주 표의 지역 평균 AP와 집계 방식 상이. 표시 점만 축소.</figcaption></figure>
<details><summary>지역 불확실성 및 초기화 시드 민감도</summary>
{ci_table}
<p>시험 지도 8개: 공유 건물 기준 6개 공간 집단.<br>재표집 단위: 공간 집단.<br>중첩된 113만 시간 창의 독립 실험 113만 회 취급 제외.</p>
{seed_table}
<p>초기화 민감도 평가: 선정 설정의 3개 시드.<br>전체 비교군의 3시드 반복: 미실시.</p></details>
<figure>{charts['robustness']}<figcaption>위험 순위 성능과 확률 보정의 구분. 보정 진단 및 시드별 결과 병행 제시.</figcaption></figure>
<h3>구간별 성능 분석</h3>
<p>추가 진단 표본: 지역당 최대 8,192개의 공통 고정 시험 표본.<br>전체 시험과 표본 수 상이.<br>이동 구간 정의: 관측 주체의 과거 창 내 0.25m/s 이상 이동. 상대 드론의 정지 여부와 별도 구분.</p>
{slices_table}
<p class="muted">간섭 시작 시각: 사후 진단에만 사용.<br>전조 없는 무작위 간섭의 사전 완전 예측 가정 없음.</p>
<details><summary>단순 관측 기준과의 비교</summary>
<p>비교 조건: 동일 고정 시험 표본.<br>결측 점수: 정규화 평균으로 대체.<br>비교 기준: 단조 변환한 위험 순위. 추가 딥러닝 모델 미사용.<br>통합 AP·지역 평균 AP 병행 평가.</p>
{simple_table}</details></section>

<section id="events"><h2>5. 단절 사건의 선행 경고 성능</h2>
<p>평가 목적: <b>재배치 대응 시간</b> 확인.<br>AP 외 추가 지표: 단절 사건별 1초·3초 이상 선행 경고 비율.</p>
{event_table}
<p>집계 단위: 관측 주체별 방향성 링크 사건.<br>동일 물리 링크의 양방향 사건: 비독립.<br>새 단절 평가 제외: 기록 시작부터 지속된 단절 및 과거 창 20초 확보 이전 사건.<br>관측 가능한 경고 창 누락 사건: 제외 수 및 전체 사건 대비 재현율 병기.</p>
<p>전체 사건에 실제 전송 경로 미사용 쌍 포함.<br>후속 평가 과제: 활성 경로·핵심 링크 정의 및 주기적 상태·링크 관측 필요성 검토.<br>비콘 도입 시 TDMA 슬롯·전송 비용 반영 필요.</p>
<div class="note caution"><b>재배치 성공률과의 구분</b><br>후속 제어 실험의 평가 대상: 3초의 이동 시간 충분성 및 이동 후 군집 연결 유지 여부.</div></section>

<section id="operation"><h2>6. 온라인 추론 및 운용 진단</h2>
<p>학습: PC 기반 오프라인 수행.<br>실행: 저장 가중치 기반 추론.<br><code>OnlineLinkPredictor</code> 입력: 과거 관측값·마스크. 출력: 방향별 위험 행렬.<br>추론 시 실제 현재 채널 정답 미사용.<br>최근 ACK 기준의 추가 운용 진단: 동일 가중치 적용.</p>
{operational_table}
<p>모든 관측 쌍·최근 ACK 집단에 이미 진행 중인 단절 포함 가능.<br>해당 AP와 주 평가의 <em>새 단절 사전 예측 AP</em> 혼용 금지.<br>최근 1초 ACK 성공에도 잠재 채널 정답이 불가용인 비율: {percent(operational['recent_ack_but_latently_unavailable_fraction'])}.</p>
<p>동일 입력의 저장 창·온라인 API 위험도 일치 확인.<br>측정 조건: CPU 2스레드, 드론 8대 행렬 중 관측 가능 {operational['latency_evaluated_pair_count']}개 쌍.<br>평균 계산 시간: <b>{operational['cpu_matrix_prediction_ms']:.2f}ms</b>.<br>지도 특징 계산·시뮬레이터 시간 제외.</p>
<pre>from upgrade.inference import OnlineLinkPredictor
predictor = OnlineLinkPredictor()
# 새 비행·지역 시작 시 predictor.reset() 호출
answer = predictor.observe(time_end_s, values_8x8x28, observed_mask)
# 첫 20초: None. 이후 risk, warning, observed_history 반환
# 관측 부족: NaN. 안전 상태로 간주 금지. 이동 명령 생성 미포함.</pre>
<h3>통신 모델 구성</h3>
{table(['구성','구현과 역할'],[['전파 손실','ITU 자유공간 손실 + 건물 지붕 프로파일의 Bullington 근사. 평탄 지면 부분 구현이며 P.526 전체를 구현한 것은 아님.'],['확률적 변동','시간 상관 섀도잉 표준편차 1.5dB, 상관 시간 4초. 미래 난수 실현값은 입력하지 않음.'],['무선 설정','2.4GHz / 1MHz / 10dBm / 1Mbps / 잡음지수 7dB. 데이터 576B, ACK 14B.'],['외부 간섭','높이 2m 간섭원. 비행별 위치 무작위, 활성 시 출력 −3~3dBm. 간헐 시작은 15~45%, 종료는 65~90% 비행 시간 범위.'],['패킷 성공','BPSK 비트 오류 확률에서 데이터와 역방향 ACK 성공을 계산. 실제 송신 성공 기록은 관측, 잠재 성공 확률은 정답에만 사용.'],['MAC·라우팅','이상적 TDMA 5ms 슬롯, 전체 정보 기반 ETX형 경로. 정보 공유 제어 비용은 미반영. AODV/OLSR 구현을 의미하지 않음.'],['적용하지 않은 것','3GPP TR 36.777 계수와 ns-3는 구현에 포함하지 않음. 이번에는 채널 수식을 바꾸지 않고 예측 개선 효과를 비교.']])}
</section>

<section id="limits"><h2>7. 한계 및 향후 계획</h2>
<p>{choice_reason}<br>검증 결과 기준의 모델 선정 고정. 시험 순위에 따른 재선정 없음.<br>모델 복잡도에 따른 일반적 우열 판단 제외.</p>
<ul>
<li>실측 전파·실물 드론 검증 없음. 건물 높이 일부 보간. 기하 추정과 정답 전파 모델의 시뮬레이터 가정 공유.</li>
<li>고정 조건: 고도 50m·8대·중계 5대·V·40m. 새 지역 일반화 평가에 한정. 모든 대형·고도·드론 수의 범용성 미검증.</li>
<li>미완료 경로 19개도 평가에 포함. 이동 엔진 변경 없음. 경로 반복·정체 개선 필요.</li>
<li>현재 가용 집단: 잠재 채널 정답을 사용한 사후 조건. 실제 배포 경고의 관측 기준 평가 필요.</li>
<li>제한 조건: 소수 공간 집단, 유한한 학습 예산, 선정 설정만 3시드 반복. 사전학습 500회 결과의 전체 사전학습 방식 최대 성능 일반화 불가.</li>
<li>고정 이동 방식 기반 기록 생성. 재배치 정책 적용 시 입력 분포 변화 가능. 해당 정책 기록의 추가 수집·재평가 필요.</li>
<li>모델 출력: 위험도. 지역별 경고 기준 변화, 확률 보정, 예측 선행시간의 병행 확인 필요.</li>
</ul>
<ol class="steps">
<li><b>경로 반복 수정.</b> 새 환경의 임무 완료 여부·안전·불필요한 반복 점검. 기존 기준 결과 보존 및 전 비교군 기록 재생성. 기록 삭제에 의한 점수 보정 제외.</li>
<li><b>입력 신뢰성·관측 범위 점검.</b> 검증 지역의 등속 외삽 오차, 오래된 이웃 상태, 핵심 링크 관측 부족 분석. 현재 시험 지역 확인 후 수정 방향 결정 시 새 독립 지역의 최종 평가 필요.</li>
<li><b>고정 예측기와 동일 재배치 규칙 비교.</b> 위험도 사용·미사용 조건 비교. 연결 유지율·단절 시간·패킷 전달률·수색 완료율·이동량 개별 평가.</li>
<li><b>PPO 등 학습 제어기 연결.</b> 현재 관측·방향별 위험 입력. 실행 시 실제 수신 정보만 사용. 실행 단계의 안전 제약 적용.</li>
<li><b>E2E 공동학습 검토.</b> Transformer 인코더의 예측 헤드·정책 헤드 공유. 정책 손실과 보조 예측 손실 결합. 비미분 Python 시뮬레이터에 대한 직접 역전파 대신 정책 경사 기반 행동 효과 학습.</li>
</ol></section>

<section id="guide"><h2>8. 기타 설명</h2>
<dl class="qa">
<dt>Transformer 선정 근거</dt><dd>과거 기록 복원 사전학습 및 예측·정책 인코더 공유 검토에 적합한 계열. GRU·TCN 대비 일반적 우월성 미입증.</dd>
<dt>비교 조건 및 모델 규모</dt><dd>A0~A4 동일 백본 적용. 자료 확대·관측 보강·사전학습·추가 지도학습의 효과 비교. 이전 소형 모델 대비 규모 효과는 본 검증 대상에서 제외.</dd>
<dt>입력 정보와 정답의 분리</dt><dd>자기 상태·수신 이웃 상태만 입력. 5초 후 상태는 등속 외삽. 실제 미래 궤적·채널 확률은 별도 정답 파일 저장. 미래 변경 시 과거 입력 불변 검사 통과.</dd>
<dt>사전학습 채택 여부</dt><dd>{choice_reason}</dd>
<dt>예측 성능과 임무 성과의 구분</dt><dd>현재 결과: 예측 단계에 한정. 재배치 후 연결 유지율·수색 성과 별도 검증 필요. AP·오경보율·선행시간 개별 평가.</dd>
<dt>학습 환경과 분산 실행</dt><dd>한 PC의 시뮬레이션 기록으로 학습 후 드론별 관측에 적용. 실행 중 지상 지휘소 미요구. 연합학습 미적용.</dd>
</dl>
<details><summary>주요 용어</summary>
{table(['용어','간결한 설명'],[['AP','장애 위험이 높은 사례를 앞에 놓는 능력을 정밀도–재현율 관점에서 요약한 값. 정확도와 다름.'],['정밀도 Precision','경고한 사례 중 실제 향후 장애가 있었던 비율.'],['재현율 Recall','실제 향후 장애 사례 중 경고한 비율.'],['FPR','향후 장애가 없던 사례 중 잘못 경고한 비율. 1−정밀도와 다름.'],['선행시간','단절 시작 시각 − 첫 유효 경고 시각. 3초 이상이면 3초 선행 사건 감지.'],['지역 평균 AP','각 지역 AP를 계산한 뒤 평균. 긴 비행 지역의 표본 수가 평균을 지배하지 않게 함.'],['통합 AP','모든 평가 시점을 합쳐 계산. 지역 평균 AP와 수치가 다를 수 있음.'],['지도학습','미래 정답과 예측 오차로 가중치를 학습. 이번 최종 예측 과제의 학습 방법.'],['자기지도 사전학습','관측 일부를 가리고 원래 과거 관측을 복원. 미래 정답을 입력하지 않음.'],['E2E','예측 표현과 행동 정책이 최종 통신·임무 목적을 향해 공동 학습하도록 연결하는 후속 설계. 단순 파일 전달만으로 공동학습이 된 것은 아님.'],['LoS/NLoS','건물에 의해 직접 경로가 가려지는지 구분. 이 기하 판정 자체가 3GPP 모델 채택을 뜻하지 않음.'],['섀도잉','장애물/주변 환경에 따른 느린 수신 신호 변동의 확률 모사. 여기서는 시간 상관을 부여.'],['SINR','원하는 신호 전력 ÷ (간섭+잡음 전력). 전력을 더할 때 dB 값을 직접 더하지 않음.'],['TDMA','드론이 정해진 시간 슬롯에 번갈아 송신하는 방식. 여기서는 이상적 동기화 가정.'],['ETX형 경로','홉의 기대 전송 비용을 더해 비용이 작은 경로를 선택하는 방식.']])}
</details>
<h3>실행 방법 및 결과 파일</h3>
<p>Windows 설치·단계별 명령: <code>README_UPGRADE.md</code> 참조.<br>현재 실행 설정: <code>configs/upgrade_*.json</code>.<br>기존 01~13번 배치 파일·이전 forecast 설정: 이번 수치의 실행 설정에서 제외.</p>
<pre>python -m upgrade.cli collect --workers 3
python -m upgrade.cli audit
python -m upgrade.cli prepare
python -m upgrade.cli train --workers 2
python -m upgrade.cli evaluate
python -m upgrade.cli report</pre>
<p>전체 시험 원시 수치: <code>results/upgrade/dense_comparison.json</code>.<br>표 계산용 파일: <code>docs/tables/model_comparison.csv</code>, <code>region_comparison.csv</code>.<br>실제 가중치: <code>results/upgrade/runs/</code>.<br>재현 조건: 제공 버전·시드 사용. 운영체제에 따른 미세 수치 차이 가능.</p></section>

<section id="sources"><h2>9. 참고자료</h2>
<ol>
<li>Zerveas et al., <a href="https://arxiv.org/abs/2010.02803">A Transformer-based Framework for Multivariate Time Series Representation Learning</a>, KDD 2021 / arXiv:2010.02803.<br>적용 근거: 본문 §3.3의 연속 가림 복원 및 사전학습→미세조정 원리.<br>변형 적용값: 가림 15%·평균 3구간. UAV 최적값 검증 없음.</li>
<li>Ohta et al., <a href="https://arxiv.org/abs/2301.00752">Point Cloud-based Proactive Link Quality Prediction for Millimeter-wave Communications</a>, IEEE TMLCN 2023.<br>참고 범위: 공간·운동 단서에 의한 차폐 예측 동기.<br>원 논문 조건: 60GHz 실내. 본 과제의 2.4GHz 성능 근거로 직접 전용 제외. 포인트클라우드 모델 미구현.</li>
<li>ITU-R <a href="https://www.itu.int/rec/R-REC-P.525/en">P.525</a>: 자유공간 전파 감쇠. <a href="https://www.itu.int/rec/R-REC-P.526/en">P.526</a>: 회절.<br>적용 범위: 기존 엔진의 부분 근사 구현. 표준 전체 적합성·실측 정확도 검증 미수행.</li>
<li>scikit-learn 공식 문서: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html">average_precision_score</a>, <a href="https://scikit-learn.org/stable/modules/cross_validation.html">그룹 기반 교차검증</a>.<br>AP 산출: 라이브러리 함수 사용. 공간 분할: 공유 건물·인접성을 반영한 프로젝트 규칙.</li>
<li>지도 출처: 팀원 제공 VWorld 건물 지도 ZIP 두 묶음 및 기존 5개 지도.<br>원자료 날짜·출처·높이 보간 표시: 지도 JSON과 quality/license 파일에 유지.</li>
</ol>
<p>논문 활용 범위: 방법의 동기·원리 근거.<br>본 조건의 성능 개선 폭: 실제 실험 결과 기준 판단.</p>
</section>
{font_license}</main>
<script id="report-data" type="application/json">{payload_json}</script>
<script>document.querySelectorAll('details').forEach(el=>el.addEventListener('toggle',()=>{{el.dataset.open=el.open?'true':'false'}}));</script>
</body></html>'''
    path = DOCS/'Transformer_개선실험_보고서.html'
    path.write_text(body, encoding='utf-8')
    print(json.dumps({'report': str(path), 'selected': selected, 'test_macro_ap': selected_ap, 'baseline_macro_ap': baseline_ap}, ensure_ascii=False), flush=True)
    return path


if __name__ == '__main__':
    create()
