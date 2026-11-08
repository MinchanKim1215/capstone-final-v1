"""Audit the frozen study and report prediction and prevention separately."""
from collections import Counter

import numpy as np

from . import ARTIFACTS, ROOT
from .protocol import CAP, V20_THRESHOLD, now, read, sha, write
from .training import load, metrics, ranking, restore_model, scores_for
from forecast_prevention.diagnostics import connected, outage_runs
from forecast_prevention.evaluation import aggregate, case_key, comparison, match, shared_prefix
from forecast_shield.evaluate import paired


def read_checked(item):
    if sha(item['path'])!=item['sha256']:
        raise ValueError('Paired result metadata changed')
    row = read(item['path'])
    if sha(row['archive_path'])!=row['archive_sha256'] or not row['audited']['passed']:
        raise ValueError('Paired flight changed or lacks a safety audit')
    with np.load(row['archive_path'],allow_pickle=False) as a:
        healthy = connected(a['links'],row['search_drones'])
        np.testing.assert_array_equal(healthy,a['network'])
        runs = outage_runs(healthy)
        if sum(start>0 for start,_ in runs)!=row['persistent_onsets']:
            raise ValueError('Physical event count differs from the recorded cut count')
    return row


def predict(flights, freeze, setting):
    metadata = freeze['models'][setting['model']]
    model = restore_model(metadata)
    return scores_for(model,setting,flights)


def compare_scores(flights, baseline, newer, threshold):
    old = [np.where(np.isfinite(score),score>=V20_THRESHOLD,np.nan) for score in baseline]
    rows = [(f.row,dict(time_s=f.raw[:,0],v18=a,shield=b)) for f,a,b in zip(flights,old,newer)]
    return paired(rows,'shield_3s',threshold)


def unmatched_alerts(flights, scores, threshold):
    groups = Counter()
    for f, score in zip(flights,scores):
        high = score>=threshold
        starts = f.raw[high & ~np.r_[False,high[:-1]],0]
        onsets = np.asarray([e['onset_s'] for e in f.row['events']])
        used = set()
        for t in starts:
            lo,hi = np.searchsorted(onsets,[t+3.,t+7.])
            match_index = next((i for i in range(lo,hi) if i not in used),None)
            if match_index is not None:
                used.add(match_index)
            elif lo<hi:
                groups['duplicate'] += 1
            elif np.any((onsets>=t) & (onsets<t+3.)):
                groups['late'] += 1
            elif np.any((onsets>=t+7.) & (onsets<t+10.)):
                groups['early'] += 1
            else:
                groups['no_onset_within_10s'] += 1
    return groups


def forecast_line(label, row):
    return (f'| {label} | {row["hits"]}/{row["events"]} | {row["recall"]:.2%} | {row["precision"]:.2%} | '
            f'{row["false_per_hour"]:.2f} | {row["ranking"]["ap"]:.2%} |')


def control_line(label, row):
    return (f'| {label} | {row["persistent_onsets"]} | {row["onsets_per_hour"]:.2f} | {row["cut_seconds_per_hour"]:.2f} | '
            f'{row["timely_pdr"]:.2%} | {row["mission_completed"]}/{row["flights"]} | {row["safety_violations"]} |')


def finish(plan,freeze,forecast,loops):
    fresh = [read_checked(item) for item in forecast]
    baseline_keys = {case_key(row) for row in fresh}
    expected = {(s['region'],s['kind'],s['direction'],s['seed']) for s in plan['fresh_specs']}
    if baseline_keys!=expected or len(fresh)!=len(expected):
        raise ValueError('Fresh flight coverage changed')
    flights = [load(item) for item in forecast]
    baseline = [f.raw[:,1] for f in flights]
    chosen = freeze['chosen']
    p = predict(flights,freeze,chosen)
    booster = restore_model(freeze['models'][chosen['model']])
    gains = booster.feature_importance(importance_type='gain',iteration=chosen['rounds'])
    feature_gain = dict(sorted(zip(booster.feature_name(),gains.tolist()),key=lambda item:-item[1])[:8])
    reference = dict(**metrics(flights,baseline,V20_THRESHOLD,True),ranking=ranking(flights,baseline))
    learned = dict(**metrics(flights,p,chosen['threshold'],True),ranking=ranking(flights,p),
                   paired=compare_scores(flights,baseline,p,chosen['threshold']))
    reference['unmatched_groups'] = unmatched_alerts(flights,baseline,V20_THRESHOLD)
    learned['unmatched_groups'] = unmatched_alerts(flights,p,chosen['threshold'])
    if any(sum(row['unmatched_groups'].values())!=row['false_or_late'] for row in (reference,learned)):
        raise ValueError('Unmatched warning categories do not conserve the alarm count')
    ablations = {}
    for family in ('held_early','dual_early'):
        opts = [v for v in freeze['choices'] if v['model']==family and v['learned_weight']==1.]
        valid = [v for v in opts if v['selection']['false_per_hour']<=CAP and v['selection']['alerts']>0]
        pick = max(valid or opts,key=lambda v:(v['selection']['recall'],v['selection']['precision'],-v['selection']['false_per_hour']))
        values = predict(flights,freeze,pick)
        ablations[family] = dict(model=pick['model'],rounds=pick['rounds'],threshold=pick['threshold'],
            **metrics(flights,values,pick['threshold'],True),ranking=ranking(flights,values))
    controls = [read_checked(item) for item in loops]
    existing = [r for r in controls if r['arm']=='joint_geometry']
    extended = [r for r in controls if r['arm']=='learned']
    match(fresh,existing)
    match(fresh,extended)
    if len(controls)!=2*len(fresh):
        raise ValueError('Extra or missing closed-loop comparisons')
    summary = aggregate(fresh+controls)
    guards = dict(decision_guards=plan['decision_guards'])
    vs_v7 = comparison(fresh,controls,guards)
    vs_v18 = comparison(existing,extended,guards)['learned']
    direct = shared_prefix(existing,extended)
    added = [a for row in extended for a in row['intervention'] if a['expert'].startswith('v21_')]
    if any(a['motion_guard_blocked'] for a in added):
        raise ValueError('A newly added action bypassed the final motion guard')
    eligible = learned['paired']['recall_delta_ci95'][0]>0 and learned['false_per_hour']<=CAP and learned['precision']>=reference['precision']-.05
    prevention_supported = vs_v18['feasible'] and vs_v7['learned']['feasible'] and vs_v18['paired_uncertainty']['onsets_per_hour_ci95'][1]<0
    timing = None
    if (ARTIFACTS/'timing_freeze.json').exists():
        from .timing import result, scores
        timing = result(flights)
        tp = read(ARTIFACTS/'timing_freeze.json')
        time_scores = scores(restore_model(tp['model']),flights)
        w = timing['weight']
        time_scores = [w*a+(1.-w)*b for a,b in zip(time_scores,baseline)]
        timing['paired'] = compare_scores(flights,baseline,time_scores,timing['threshold'])
    info = dict(baseline=reference,learned=learned,ablations=ablations,control=summary,
                vs_v7=vs_v7,vs_v18=vs_v18,control_forecast=freeze['control_forecast'],
                added_overrides=len(added), added_while_connected=sum(e['actual_pre_action_connected'] for e in added),
                direct_examples=direct,learning_supported=eligible,prevention_supported=prevention_supported,
                modeled_endpoints_verified=sum(r['exact_endpoints'] for r in fresh+extended),
                both_plan_forks=sum(int(np.any(f.x[:,-2:]>0,axis=1).sum()) for f in flights),
                top_tree_gain_features=feature_gain,timing_ablation=timing,created_at=now())
    write(ARTIFACTS/'final_results.json',info)
    train = freeze['training']
    ci = learned['paired']['recall_delta_ci95']
    delta = learned['recall']-reference['recall']
    weight = chosen['learned_weight']
    score_name = '명령 유지 경로를 학습한 모델' if chosen['model']=='held_early' else '두 경로 가정을 학습한 모델'
    lines = ['# 캡스톤 디자인 II — 차폐 기반 예측 학습과 예방 재배치', '',
             f'동일한 신규 42개 비행에서 3초 이상 7초 미만 선행 탐지율은 '
             f'V20 {reference["recall"]:.2%}, 학습 후보 {learned["recall"]:.2%}였다. '
             f'차이는 {100*delta:+.2f}%p, 공간 그룹 재표집 95% 구간은 [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]%p다.', '',
             '## 실제 학습 내용', '',
             f'훈련 전용 {train["maps"]}개 지도에서 {train["rows"]:,}개 제어 시점의 입력을 만들었다. '
             f'3~7초 선행 단절 정답 {train["early_positive"]:,}개와 3초 미만 임박 단절 정답 '
             f'{train["short_positive"]:,}개가 포함됐다. 단절 구간은 수색기 사이에 직접 또는 중계 연결이 1초 이상 사라진 경우다.', '',
             '첫 3초는 기존 공동 이동 엔진으로 계산한다. 이후 보정을 유지하는 경로와 명목 경로로 돌아가는 경로를 둘 다 계산했다. '
             '각 경로에 대해 0.5초별 수색망 여유, 건물 손실 제거·고정 시의 여유, 취약 링크 비율, 최신 무선 관측을 모델에 전달했다. '
             '실제 미래 위치와 채널은 정답과 평가에 사용했다.', '',
             f'LightGBM 세 모델을 각각 {freeze["models"]["dual_early"]["rounds"]}회 갱신했다. '
             '명령 유지 경로만 보는 선행 예측, 두 경로를 보는 선행 예측, 두 경로를 보는 3초 미만 예측이다. '
             '보정 45비행과 선정 45비행을 분리하고, 새 난수 42조건을 열기 전에 모델과 문턱을 고정했다.', '',
             f'고정한 선행 설정은 {score_name} {chosen["rounds"]}회 갱신, 학습 점수 {weight:.0%}와 '
             f'V20 점수 {1-weight:.0%}, 경보 문턱 {chosen["threshold"]:.4f}다. '
             f'선정 비행에서는 탐지율 {chosen["selection"]["recall"]:.2%}, 정밀도 {chosen["selection"]["precision"]:.2%}, '
             f'오경보·늦은 경보 {chosen["selection"]["false_per_hour"]:.2f}회/편대·시간이었다.', '',
             '## 같은 궤적에서 선행 예측', '',
             '| 방법 | 제때 탐지 | 선행 탐지율 | 정밀도 | 오경보·늦은 경보/편대·시간 | 정밀도–재현율 평균(AP) |',
             '|---|---:|---:|---:|---:|---:|', forecast_line('V20 기존 물리 점수',reference),
             forecast_line('사전 선정 학습 설정',learned),
             forecast_line('학습 · 명령 유지 경로만',ablations['held_early']),
             forecast_line('학습 · 두 경로',ablations['dual_early'])]
    if timing is not None:
        tci = timing['paired']['recall_delta_ci95']
        lines += [forecast_line('보조 학습 · 단절 시간 구간',timing), '',
                  '시간 구간 보조 모델은 다음 단절을 0.5~3초, 3~4.5초, 4.5~5.5초, 5.5~7초, 7~9초, '
                  '구간 내 단절 없음으로 나눠 학습했다. 새 비행 결과를 집계하기 전에 독립 문턱을 고정했다. '
                  '실제 재배치에는 사전 선정한 이진 선행 설정을 사용했다. '
                  f'시간 구간 모델의 탐지율 차이 95% 구간은 [{100*tci[0]:+.2f}, {100*tci[1]:+.2f}]%p다.']
    lines += ['', '편대 단위 경보 비용 기준은 시간당 4회다. 비행 중 경보가 이어진 구간은 처음 한 번만 세고, '
             '실제 사건 하나와 최대 한 번 매칭한다. AP는 평가 가능한 3초 제어 시점의 점수 순위를 측정한다.', '',
             '| 첫 1초 복구 근거 | 전체 | V20 제때 탐지 | 학습 설정 제때 탐지 |', '|---|---:|---:|---:|']
    for key, name in [('building_supported','건물 손실 제거로 복구'),('other_or_mixed','기타·복합')]:
        a,b = reference['by_cause'].get(key,{}),learned['by_cause'].get(key,{})
        lines.append(f'| {name} | {sum(a.values())} | {a.get("hit",0)} | {b.get("hit",0)} |')
    if reference['median_lead_s'] is not None and learned['median_lead_s'] is not None:
        lines += ['', f'탐지한 사건의 선행시간 중앙값은 V20 {reference["median_lead_s"]:.1f}초, '
                  f'학습 설정 {learned["median_lead_s"]:.1f}초였다.']
    lines += ['', '| 매칭하지 못한 경보 | V20 | 학습 설정 |', '|---|---:|---:|']
    for key,name in [('duplicate','적합 시간대의 실제 사건을 이미 경보'),('late','0~3초 미만에 실제 단절'),
                     ('early','7~10초 미만에 실제 단절'),('no_onset_within_10s','향후 10초에 새 단절 없음')]:
        lines.append(f'| {name} | {reference["unmatched_groups"][key]} | {learned["unmatched_groups"][key]} |')
    lines += ['', ('학습 설정은 새 비행에서 탐지율·경보 비용의 사전 확인 조건을 충족했다.' if eligible
                   else '학습 설정은 새 비행에서 탐지율·경보 비용의 사전 확인 조건을 모두 충족하지 못했다.'), '',
              '## 실제 중계 명령을 바꾼 비교', '',
              ('선정 자료의 조건을 충족해 학습 선행 점수를 추가 예방 개입에 사용했다.' if freeze['control_forecast']=='learned'
               else '학습 선행 점수가 선정 자료의 적용 조건을 충족하지 못해 추가 예방 개입에는 V20 선행 점수를 사용했다.'),
              '기존 V18의 예방·복구 명령은 먼저 적용했다. 추가 개입은 한 대의 보정만 바꾸고 '
              '두 경로 가정에서 연결 부족량이 악화되지 않으며 가장 취약한 경로의 부족량이 감소할 때 허용했다. '
              '첫 3초 연결 여유와 학습한 임박 단절 점수도 확인했다.', '',
              '| 제어 | 새 단절 | 새 단절/비행 시간 | 단절 초/비행 시간 | 기한 내 전달 | 임무 완주 | 안전 위반 |',
              '|---|---:|---:|---:|---:|---:|---:|', control_line('V7 기존 PPO',summary['baseline']),
              control_line('V18 기존 예방',summary['joint_geometry']), control_line('V21 추가 예방',summary['learned']), '']
    ci_ctrl = vs_v18['paired_uncertainty']['onsets_per_hour_ci95']
    lines += [f'V18 대비 새 단절률 차이는 {vs_v18["onsets_per_hour_delta"]:+.2f}회/h '
              f'(95% 구간 {ci_ctrl[0]:+.2f}~{ci_ctrl[1]:+.2f}회/h)였다. '
              f'기한 내 전달률 {100*vs_v18["timely_pdr_delta"]:+.2f}%p, '
              f'임무 시간 {vs_v18["duration_ratio"]-1:+.2%}, 중계 이동 거리 {vs_v18["relay_travel_ratio"]-1:+.2%}였다.', '',
              f'추가 명령은 {len(added)}회였고 그중 {info["added_while_connected"]}회는 실제 수색망이 연결된 상태였다. '
              f'첫 명령 분기 직전까지 두 비행의 명령·위치·링크가 일치하며 대안이 이후 12초 동안 연결을 유지한 사례는 {len(direct)}건이었다. '
              '개입 횟수는 피한 단절 수와 구분한다.', '',
              ('실제 이동 비교가 새 예방 제어의 사전 채택 조건을 충족했다.' if prevention_supported
               else '실제 이동 비교가 새 예방 제어의 사전 채택 조건을 충족하지 못했다.'), '',
              '## 적용 범위', '',
              '알려진 지도 14개에서 세 간섭 조건의 새 난수를 사용했다. 기체의 현재 항법 상태를 공유하는 공동 계획기를 전제했다. '
              '두 미래 경로는 다음 PPO 선택을 추정하는 가설이다. 실제 장비의 관측 지연과 비컨 중계는 별도 조건이다. '
              '이전 최종 평가 지도 32개와 격리 지도는 학습과 검증에서 제외했다. 기존 운영 프로필을 유지했다.', '',
              '[입력과 이동 명령 인터페이스](README.md) · [검증 집계](artifacts/final_results.json)', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
