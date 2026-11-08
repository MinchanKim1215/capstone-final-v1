"""Verify and explain the closed-loop comparison after the frozen runs."""
from collections import Counter

import numpy as np

from . import ARTIFACTS, ROOT, WORKSPACE
from .diagnostics import connected, factorial_radio, minimal_repairs
from .evaluation import aggregate, comparison, match, shared_prefix
from .protocol import now, read, sha, status, verify_freeze, write
from adaptive_joint_expanded.features import OFFSETS
from adaptive_joint_expanded.world import World
from forecast_causal.delivery import already_sealed
from forecast_rebuild.contracts import verify_prior

NAMES = {'baseline': '기존 V7', 'reactive': '관측상 단절 뒤 개입',
         'geometry': '경로·차폐 예측으로 선행 개입', 'switch': '차폐·간섭 예측 전환'}


def audit_fresh(plan, frozen):
    verify_freeze()
    prior = verify_prior()
    if not already_sealed():
        raise ValueError('Prior V16 delivery is missing')
    access = read(ARTIFACTS / 'fresh_access.json')
    manifest = read(ARTIFACTS / 'fresh_manifest.json')
    freeze_sha = sha(ARTIFACTS / 'freeze.json')
    if (access['opened_at'] < frozen['created_at'] or access['freeze_sha256'] != freeze_sha or
            manifest['freeze_sha256'] != freeze_sha):
        raise ValueError('Fresh runs must be collected after the fixed selection')
    old_seeds = {x['source_record']['spec']['seed'] for x in plan['archived_records']}
    fresh_seeds = {spec['seed'] for spec in plan['fresh_specs']}
    if old_seeds & fresh_seeds:
        raise ValueError('An archived seed was reused')
    expected = {(spec['region'], spec['kind'], arm)
                for spec in plan['fresh_specs'] for arm in ('baseline', 'reactive', frozen['chosen'])}
    observed = set()
    rows = []
    for entry in manifest['rows']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Mission metadata changed')
        row = read(entry['path'])
        key = (row['spec']['region'], row['spec']['kind'], row['arm'])
        if key in observed or key not in expected or row['spec'] not in plan['fresh_specs']:
            raise ValueError('Invalid fresh scenario')
        observed.add(key)
        if not row['audited']['passed'] or sha(row['archive_path']) != row['archive_sha256']:
            raise ValueError('Mission safety/trace evidence failed')
        with np.load(row['archive_path'], allow_pickle=False) as trace:
            observer = row['search_drones']
            np.testing.assert_array_equal(trace['network'], connected(trace['links'], observer))
            np.testing.assert_array_equal(trace['commands'][:, observer+1:], OFFSETS[trace['original_actions'][:, 1:]])
            changed = np.any(trace['commands'][:, observer] != OFFSETS[trace['original_actions'][:, 0]], axis=1)
            recorded = [int(event['time_s']/3) for event in row['intervention']]
            np.testing.assert_array_equal(np.flatnonzero(changed), recorded)
        rows.append(row)
    if observed != expected:
        raise ValueError('Incomplete paired cohort')
    report = dict(created_at=now(), fresh_runs=len(rows), fresh_cases=len(plan['fresh_specs']),
                  prior=prior, prior_v16_verified=True, all_action_changes_traced=True,
                  other_relay_commands_match_original_actor=True, archive_hashes_verified=True,
                  original_final_test_opened=False)
    write(ARTIFACTS / 'audit.json', report)
    return rows


def intervention_diagnostics(rows):
    errors = []
    lead = Counter()
    new_cut_within_7s = 0
    for row in rows:
        for action in row['intervention']:
            if action['motion_guard_blocked'] or not action['actual_pre_action_connected'] or action['reason'] != 'prevention':
                continue
            errors.append(float(np.linalg.norm(np.asarray(action['actual_endpoint']) - action['projected_own_endpoint'])))
            lead[str(action['predicted_onset_in_s'])] += 1
            t = action['time_s']
            new_cut_within_7s += any(0 <= start-t < 7 for start, _ in row['persistent_cut_intervals'])
    return dict(preemptions=len(errors), projected_endpoint_different=sum(e > 1e-6 for e in errors),
                endpoint_error_median_m=float(np.median(errors)) if errors else None,
                endpoint_error_max_m=max(errors) if errors else None,
                predicted_lead_counts=lead, actual_cut_within_7s=new_cut_within_7s)


def case_details(examples, baseline, selected):
    result = []
    for example in examples:
        spec = example['spec']
        original = next(row for row in baseline if row['spec'] == spec)
        alternative = next(row for row in selected if row['spec'] == spec)
        env = World(**spec, deployment=original['deployment'], observer=None)
        frame = int(example['baseline_onset_s'] * 2)
        step = int(example['decision_s'] / 3)
        with np.load(original['archive_path'], allow_pickle=False) as trace:
            positions = env.city.points[trace['nodes']]
            repairs, cuts = factorial_radio(env, positions, trace['links'], np.array([frame, frame+1]))
        stable = [part for part in repairs[frame] if part in repairs[frame+1]]
        action = example['intervention']
        before = positions[step, env.ns]
        after = positions[step+1, env.ns]
        cutoff = example['decision_s'] + 30
        result.append(dict(spec=spec, decision_s=example['decision_s'], actual_lead_s=example['lead_s'],
                           predicted_lead_s=action['predicted_onset_in_s'],
                           original_offset_m=OFFSETS[action['base_action']].tolist(),
                           selected_offset_m=OFFSETS[action['action']].tolist(),
                           original_displacement_m=float(np.linalg.norm(after-before)),
                           selected_displacement_m=action['actual_own_displacement_m'],
                           original_margin_at_predicted_onset_db=action['original_margin_db'][action['predicted_onset_in_s']],
                           selected_margin_at_predicted_onset_db=action['alternative_margin_db'][action['predicted_onset_in_s']],
                           components_at_baseline_onset=cuts[frame], first_second_minimal_repairs=minimal_repairs(stable),
                           alternative_has_full_30s_window=alternative['duration_s'] >= cutoff,
                           alternative_cut_onsets_in_30s=[start for start, _ in alternative['persistent_cut_intervals']
                                                          if example['decision_s'] <= start < cutoff]))
    return result


def table(arms, order):
    lines = ['| 제어 | 비행 | 단절 시작 | 시간당 단절 시작 | 시간당 누적 단절 | 기한 내 패킷 전달 | 완주 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in order:
        a = arms[arm]
        lines.append(f'| {NAMES[arm]} | {a["flights"]} | {a["persistent_onsets"]} | '
                     f'{a["onsets_per_hour"]:.2f}회 | {a["cut_seconds_per_hour"]:.2f}초 | '
                     f'{100*a["timely_pdr"]:.2f}% | {a["mission_completed"]}/{a["flights"]} |')
    return lines


def make_report(r, plan):
    cause = r['causes']
    c = cause['totals']
    chosen = r['chosen']
    new = r['fresh_arms'][chosen]
    old = r['fresh_arms']['baseline']
    effect = r['versus_baseline'][chosen]
    reactive_effect = r['versus_reactive'][chosen]
    ci = effect['paired_uncertainty']['onsets_per_hour_ci95']
    hits = r['shared_prefix_examples']
    diagnostic = r['interventions']
    lines = [
        '# 캡스톤 디자인 II — 통신 단절 예방 실험', '',
        f'같은 42개 비행 조건에서 기존 PPO의 지속 단절 시작은 시간당 {old["onsets_per_hour"]:.2f}회, '
        f'사전에 선정한 예방 제어는 {new["onsets_per_hour"]:.2f}회였다. '
        f'차이는 {effect["onsets_per_hour_delta"]:+.2f}회/h이며 지도 그룹을 함께 재표집한 95% 구간은 '
        f'[{ci[0]:+.2f}, {ci[1]:+.2f}]회/h다.', '',
        ('이번 42개 조건에서 예방 제어는 두 대조군보다 단절 시작이 적었고 임무·안전·비용 조건을 충족했다.'
         if r['prevention_point_criteria_met'] else
         '예방 제어 채택 조건을 충족하지 못했다. 단절 시작을 줄였는지와 단절이 시작된 뒤 지속 시간을 줄였는지를 각각 판단한다.'), '',
        ('개발 자료에서는 단절 시작률이 늘었고 새 난수 비교의 95% 구간도 0을 포함한다. 운영 채택 판단에는 추가 검증이 필요하다.'
         if ci[0] <= 0 <= ci[1] else '재표집 구간은 이번에 비교한 지도 그룹의 변동을 반영한다.'), '',
        '## 지도별 단절 원인', '',
        f'원인 조사 범위는 312개 지도, 400비행이다. 기존 377개 전체 비행의 실제 기록을 재사용했고, '
        f'당시 자료에 빠져 있던 과거 검증 지도 23개는 새로 전체 비행을 실행했다. 기존 기록은 모든 지속 단절의 첫 1초에 '
        f'전파를 다시 계산했다. 최종 평가 지도 32개와 격리 지도 1개는 이번 실험 범위에서 제외했다.', '',
        f'1초 이상 지속된 수색망 단절 구간은 {c["network_runs"]:,}개다. 그중 비행 시작 시 이미 단절된 '
        f'{c["network_runs"]-c["network_onsets"]}구간을 제외하면 새 단절 시작은 {c["network_onsets"]:,}건이다. '
        f'개별 링크의 단절 구간은 {c["individual_link_runs"]:,}개였다. 대체 경로가 남아 있으면 개별 링크가 끊겨도 수색망은 유지된다.', '',
        '| 첫 1초 내내 수색망을 복구한 모의 요인 제거 | 해당 단절 구간 |',
        '|---|---:|',
        f'| 건물 추가 손실만 제거 | {c.get("network_repair_building", 0):,} |',
        f'| 외부 간섭만 제거 | {c.get("network_repair_interference", 0):,} |',
        f'| 섀도잉만 제거 | {c.get("network_repair_shadow", 0):,} |',
        f'| 두 요인 이상 함께 제거해야 복구 | {c.get("network_combination_required", 0):,} |',
        f'| 세 요인을 제거해도 첫 1초 전체를 복구하지 못함 | {c.get("network_no_fixed_repair", 0):,} |', '',
        f'동일한 궤적·난수에서 첫 1초의 {c["factorial_frames"]:,}개 전파 프레임을 재현했다. '
        '건물 손실·간섭·섀도잉의 유지와 제거를 조합해 8개 전파 조건을 비교했다. 한 사건에서 두 가지 단독 제거가 모두 연결을 회복할 수 있으므로 행은 겹친다. '
        '이 표는 특정 전파 요인의 증거이며, 해당 사건을 실제로 피할 수 있었는지는 제어 대조에서 따로 확인한다.', '',
        f'패킷 서비스 장애 {c["service_runs"]:,}구간 중 {c["service_network_down_at_onset"]:,}구간에서 '
        f'시작 시 수색망이 단절돼 있었다. 마감 전 패킷을 추적할 수 있는 {c["service_packet_attributed"]:,}구간 중 '
        f'{c["service_no_route_head_evidence"]:,}구간은 전달 경로 부재를 동반했다. '
        f'마감 전 재시도 실패 증거는 {c["service_retry_failure_evidence"]:,}구간에서 확인했다. '
        '대기열 적체는 차폐나 경로 부재 이후에 생길 수도 있으므로 별도의 원인 증거와 함께 해석한다.', '',
        '### 단절 전에 실제 관측할 수 있었던 신호', '',
        '| 선행시간 | 같은 비행 대조 창 | 측정값 | 단절 전 중앙값 | 연결 유지 창 중앙값 | 쌍별 평균 차이 |',
        '|---|---:|---|---:|---:|---:|',
    ]
    labels = {'recent_peer_reports': '최근 수신한 기체 수', 'weak_peer_count': '여유 3dB 미만 링크 수',
              'minimum_received_margin_db': '최소 수신 링크 여유(dB)',
              'local_noise_db': '수신 전력과 SINR로 추정한 잡음(dBm)', 'local_queue': '로컬 대기열 패킷 수'}
    for lead, windows in sorted(cause['observed_precursors'].items()):
        for key, m in windows['measurements'].items():
            lines.append(f'| {lead}초 | {m["n"]:,} | {labels[key]} | {m["before_cut_median"]:.2f} | '
                         f'{m["healthy_control_median"]:.2f} | {m["paired_difference_mean"]:+.2f} |')
    received_margin = cause['observed_precursors']['3']['measurements']['minimum_received_margin_db']
    lines += ['', '대조 창은 같은 비행에서 가까운 정상 연결 구간을 골랐다. 값이 두 창에 모두 관측된 쌍을 사용했다.', '',
              f'단절 3초 전 첫 중계기에 기록된 최소 링크 여유는 대조 창과 평균 {received_margin["paired_difference_mean"]:+.2f} dB 차이였다. '
              '이 측정은 첫 중계기에서 수신한 링크를 대상으로 한다. 수신에 실패한 링크와 다른 두 기체 사이의 모든 링크를 포괄하지 않으므로, '
              '수색망의 대체 경로까지 발행 경로로 예측할 필요가 있다.', '',
              '## 원인별 예측기와 이동 결정', '',
              '| 관측에 따른 판단 | 이번에 대조한 방식 | 이동 결정 |',
              '|---|---|---|',
              '| 발행 경로에서 차폐나 중계 경로 상실이 예상됨 | 지도 전파 손실과 1~8초 경로를 이용한 연결 여유 예측 | 단절이 예상되는 명령과 대안 명령의 수색망 여유를 비교 |',
              '| 수신기 잡음 추정치가 이미 높고 더 올라감 | 최근 잡음 상승률을 제한해 미래 링크 여유에 적용 | 이 조건에서만 간섭 추세 예측으로 전환하는 후보를 시험 |',
              '| 대기열만 커지고 미래 연결 여유가 유지됨 | 로컬 대기열 증가 감시 | 통신 경보를 기록하고 PPO의 이동 명령을 유지 |',
              '| 실제 수신한 경로 또는 최근 무선 관측이 부족함 | 관측 부족으로 분류 | PPO의 이동 명령 유지 |', '',
              '개발 15비행에서 두 예방 후보를 대조했다. 아래 선정은 새 42개 조건을 실행하기 전에 고정했다.', '']
    lines += table(r['selection_arms'], ('baseline', 'reactive', 'geometry', 'switch'))
    lines += ['', f'사전 선정안은 **{NAMES[chosen]}**이다. 비용 조건을 만족한 두 예방 후보에서 '
              '단절 시작률, 누적 단절, 중계 이동 거리 순으로 비교했다. 기존 PPO는 비교 기준으로 유지했다.', '',
              '## 새 난수에서 실제 재배치 비교', '',
              '14개 기존 지도에서 무간섭·간헐 간섭·지속 간섭의 42개 새 시나리오를 구성했다. 각 조건마다 기존 V7, 관측상 단절 뒤 개입, 사전 선정 예방 제어를 각각 끝까지 실행했다. '
              '지도·방향·편대·간섭 일정·섀도잉을 맞췄다. 경로가 달라진 뒤 패킷 및 비컨 난수 호출 순서는 달라질 수 있다.', '']
    lines += table(r['fresh_arms'], ('baseline', 'reactive', chosen))
    lines += ['', f'예방 제어와 복구 대조군의 단절 시작률 차이는 {reactive_effect["onsets_per_hour_delta"]:+.2f}회/h였다. '
              f'기존 V7 대비 중계 이동 거리는 {(effect["relay_travel_ratio"]-1)*100:+.2f}%, '
              f'임무 시간은 {(effect["duration_ratio"]-1)*100:+.2f}%, '
              f'기한 내 전달률은 {effect["timely_pdr_delta"]*100:+.2f}%p였다. '
              f'안전 위반은 {new["safety_violations"]}건이며 기존에서 완료했던 임무 중 미완료로 바뀐 임무는 '
              f'{effect["mission_completion_regressions"]}개다.', '',
              f'실제 수색망이 연결된 상태에서 승인된 예방 개입은 {diagnostic["preemptions"]}회였다. '
              f'그중 예상 이동 종점과 실제 종점이 달랐던 개입은 {diagnostic["projected_endpoint_different"]}회, '
              f'개입 뒤 7초 이내 실제 단절이 시작된 경우는 {diagnostic["actual_cut_within_7s"]}회였다. '
              '개입 횟수는 예방한 사건 수와 구분한다.', '',
              f'두 비행의 첫 명령 분기 전까지 위치·무선·서비스 기록이 일치했고, 기존에서 7초 안에 단절이 시작된 반면 '
              f'예방 제어에서 12초 동안 연결을 유지한 직접 비교 사례는 {len(hits)}건이다. '
              f'그중 실제 선행시간이 3초 이상인 사례는 {sum(e["lead_s"]>=3 for e in hits)}건이다.', '']
    if hits:
        lines += ['| 지도 | 간섭 조건 | 명령 시각 | 기존 단절 시각 | 선행시간 |', '|---|---|---:|---:|---:|']
        for e in hits:
            lines.append(f'| {e["spec"]["region"]} | {e["spec"]["kind"]} | {e["decision_s"]:.1f}초 | '
                         f'{e["baseline_onset_s"]:.1f}초 | {e["lead_s"]:.1f}초 |')
    detailed = r['direct_case_details']
    if detailed:
        first = detailed[0]
        maps_count = len({case['spec']['region'] for case in detailed})
        lines += ['', f'이 사례는 {maps_count}개 지도에서 얻었다. 첫 사례에서는 기존 제어가 선택한 '
                  f'{first["original_offset_m"]} m 보정을 {first["selected_offset_m"]} m로 바꿨다. '
                  f'예상 단절 시점의 수색망 연결 여유는 {first["original_margin_at_predicted_onset_db"]:.2f} dB에서 '
                  f'{first["selected_margin_at_predicted_onset_db"]:.2f} dB로 바뀌었다. '
                  f'다음 3초간 첫 중계기의 실제 이동 거리는 기존 {first["original_displacement_m"]:.2f} m, '
                  f'예방 제어 {first["selected_displacement_m"]:.2f} m였다.', '',
                  '| 간섭 조건 | 기존 단절 첫 1초를 복구한 단독·결합 요인 제거 | 새 단절 시작, 개입 후 30초 |',
                  '|---|---|---:|']
        factor_names = {'building': '건물 손실', 'interference': '외부 간섭', 'shadow': '섀도잉'}
        for case in detailed:
            fixes = ' 또는 '.join('과 '.join(factor_names[factor] for factor in part)
                                  for part in case['first_second_minimal_repairs'])
            count = len(case['alternative_cut_onsets_in_30s']) if case['alternative_has_full_30s_window'] else '관측 부족'
            lines.append(f'| {case["spec"]["kind"]} | {fixes} | {count} |')
        rest = r['other_maps_sensitivity']
        if rest:
            lines += ['', f'위 사례의 지도를 제외한 나머지 조건에서 기존과 예방 제어의 단절 시작은 '
                      f'{rest["baseline"]["persistent_onsets"]}건과 {rest[chosen]["persistent_onsets"]}건이었다. '
                      '사례를 확인한 뒤 계산한 민감도 점검이다.']
    lines += ['', '## 적용 범위와 다음 실험', '',
              '기존과 새 제어의 입력·위험값·이동 명령은 `INTERFACE.md`에 정리했다. 이번에는 첫 중계기 한 대의 명령만 바꿀 수 있게 구성했다. '
              '후보별 경로는 마지막으로 받은 동료 경로와 현재 후보 오프셋을 유지한다는 가정으로 계산했다. 실제 비행의 공동 이동 제약과 다음 3초 재계획 때문에 결과가 달라질 수 있다. '
              '예측 여유는 1초 간격으로 평가했고 실제 단절은 0.5초 간격에서 판정했다.', '',
              '이 설정의 이동 비용 한도는 임무 시간 5%, 중계 이동 거리 10%, 기한 내 전달률 감소 0.5%p다. '
              '완주 퇴행과 안전 위반은 허용하지 않았다. 새 난수 시험은 알려진 지도와 시뮬레이터의 전파·패킷 조건에 해당한다.', '',
              '다음 후보는 실제 이동 제약을 통과한 궤적을 먼저 만들고, 대체 경로까지 포함한 1초 지속 단절 위험을 그 궤적마다 비교하는 방식이다. '
              '건물·경로 모델을 주 예측기로 두고, 실제 잡음 상승 증거가 있을 때만 간섭 추세를 더하는 방식은 별도 비교 대상으로 남긴다. '
              '대기열 전조는 경로 단절과 분리해 패킷 수준에서 검증한다.', '',
              '원시 근거는 `artifacts/survey_manifest.json`, `artifacts/cause_summary.json`, `artifacts/selection.json`, '
              '`artifacts/freeze.json`, `artifacts/fresh_manifest.json`, `artifacts/final_results.json`, `artifacts/audit.json`에 저장했다.', '',
              '경로 계획에서 가시선 연결 조건을 명시적으로 비교하는 접근을 참고했다. '
              '[Chen·Guo, Multi-UAV Deployment in Obstacle-Cluttered Environments with LOS Connectivity](https://arxiv.org/abs/2308.12117)', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def finish(plan, frozen):
    rows = audit_fresh(plan, frozen)
    baseline = [row for row in rows if row['arm']=='baseline']
    reactive = [row for row in rows if row['arm']=='reactive']
    chosen = [row for row in rows if row['arm']==frozen['chosen']]
    versus_baseline = comparison(baseline, reactive+chosen, plan)
    versus_reactive = comparison(reactive, chosen, plan)
    a = versus_baseline[frozen['chosen']]
    b = versus_reactive[frozen['chosen']]
    examples = shared_prefix(baseline, chosen)
    direct_maps = {e['spec']['region'] for e in examples}
    other_maps = [row for row in baseline + chosen if row['spec']['region'] not in direct_maps]
    result = dict(created_at=now(), chosen=frozen['chosen'], causes=read(ARTIFACTS / 'cause_summary.json'),
                  selection_arms=read(ARTIFACTS / 'selection.json')['arms'], fresh_arms=aggregate(rows),
                  versus_baseline=versus_baseline, versus_reactive=versus_reactive,
                  shared_prefix_examples=examples, direct_case_details=case_details(examples, baseline, chosen),
                  other_maps_sensitivity=aggregate(other_maps) if direct_maps and other_maps else {},
                  interventions=intervention_diagnostics(chosen),
                  prevention_point_criteria_met=a['feasible'] and a['onsets_per_hour_delta'] < 0 and b['onsets_per_hour_delta'] < 0,
                  original_final_test_opened=False, freeze_sha256=sha(ARTIFACTS / 'freeze.json'))
    write(ARTIFACTS / 'final_results.json', result)
    make_report(result, plan)
    write(ARTIFACTS / 'completed.json', dict(created_at=now(), results_sha256=sha(ARTIFACTS / 'final_results.json'),
                                           report_sha256=sha(ROOT / 'RESULTS.md'), audit_sha256=sha(ARTIFACTS / 'audit.json')))
    status('cause_survey_and_paired_prevention_evaluation_complete', state='completed',
           maps=result['causes']['maps'], fresh_cases=len(baseline), report=str(ROOT / 'RESULTS.md'))
    print('Prevention study complete', result['fresh_arms'], flush=True)
