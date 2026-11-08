"""Publish the cause evidence and paired results in Korean."""
from . import ARTIFACTS, ROOT
from .protocol import read


LABELS = dict(baseline='기존 V7 PPO', geometry='V17 첫 중계기 예방',
              joint_geometry='V18 공동 궤적·차폐', joint_radio='V18 공동 궤적·차폐와 간섭 전환')
CAUSES = dict(building='건물', interference='간섭', shadow='섀도잉',
              combined_only='결합 제거 필요', unresolved='미해결')
MISSES = dict(timely_alert='3~7초 선행 경보', alert_already_active_or_used='경보가 이미 활성화됐거나 다른 사건과 매칭',
              no_decision_in_lead_window='선행 구간에 평가 시점 없음', observations_missing='수신 근거 부족',
              forecast_missed_window='위험 또는 예상 시점이 선행 구간 조건 미충족')


def label(key):
    return CAUSES.get(key) or '·'.join(CAUSES[k] for k in key.split('+'))


def table(title, values):
    lines = [f'## {title}', '',
             '| 제어 | 비행 | 단절 시작 | 시간당 단절 | 시간당 단절 시간 | 기한 내 전달률 | 완주 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, row in values.items():
        lines.append(f'| {LABELS[arm]} | {row["flights"]} | {row["persistent_onsets"]} | '
                     f'{row["onsets_per_hour"]:.2f}회 | {row["cut_seconds_per_hour"]:.2f}초 | '
                     f'{row["timely_pdr"]:.2%} | {row["mission_completed"]}/{row["flights"]} |')
    return lines


def finish(result, development, survey):
    chosen = result['chosen']
    decision = ('이번 시뮬레이션에서 사전 채택 조건을 충족했다.' if result['adoption_supported']
                else '이번 시뮬레이션에서 사전 채택 조건에 미달했다. 운영 프로필은 기존 상태로 유지했다.')
    lines = ['# 캡스톤 디자인 II — 단절 전 재배치 검증', '', decision, '',
             '수색기 사이에 직접 또는 중계 경로가 1초 이상 사라지는 새 사건을 주 지표로 삼았다. '
             '같은 지도·방향·난수에서 제어를 바꿔 비행을 끝까지 실행했다. 단절 시간, 패킷 전달, 완주, 안전을 함께 비교했다.', '',
             '## 312개 지도에서 확인한 단절 패턴', '',
             f'앞서 수행한 {survey["flights"]}비행의 개별 원인 근거를 다시 검사했다. '
             f'총 {survey["total"]:,}구간 중 수색기 한 대가 중계기 없이 단독으로 고립된 경우는 '
             f'{survey["one_aircraft_isolated"]:,}구간이다. 두 수색 연결 성분 중 하나에 기체가 한 대뿐인 경우다.', '',
             '| 같은 궤적에서 각각 제거해 첫 1초를 복구한 요인 조합 | 구간 | 무간섭 | 간헐 간섭 | 지속 간섭 |',
             '|---|---:|---:|---:|---:|']
    for cause, n in sorted(survey['exclusive_evidence'].items(), key=lambda row: -row[1]):
        counts = [survey['by_condition'][k].get(cause, 0) for k in ('clear', 'intermittent', 'continuous')]
        lines.append(f'| {label(cause)} | {n:,} | {counts[0]:,} | {counts[1]:,} | {counts[2]:,} |')
    lines += ['', '표의 각 구간은 한 행에만 포함된다. 예를 들어 건물·간섭은 두 요인을 각각 제거해도 복구된 경우다. '
              '여러 요인을 동시에 제거해야만 복구된 경우는 결합 제거로 분리했다. 이 결과는 고정 궤적의 전파 원인에 관한 증거다.', '',
              '차폐와 우회 경로 상실에는 지도·궤적 기반 예측을 사용했다. 현재 수신 잡음이 첫 중계기와 또 다른 수신 지점에서 '
              '함께 높고 상승할 때만 간섭 외삽을 추가하는 후보도 비교했다. 대기열 단독 증가는 서비스 감시로 기록했다.', '']
    from .precursors import audit
    precursors = audit()['groups']
    radio = precursors['interference_evidence']
    building = precursors['building_without_interference_evidence']
    lines += ['### 원인별로 수신 신호에서 보인 전조', '',
              f'간섭 제거로 복구된 사건에서 첫 중계기의 6초 전과 3초 전 잡음이 모두 관측된 '
              f'{radio["noise_pairs_3s_6s"]}건 중 {radio["local_noise_high_and_rising"]}건이 높은 잡음과 상승률 조건을 함께 만족했다. '
              f'3초 전 잡음이 높았던 경우는 같은 비행 정상창과 짝지은 {radio["matched_noise"]}쌍에서 '
              f'단절 전 {radio["high_noise_before"]}건, 정상창 {radio["high_noise_normal"]}건이었다. '
              '이 집계는 첫 중계기만 측정한 과거 기록이다. 두 수신 지점을 요구하는 실제 전환 규칙의 동작률과 구분한다.', '',
              f'건물 제거로 복구되고 간섭 단독 제거로는 복구되지 않은 사건에서는 정상창과 짝지은 '
              f'{building["matched_margin"]:,}쌍 중 수신 여유가 3dB 미만인 링크가 '
              f'단절 전 {building["weak_before"]}쌍, 정상창 {building["weak_normal"]}쌍에서 관측됐다. '
              '첫 중계기가 받은 링크만으로는 두 창의 차이가 작았다. 차폐 예측에는 발행 경로와 우회 경로 정보를 함께 썼다.', '']
    lines += table('개발 15조건', development['arms'])
    lines += ['', f'새 난수를 실행하기 전에 **{LABELS[chosen]}**를 고정했다. '
              '안전과 임무 비용을 충족하는 후보에서 시간당 단절 시작, 단절 시간, 중계 이동 거리 순으로 비교했다.', '']
    joint = development['arms']['joint_radio']
    lines += [f'간섭 전환 후보가 실제로 반영한 간섭 기반 명령은 '
              f'{joint["intervention_experts"].get("radio_trend", 0)}회다.', '']
    lines += table('새 난수 42조건', result['arms'])
    lines += ['', '14개 지도 각각에 세 간섭 조건을 적용했다. 같은 조건의 외부 간섭과 섀도잉은 동일하게 설정했다. '
              '경로가 바뀐 이후에는 패킷과 비컨의 난수 호출 순서가 달라질 수 있다.', '']
    for other, reference in [('V7', 'vs_baseline'), ('V17', 'vs_v17')]:
        values = result[reference][chosen]
        low, high = values['paired_uncertainty']['onsets_per_hour_ci95']
        lines.append(f'- {other} 대비 단절 시작률 차이 {values["onsets_per_hour_delta"]:+.2f}회/h, '
                     f'지도 그룹 재표집 95% 구간 [{low:+.2f}, {high:+.2f}]회/h. '
                     f'기한 내 전달률 {100*values["timely_pdr_delta"]:+.2f}%p, '
                     f'중계 이동 거리 {(values["relay_travel_ratio"]-1):+.2%}, '
                     f'임무 시간 {(values["duration_ratio"]-1):+.2%}.')
    v18 = result['arms'][chosen]
    fresh = [read(item['path']) for item in read(ARTIFACTS / 'fresh_manifest.json')['rows']]
    lines += ['', f'V18은 연결이 유지 중인 상태에서 {v18["prevention_before_actual_cut"]}회 선행 개입했다. '
              f'변경 명령 {result["exact_endpoint_checks"]}회의 예측된 첫 구간 종점이 비행 엔진과 일치했다. '
              f'안전 위반은 {v18["safety_violations"]}건이다.', '',
              '| 간섭 조건 | V7 시간당 단절 | V17 시간당 단절 | V18 시간당 단절 |',
              '|---|---:|---:|---:|']
    for kind, description in [('clear', '무간섭'), ('intermittent', '간헐 간섭'), ('continuous', '지속 간섭')]:
        rates = []
        for arm in ('baseline', 'geometry', chosen):
            rows = [row for row in fresh if row['arm']==arm and row['spec']['kind']==kind]
            rate = 3600 * sum(row['persistent_onsets'] for row in rows) / sum(row['duration_s'] for row in rows)
            rates.append(f'{rate:.2f}회')
        lines.append('| ' + ' | '.join([description, *rates]) + ' |')
    lines += ['', '## 순수 선행 예측과 실제 예방', '']
    a = result['advisory']
    lines += [f'기존 PPO의 행동을 유지하면서 선정한 V18 예측을 그림자 실행했다. 전체 단절 {a["onsets"]}건 중 '
              f'{a["hits"]}건을 3초 이상 7초 미만 앞서 경보했다. 선행 탐지율 {a["recall"]:.2%}, 경보 정밀도 '
              f'{a["precision"]:.2%}, 오경보와 늦은 경보 합계 {a["false_or_late_per_fleet_hour"]:.2f}회/편대·시간이다. '
              '경보가 연속으로 켜진 구간은 처음 한 건만 센다. 기존 서비스 경보의 드론·시간 단위와 구분한다.', '',
              '| 원인 근거 | 전체 단절 | 제때 경보 | 수신 근거 부족 |', '|---|---:|---:|---:|']
    for cause, counts in sorted(a['by_cause'].items(), key=lambda row: -sum(row[1].values())):
        lines.append(f'| {label(cause)} | {sum(counts.values())} | {counts.get("timely_alert", 0)} | '
                     f'{counts.get("observations_missing", 0)} |')
    lines += ['', '| 전체 단절의 평가 결과 | 건수 |', '|---|---:|']
    for key, name in MISSES.items():
        lines.append(f'| {name} | {a["reasons"].get(key, 0)} |')
    lines += ['', f'예측 가능 판단 시점은 {a["usable_decisions"]:,}/{a["total_decisions"]:,}회다. '
              f'선정 모델이 간섭 외삽을 선택한 시점은 {a["trend_decisions"]:,}회다.', '',
              '### 명령 한 번만 바꾼 동일 이력 비교', '']
    b = result['branches']
    if b:
        kept = sum(row['alternative_connected_first_12s'] for row in b)
        original = sum(len(row['baseline_runs_30s']) for row in b)
        alternative = sum(len(row['alternative_runs_30s']) for row in b)
        lines += [f'선행 신호와 실행 가능한 대안이 확인된 {len(b)}개 조건에서 명령 한 번만 바꿨다. '
                  f'{kept}개 조건에서 다음 12초 동안 연결이 계속 유지됐다. '
                  f'30초 단절 구간 수는 기존 {original}개, 대안 {alternative}개였다. '
                  '개입 전 위치, 명령, 실제 채널, 패킷 서비스 이력이 일치함을 검사했다. '
                  '이 표본은 대안이 있는 사례를 고른 결과이므로 전체 단절의 예방률은 전체 비행 표를 기준으로 판단한다.', '',
                  '| 지도 | 간섭 조건 | 선행시간 | 기존 30초 단절 | 대안 30초 단절 |', '|---|---|---:|---:|---:|']
        for row in b:
            lines.append(f'| {row["spec"]["region"]} | {row["spec"]["kind"]} | '
                         f'{row["baseline_onset_s"]-row["time_s"]:.1f}초 | '
                         f'{row["baseline_down_seconds"]:.1f}초 | {row["alternative_down_seconds"]:.1f}초 |')
        example = next((row for row in b if row['alternative_connected_first_12s']), None)
        if example is not None:
            from adaptive_joint_expanded.features import OFFSETS
            event = example['intervention']
            old_offset = OFFSETS[event['base_action']].tolist()
            new_offset = OFFSETS[event['action']].tolist()
            lines += ['', f'`{example["spec"]["region"]}`의 `{example["spec"]["kind"]}` 사례는 '
                      f'{example["time_s"]:.1f}초에 기체 인덱스 {event["changed_relay"]}인 중계기의 목표 보정을 '
                      f'{old_offset}m에서 {new_offset}m로 바꿨다. 해당 중계기의 첫 3초 실제 이동은 '
                      f'{event["actual_own_displacement_m"]:.1f}m였다. 기존 비행의 단절 시작은 '
                      f'{example["baseline_onset_s"]:.1f}초였다.']
    else:
        lines.append('사전에 정의한 조건을 충족한 사례가 0건이었다.')
    lines += ['', '## 운영 연결과 검증 범위', '',
              '공동 비행 계획기의 현재 위치·임무 진행 상태를 읽어 후보를 구성했다. 전파 위험에는 실제 수신 기록만 반영했다. '
              '첫 3초는 실행 계획기로 해석한 확정 전 후보이며, 3초 이후는 현재 보정을 유지한다는 가정이다. '
              '기체 간 현재 항법 상태를 공유할 수 있는 중앙 조정 구조를 전제한다. 실제 수신 지연에서의 동등성은 별도 시험이 필요하다.', '',
              '같은 지도에서 난수와 방향을 다시 선택했다. 최종 평가 지도 32개와 격리 지도는 제외했다. '
              '모델 가중치는 고정하고 예측의 제어 연결을 대조했다. 운영 프로필은 별도로 유지했다.', '',
              '예측 입력과 PPO에 전달되는 항목은 `INTERFACE.md`에 기록했다. '
              '비행별 자료, 지도별 원인 집계, 선정 규칙, 직접 분기 결과는 `artifacts`에 저장했다.', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
