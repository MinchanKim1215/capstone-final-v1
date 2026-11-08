"""Write a concise Korean comparison with the operating point held fixed."""
from . import ARTIFACTS, ROOT


LABEL = dict(v18='V18 기존 3초', planner_3s='V20 계획 경로 사용 · 3초',
             shield_3s='V20 차폐 점수 · 3초', shield_1s='V20 차폐 점수 · 1초')
REASONS = dict(hit='선행 탐지', no_tick='선행 구간에 평가 시점 없음', no_input='수신 요건 미충족',
               below_threshold='점수 미달', already_active_or_used='기존 경보 활성 또는 다른 사건과 매칭')


def metric_line(label, row):
    return (f'| {label} | {row["hits"]}/{row["events"]} | {row["recall"]:.2%} | '
            f'{row["precision"]:.2%} | {row["false_per_hour"]:.2f} | {row["alerts"]} |')


def finish(selection, results):
    old, priority, matched = (results[k] for k in ('v18','priority','matched'))
    delta = priority['recall']-old['recall']
    ci = priority['paired']['recall_delta_ci95']
    lines = ['# 캡스톤 디자인 II — 차폐 기반 선행 예측', '',
             f'같은 {results["native_missions"]}개 신규 비행에서 V18 선행 탐지율은 {old["recall"]:.2%}, '
             f'사전 선정한 {LABEL[priority["variant"]]}는 {priority["recall"]:.2%}였다. '
             f'차이는 {100*delta:+.2f}%p, 지도 그룹 재표집 95% 구간은 '
             f'[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]%p다.', '',
             '## 구현과 근거', '',
             '전체 지도 조사에서 지속 단절 2,586구간 중 2,407구간의 첫 1초가 건물 손실 제거로 복구됐다. '
             '또한 1,485건은 단절 3초 이상 7초 미만 전에 필요한 경로 비컨을 받지 못했다. '
             '그래서 계획기가 아는 경로와 실제 받은 무선 관측을 별도로 처리했다. [전체 지도 원인 조사](../outage_atlas/RESULTS.md)', '',
             '새 예측기는 반초 단위의 편대 궤적에서 수색기 사이의 우회 연결까지 계산한다. '
             '연속 두 프레임의 연결 상실이 예상되면 해당 프레임에서 건물 손실을 제거하거나 현재 값으로 고정한 경우를 대조한다. '
             '경로 비컨이 오래돼도 계획기 궤적과 최근 로컬 RF가 있으면 평가한다. '
             '어떤 수신기의 RF가 비어 있으면 해당 수신기의 링크 여유 변동 폭을 크게 잡는다.', '',
             '## 비교 방법', '',
             '검증 지도 15개에서 경보 문턱을 보정하고 다른 15개에서 예측 변형을 선정했다. '
             '두 코호트는 V19의 명령을 한 단계씩 재생해 위치·링크·패킷 결과가 일치함을 확인했다. '
             '그 뒤 사전에 고정한 모델·문턱으로 알려진 지도 14개의 새 난수 42조건을 평가했다. '
             '모든 예측기가 각 비행에서 같은 V7 PPO 경로를 보므로, 이동 개입의 효과가 선행 예측 수치에 섞이지 않는다.', '',
             '| 새 난수 42조건 | 제때 탐지 | 탐지율 | 정밀도 | 오경보·늦은 경보/편대·시간 | 전체 경보 |',
             '|---|---:|---:|---:|---:|---:|', metric_line('V18 기준',old),
             metric_line(f'탐지 우선 · {LABEL[priority["variant"]]}',priority),
             metric_line(f'기존 경보 비용 기준 · {LABEL[matched["variant"]]}',matched), '',
             '실제 단절보다 3초 이상 7초 미만 앞서 시작한 경보를 한 건씩 매칭했다. '
             '이어진 경보는 첫 시점 한 건만 센다. 매칭되지 않은 경보에는 틀린 시점의 경보가 포함된다. '
             '오경보 기준은 편대 전체에 대한 경보로, 과거 서비스 예측의 드론·시간 단위와 구분한다.', '',
             f'탐지 우선 문턱은 {priority["threshold"]:.2f}, 보정 단계의 비용 상한은 {priority["calibration_budget"]:.2f}회/편대·시간이다. '
             f'기존 경보 비용 기준 문턱은 {matched["threshold"]:.2f}, 보정 비용 상한은 '
             f'{matched["calibration_budget"]:.2f}회/편대·시간이다.', '']
    if priority['false_per_hour'] > priority['calibration_budget']:
        lines += ['탐지 우선 설정은 새 난수 비행에서 사전 경보 비용 상한을 넘었다.', '']
    if matched['false_per_hour'] > old['false_per_hour']:
        lines += ['기존 경보 비용으로 보정한 설정의 새 난수 경보 비용은 V18보다 높았다.', '']
    if not matched['alerts']:
        lines += ['V18 보정 비용을 상한으로 두면 시험한 새 문턱에서 경보를 유지할 수 없었다. '
                  '표의 0% 정밀도는 경보 0건을 표시하기 위한 값이다.', '']
    if (ARTIFACTS/'tracking_freeze.json').exists():
        from .tracking import final_evaluation
        tracked = final_evaluation()
        gap = tracked['recall']-old['recall']
        ci_t = tracked['paired']['recall_delta_ci95']
        lines += ['## 1초 예측에서 경보 중복을 줄인 별도 대조', '',
                  '예측한 절대 단절 시각이 연속 두 초 동안 1초 이내로 일치하면 경보를 낸다. '
                  '같은 예상 사건을 중복 경보하지 않도록 2초 시각 범위와 최소 3초 경보 간격을 적용한다. '
                  '두 개발 코호트에서 비용 한도를 만족한 문턱을 새 비행 결과 집계 전에 고정했다.', '',
                  '| 방식 | 제때 탐지 | 탐지율 | 정밀도 | 오경보·늦은 경보/편대·시간 | 전체 경보 |',
                  '|---|---:|---:|---:|---:|---:|', metric_line('V18 기존',old),
                  metric_line(f'1초 추적 경보 · 문턱 {tracked["threshold"]:.2f}',tracked), '',
                  f'탐지율 차이는 {100*gap:+.2f}%p, 지도 그룹 재표집 95% 구간은 '
                  f'[{100*ci_t[0]:+.2f}, {100*ci_t[1]:+.2f}]%p다. 사전 선정한 3초 설정과 별도로 평가했다.', '']
        if tracked['false_per_hour'] > priority['calibration_budget']:
            lines += ['1초 추적 경보는 새 난수에서 편대 시간당 4회 경보 비용 상한을 넘었다.', '']
    lines += ['## 기능을 나눠 비교한 결과', '',
              '| 보정된 변형 | 제때 탐지 | 탐지율 | 정밀도 | 오경보·늦은 경보/편대·시간 | 전체 경보 |',
              '|---|---:|---:|---:|---:|---:|']
    for variant,row in results['ablations'].items():
        lines.append(metric_line(LABEL[variant],row))
    lines += ['', '계획 경로 사용만 적용한 변형은 V18의 수색기 비컨 최신도 조건을 완화한다. '
              '차폐 점수 변형은 여기에 무선 관측의 불확실성과 건물 대조를 반영한다. '
              '1초 변형은 진행 중인 3초 이동 구간에서 새로 받은 무선 관측으로 경보를 갱신한다.', '',
              '| 새 단절의 결과 | V18 | 탐지 우선 설정 |', '|---|---:|---:|']
    for key,name in REASONS.items():
        lines.append(f'| {name} | {old["reasons"].get(key,0)} | {priority["reasons"].get(key,0)} |')
    lines += ['', '| 첫 1초 복구 근거 | 전체 사건 | V18 제때 탐지 | 탐지 우선 설정 제때 탐지 |',
              '|---|---:|---:|---:|']
    for key,name in [('building_supported','건물 손실 제거로 복구'),('other_or_mixed','기타 또는 복합')]:
        a = old['by_cause'].get(key,{})
        b = priority['by_cause'].get(key,{})
        lines.append(f'| {name} | {sum(a.values())} | {a.get("hit",0)} | {b.get("hit",0)} |')
    lines += ['', '## 적용 범위', '',
              f'새 난수 비행에서 {results["live_forecast_count"]:,}개 1초 예측을 실행했다. '
              f'예측 함수 호출 시간의 95백분위는 {results["latency_p95_ms"]:.2f}ms였다. '
              f'기존 엔진과 첫 이동 종점 {results["exact_first_segments"]:,}개가 일치했다. '
              f'임무 완주는 {results["mission_completed"]}/{results["native_missions"]}, '
              f'안전 위반은 {results["safety_violations"]}건이다.', '',
              '공동 계획기가 현재 편대 항법 상태를 공유한다고 가정했다. '
              '이미 발행한 구간 이후의 궤적은 최근 목표 보정을 유지한다는 가정이다. '
              '기존 최종 평가 32개와 격리 지도는 이번 보정·선정·확인에 사용하지 않았다. '
              '예측 출력을 새 이동 명령에 연결한 폐루프 효과는 후속 검증 대상이며 기존 운영 설정을 유지했다.', '',
              '[예측 입력과 재배치에 전달할 값](README.md) · [검증 집계 원본](artifacts/final_results.json)', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
