"""Primary event recall, controls, actual counts and unrelaxed limitations."""
from . import ROOT,ARTIFACTS
from .contracts import read,sha
from .figures import LABELS


def pct(x):return '정의 불가' if x is None else f'{x*100:.2f}%'
def label(tag):return LABELS.get(tag,tag).replace('\n',' ')


def main():
    d=read(ARTIFACTS/'final_results.json');audit=read(ARTIFACTS/'audit.json');runtime=read(ARTIFACTS/'runtime_checks.json');secondary=read(ARTIFACTS/'secondary_operating_points.json')
    selected=d['models'][d['selected_tag']];a=d['models']['frozen_parent']['calibrated']['alarms'];b=selected['calibrated']['alarms'];c=d['models']['threshold_only_parent']['calibrated']['alarms']
    policy=read(selected['calibrated']['policy_path']);difference=d['selected_vs_frozen'];diagnosis=read(ARTIFACTS/'selection_diagnosis.json')['tasks']['service']['causes']
    lines=['# 캡스톤 디자인 II — 선행 탐지율 개선 학습·검증 결과','',
        '작성일: 2026-11-03. 대상은 첫 중계드론의 패킷 서비스 장애다. 물리 링크 예측기를 새로 학습한 결과와 구분한다.','',
        '## 핵심 결과','',
        f'사전 선정 모델: **{label(d["selected_tag"])}**. 같은 새42비행에서 기존 고정 모델·문턱 대비 선행 탐지율 {pct(a["event_recall_3s"])} → {pct(b["event_recall_3s"])} ({difference["event_recall_difference"]*100:+.2f}%p). 정밀도 {pct(a["precision"])} → {pct(b["precision"])}. 오경보·늦은 경보 {a["false_or_late_per_drone_hour"]:.2f} → {b["false_or_late_per_drone_hour"]:.2f}/h.','',
        f'문턱만 조정한 대조군의 선행 탐지율은 {pct(c["event_recall_3s"])}다. 이 대조군과의 차이는 {d["selected_vs_threshold_only"]["event_recall_difference"]*100:+.2f}%p다. 모델 학습 효과와 문턱 변경 효과를 구분한다.','',
        f'실제 탐지 사건: 기존 {a["event_hits_3s"]:,}/{a["eligible_directed_link_events"]:,}건, 선정안 {b["event_hits_3s"]:,}/{b["eligible_directed_link_events"]:,}건. 제때 맞은 최초 경보: 기존 {a["timely_correct_first_alerts"]:,}/{a["first_alerts"]:,}개, 선정안 {b["timely_correct_first_alerts"]:,}/{b["first_alerts"]:,}개.','',
        f'기존 개발 목표(P90%·선행 탐지율80%·FA1/h·보정 경보50개 이상) 판정: **{"충족" if selected["calibrated"]["current_fold_goal_met"] and selected["calibrated"]["calibration_goal_met"] else "미달"}**. 실제 장비 검증은 포함하지 않았다.','',
        f'선정안의 보정 운영점: {policy.get("rule","기존 고정 정책")}, 보정 경보 {policy["selected"]["first_alerts"]}개. 최소50개 미만인 경우 탐색용 결과로 구분한다. 확인 비행에서 경보가 늘어도 보정 근거가 충분했다는 뜻으로 바꾸지 않는다.','',
        '이전 보고의15.625%(25/160건)는V12의 다른42비행 결과다. 이번 새 비행의 기존 모델 수치를 기준으로 비교하며15.625%를 직접 빼서 향상률을 만들지 않는다.','',
        '## 1. 원인 분석과 구현','',
        f'- 선택45비행의 실제 서비스 사건 {diagnosis["events"]}건 중 {diagnosis.get("risk_below_threshold",0)}건은 필요한3~7초 전 위험이 문턱에 미달했다. 이미 켜진 경보 때문에 놓친 경우는 {diagnosis.get("already_active_without_new_alert",0)}건, 입력 자격이 없던 경우는 {diagnosis.get("no_qualified_input",0)}건이었다. 이전 확인 비행을 새 모델 훈련에 사용하지 않았다.','',
        '- **사건 전 구간 보강:** 훈련245비행에서148,949시점과 자격 있는 사건 전 구간2,393개를 구성했다. 별도 무작위 시간 표본125,440개(비행당512개)는 각 비행 내부의 시점 분포를 유지한다. 사건을 늘려 뽑은 자료로 정밀도 분모를 계산하지 않았다.','',
        '- **사건 단위 손실:** 발생3초 이상·7초 미만 전의 기회 중 적어도 한 번 위험을 충분히 높이도록 학습했다. 사건을 같은 비중으로 반영했다. 자연 표본에서는 정상 구간의 잘못된 고위험 점수를 억제하는 정밀도 근사 제약을 더했다. 이 제약은 실제 경보 정밀도90%를 자동 보장하지 않는다.','',
        '- **20초 과거 관측:** 수신자가 가진 RF·현재 발행 의도·측정 에너지만480항목으로 집계했다.1초 간격20표본을64차원GRU가 처리한다. 미래 채널·간섭 일정·패킷 정답·실제 미래 명령은 입력하지 않았다.','',
        '- **기존 지식 보존:** 기존V12 경로·시간 그래프·물리 모델의 가중치는 고정했다. 해당 모델의481차원 표현과 출력에 작은 보정 예측기를 학습했다. PPO 자료 생성 정책도 고정했다. 이번 여섯 모델은 지도학습이다.','',
        '- 일반BCE 대조군, 사건 중심 학습,20초 이력+사건 학습을 동일 초기 상태·자연 표본·4,000업데이트·고정 난수2개로 비교했다. 체크포인트는 평균AP 대신 실제 최초 경보의 선행 탐지율과 오경보 예산으로 선택했다.','',
        '학습 목적 설계는 [Rath·Hughes, AISTATS2022](https://proceedings.mlr.press/v151/rath22a.html)의 정밀도 제약 아래 탐지율을 최적화하는 접근을 참고했다. 본 구현은 자체 사건 구간 손실과 근사 제약이다. 논문의 보장을 그대로 적용하거나 다른 분야의 성능을 이전했다고 주장하지 않는다.','',
        '## 2. 같은 새42비행의 전체 후보','',
        '| 후보 | 선행 탐지율 | 탐지 사건/전체 | 경보 정밀도 | 맞은 경보/전체 | 오경보·늦은 경보/h |','|---|---:|---:|---:|---:|---:|']
    high_old=secondary['models']['frozen_parent']['5'];high_new=secondary['models'][d['selected_tag']]['5']
    if high_old and high_new:
        x=high_old['confirmation'];y=high_new['confirmation'];at=lines.index('## 1. 원인 분석과 구현')
        lines[at:at]=[f'**탐지 우선 보조 운영점(보정 FA5):** 같은 새42비행에서 기존 {pct(x["event_recall_3s"])} ({x["event_hits_3s"]}/{x["eligible_directed_link_events"]}건) → 선정안 {pct(y["event_recall_3s"])} ({y["event_hits_3s"]}/{y["eligible_directed_link_events"]}건), {(y["event_recall_3s"]-x["event_recall_3s"])*100:+.2f}%p. 정밀도 {pct(x["precision"])} → {pct(y["precision"])}. 실제 오경보 {x["false_or_late_per_drone_hour"]:.2f} → {y["false_or_late_per_drone_hour"]:.2f}/h. 기본FA1 조건의 탐지율 개선이나 실사용 목표 달성으로 해석하지 않는다.','']
    for tag,m in d['models'].items():
        p=m['calibrated']['alarms'];mark=' **사전 선정**' if tag==d['selected_tag'] else ''
        lines.append(f'| {label(tag)}{mark} | {pct(p["event_recall_3s"])} | {p["event_hits_3s"]}/{p["eligible_directed_link_events"]} | {pct(p["precision"])} | {p["timely_correct_first_alerts"]}/{p["first_alerts"]} | {p["false_or_late_per_drone_hour"]:.2f} |')
    lines.extend(['','모든 문턱은 확인 전 보정 자료에서 고정했다. 확인 결과를 보고 모델이나 문턱을 다시 고르지 않았다. 기존 모델·문턱은 실제 기존 운영점을 보존한 기준이며, 문턱 대조군과 새 학습 후보는 같은 탐지율·오경보 선택 절차를 적용했다.','',
        '| 후보 | 5초 AP | 3~7초 AP | 보정 3~7초 Brier |','|---|---:|---:|---:|'])
    for tag,m in d['models'].items():lines.append(f'| {label(tag)} | {pct(m["raw"]["next5_on_common_currently_up_links"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["ap"])} | {m["calibrated"]["calibrated_any3to7"]["brier"]:.5f} |')
    lines.extend(['','AP는 위험 순위 지표이며 정답률이 아니다. 이번 주요 목표는 사건의 선행 탐지율이다. 확률 오차·정밀도·오경보도 함께 판단한다.',''])
    for name,title in [('selected_vs_frozen','기존 고정 정책 대비'),('selected_vs_threshold_only','문턱만 조정한 대조군 대비')]:
        u=d[name]['uncertainty']
        if u:
            interval=u['metrics']['calibrated_recall_difference']['percentile95']
            if all(x is not None for x in interval):lines.extend([f'{title} 사건 탐지율 차이의 공간 그룹 짝지은 재표집95% 구간: [{interval[0]*100:+.2f}, {interval[1]*100:+.2f}]%p.14지도와 세 간섭 조건을 공간 그룹으로 묶어10,000회 재표집했다. 제한된 지리 표본의 불확실성이다.',''])
    lines.extend(['## 3. 학습 요인 대조',''])
    for c in d['contrasts']:
        lines.extend([f'- {label(c["reference"])} → {label(c["candidate"])}: 실제 기본 경보의 선행 탐지율 {c["event_recall_difference"]*100:+.2f}%p, 오경보 {c["FA_per_hour_difference"]:+.2f}/h,3~7초 AP {c["ap37_difference"]*100:+.2f}%p. 각 문턱은 같은 보정 절차에서 정했으며 경보 비용 변화도 함께 본다.',''])
    lines.extend(['| 간섭 조건 | 기존 탐지율 | 선정안 탐지율 | 선정안 정밀도 | 선정안 오경보/h |','|---|---:|---:|---:|---:|'])
    for kind,title in [('clear','추가 방해 없음'),('intermittent','간헐 방해'),('continuous','지속 방해')]:
        g=d['condition_alarm_groups'][kind];old=g['frozen_parent'];new=g[d['selected_tag']]
        lines.append(f'| {title} | {pct(old["event_recall_3s"])} | {pct(new["event_recall_3s"])} | {pct(new["precision"])} | {new["false_or_late_per_drone_hour"]:.2f} |')
    lines.extend(['','## 4. 오경보 허용 범위의 보조 비교','',
        '주 모델·주 문턱 고정 후 새 비행의 예측 결과를 열기 전에, 보정 자료만으로FA1/3/5 운영점을 추가 고정했다. 기본 평가를 교체하지 않는 보조 진단이다. FA3·FA5는 기존FA1 목표 달성으로 해석하지 않는다. 아래 오경보 값은 확인 비행에서 실제 측정한 값이다.','',
        '| 후보 | 보정 FA 예산/h | 선행 탐지율 | 경보 정밀도 | 실제 오경보/h |','|---|---:|---:|---:|---:|'])
    for tag in dict.fromkeys(['frozen_parent',d['selected_tag']]):
        for budget,point in secondary['models'][tag].items():
            if point is None:lines.append(f'| {label(tag)} | {budget} | 보정 경보50개 운영점 없음 | - | - |')
            else:
                p=point['confirmation'];lines.append(f'| {label(tag)} | {budget} | {pct(p["event_recall_3s"])} | {pct(p["precision"])} | {p["false_or_late_per_drone_hour"]:.2f} |')
    lines.extend(['','## 5. 검증과 한계','',
        f'- 원래 의존 파일 {audit["original_dependencies_preserved"]}개, 이전 납품 파일 {audit["previous_delivery_files_preserved"]}개와 보조 산출물 {audit["previous_supporting_outputs_preserved"]}개를 보존했다. 원래 최종32지도는 사용하지 않았다.','',
        f'- {audit["source_flights"]}비행의 자료·가중치·입력 시각·정답을 검사했다. 선택/보정90비행의 정답·자격·시점은 기존 자료와 일치했다. 사건 정의,3≤선행시간<7초, 최초 경보·히스테리시스·사건 분모를 유지했다.','',
        f'- 실제 내보낸API를 {runtime["frames"]:,}프레임에서 실행했다. 최초 경보 판단 차이 {runtime["first_alert_mismatches"]}건, 최대 확률 차이 {runtime["errors"]["probability"]:.8g},480항목 과거 관측 차이0이다. RF 결측은None을 반환한다.','',
        f'- 데스크톱 CPU 추론 중앙값 {runtime["cpu_ms"]["warm_median"]:.2f}ms. 실제 탑재 장치 지연 인증과 구분한다.','',
        '- 새42비행은 알려진14지도의 새 난수 실현이다. 실제 드론·통신모듈·혼잡·안테나·실측 지형 검증은 없다. 공유 경로를 생산하는 임무 제어 상태의 실제 분산 동기화도 미검증이다.','',
        '- 이번20초 이력은 팀·역할별 관측 통계다. 기존 그래프의 노드·링크 표현은3초 과거를 반영한 채 고정했다. 개별 드론·링크의20초 관계 이력을 보존하고 그래프 자체를 다시 학습하는 구조는 이번 비교에 포함하지 않았다. 현재 결과로 예측 가능성의 상한을 단정하지 않는다.','',
        '- 정밀도와 탐지율이 함께 충분히 높아져야 신뢰 가능한 경보다. 실제 무선 환경에 없는 예지 정보를 추가하거나 사건 분모를 줄여 성능을 높이지 않았다. 긴 학습 시간만으로 충분한 전조 정보가 생긴다고 결론내리지 않는다.','',
        '## 용어·근거','',
        '- 선행 탐지율: 실제 사건 중 발생3초 이상·7초 미만 전에 유효 최초 경보한 비율.','',
        '- 경보 정밀도: 발생한 최초 경보 중 정해진 시간에 맞은 경보 비율.','',
        '- FA: 오경보와 늦은 경보를 드론·시간으로 나눈 값.','',
        '- `artifacts/protocol.json`, `training_plan.json`: 데이터·학습·선택 사전 설계.','',
        '- `artifacts/freeze.json`, `confirmation_access.json`: 모델·문턱 고정과 새 확인 시작 시각.','',
        '- `artifacts/final_results.json`, `secondary_operating_points.json`: 모든 후보·대조·불확실성과 보조 운영점.','',
        '- `artifacts/audit.json`, `runtime_checks.json`: 보존·정답·실시간 실행 검증.','',
        f'결과 SHA256: `{sha(ARTIFACTS/"final_results.json")}`.',''])
    options=ARTIFACTS/'runtime_options_checks.json'
    if options.exists():
        q=read(options);lines.extend(['## 보조 운영점 실행 코드','',
            '기본 실행은 `runtime.load_service`다. `runtime_options.load_service(..., budget_per_hour=5)`는 확인 예측 결과를 보기 전에 보정 자료에서 고정한FA5 문턱을 사용한다. 학습 가중치·예측 확률·사건 정의를 변경하지 않는다. 기본 문턱은 자동 교체하지 않는다.','',
            f'보조 실행 인터페이스는 결과 확인 후 추가했다. 문턱 자체는 사전 보정 계획의 값이다. 실제 {q["frames"]:,}프레임 실행에서 최초 경보 차이 {q["first_alert_mismatches"]}건, 최대 확률 차이 {q["max_probability_error"]:.8g}였다. 신뢰 경보와 자동PPO 교체는 비활성이다.',''])
    (ROOT/'RESULTS_20261103.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
