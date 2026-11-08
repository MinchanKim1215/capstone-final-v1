"""Formal Korean report; publish actual results without changing task denominators."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .figures import LABELS


def pct(x):return '정의 불가' if x is None else f'{x*100:.2f}%'
def num(x):return '없음' if x is None else f'{x:.2f}'
def label(tag):return LABELS.get(tag,tag).replace('\n',' ')


def main():
    result=read(ARTIFACTS/'final_results.json');audit=read(ARTIFACTS/'audit.json');runtime=read(ARTIFACTS/'runtime_checks.json')
    lines=['# 캡스톤 디자인 II — 예정 경로 공유·시간 그래프·물리 보정 학습 결과','',f'작성: {now()}. 학습·모델 고정·새 확인 평가·보존 및 실시간 검사를 완료했다.','',
        '## 판정','',
        '두 과제 모두 같은 첫 번째 중계드론의 관측으로 비교했다. 이전 보고서의 전체 중계드론 평균과 직접 차감하지 않는다. 물리 링크 단절과 패킷 서비스 장애의 정답·사건 분모도 다르다.','']
    for task,name in [('link','물리 링크 단절'),('service','패킷 서비스 장애')]:
        data=result['tasks'][task];chosen=data['models'][data['selected_tag']];old=data['models']['previous_selected'];a=chosen['raw'];b=old['raw']
        d5=(a['next5_on_common_currently_up_links']['ap']-b['next5_on_common_currently_up_links']['ap'])*100
        d37=(a['any3to7_on_common_support']['ap']-b['any3to7_on_common_support']['ap'])*100
        goal=chosen['calibrated']['current_fold_goal_met'];point=chosen['calibrated']['common_FA_budget1']
        lines.extend([f'- **{name}**: 확인 전에 선정한 {label(data["selected_tag"])}. 같은 새42비행의 이전 선정 모델 대비 5초 AP {d5:+.2f}%p, 3~7초 AP {d37:+.2f}%p. 개발 전체 경보 목표 {"충족" if goal else "미충족"}.',''])
        if data['selected_tag']=='previous_selected':lines.extend(['  기존 모델을 그대로 유지한 결과다. 0%p 차이는 새 모델의 통계적 동등성을 입증한 수치가 아니다.',''])
        if point:lines.extend([f'  보정 FA1 문턱의 새 확인 결과: 정밀도 {pct(point["precision"])}, 3초 선행 탐지율 {pct(point["event_recall_3s"])} ({point["event_hits_3s"]:,}/{point["eligible_directed_link_events"]:,}건), 오경보·늦은 경보 {point["false_or_late_per_drone_hour"]:.2f}/h.',''])
        else:lines.extend(['  보정 자료에서 시간당 오경보1회 이하와 최소50개 경보를 함께 만족한 운영점이 없었다.',''])
    lines.extend(['AP는 위험 사례를 높은 순위에 배치하는 성능 지표다. 일반적인 정답률이나 예측 확률의 신뢰도를 뜻하지 않는다. 개발 목표는 정밀도90%·3초 선행 사건 탐지율80%·오경보/늦은 경보1회/h 이하·최소50개 경보다. 교수 지정 사양이나 실제 장비 인증 기준이 아니다.','',
        '[물리 링크 비교](artifacts/figures/link_confirmation.png) · [서비스 비교](artifacts/figures/service_confirmation.png) · [학습 곡선](artifacts/figures/training_curves.png)','',
        '## 1. 구현한 세 구성','',
        '1. **예정 경로 공유:** 기존128바이트·1Hz 비콘 안에 시각, 현재 발행 명령, 0~8초 경로, 수신한 이웃의 신호 보고를 직렬화했다. 원래1024비트 비콘의 성공 여부와5ms 예약 슬롯을 그대로 사용했다. 새 공중 송신은 추가하지 않았다.','',
        '   현재 제어기가 이미 발행한3초 이동 구간을 공유한다. 그 이후 경로는 현재 목표까지의 잠정 경로다. 실제 미래 명령·위치·대기·간섭 상태는 입력하지 않았다. 실제 장비용 MAC·센서 통합 구현은 검증하지 않았다.','',
        '2. **시간 그래프:** 드론과 통신 링크의 관계를 유지했다. 현재부터3초 전까지 네 시점, 노드34항목·링크4항목을 사용했다. 메시지 전달2회와GRU가 공간 관계와 시간 변화를 처리한다. 한 중계드론이 실제로 수신한 보고만 사용한다.','',
        '3. **물리 계산과 학습 보정:** 공개 지도와 수신한 계획/등속 가정으로0~8초 전파 변화를 계산했다. 시점별16개 불확실성 표본으로 연결 가능성의 참고값을 만들었다. 모델이 미래 SINR 여유의 잔차를 예측하고, 보정된 여유와 경로 연결 참고값을 장애 예측에 사용하도록 구성했다. 이는 전체 패킷 엔진의 미래 재생이나 실제 전파의 완전한 복제는 아니다.','',
        '## 2. 학습과 비교 설계','',
        '- 기존 신규 비행245개에서 훈련 표본94,080개를 구성했다. 훈련은 모든 중계 관측자를 포함한다. 선택45비행·보정45비행은 첫 중계드론의 모든 유효1Hz 시점을 평가했다. 이전 확인 자료는 최적화에 사용하지 않았다.','',
        '- 그래프3구성×고정 난수2개, 총6개를 각각6,000회 학습했다. 같은 선택 표본11,520개에서 물리/서비스 AP5·AP37 평균으로 체크포인트를 고르고 두 난수의 출력을 동일 비율로 결합했다.','',
        '- 트리 대조군은 기존728/432항목만 쓰는 구성과 공유 정보를 더한 구성을 같은 표본·난수·설정으로 학습했다. 총4개 트리 모델이다. 새 학습 모델은 합계10개다.','',
        '- 각 과제에서 이전 선정 모델, 세 그래프, 두 트리, 이전/물리 그래프50% 혼합의7후보를 비교했다. 모델·보정·경보 문턱을 고정한 뒤 새42비행을 생성했다. 확인 결과로 모델을 다시 고르지 않았다.','',
        '- PPO는 기존 자료 생성 정책으로 고정했다. 새 예측기는 지도학습했다. 이번 결과는 새 PPO 재배치 정책의 연결 유지율 향상을 의미하지 않는다.','',
        '## 3. 같은 새42비행의 평가',''])
    for task,title in [('link','물리 링크 단절'),('service','패킷 서비스 장애')]:
        data=result['tasks'][task];lines.extend([f'### {title}','',
            '| 후보 | 5초 AP | 3~7초 AP | FA1 문턱 정밀도 | 3초 선행 탐지율 | 오경보·늦은 경보/h |','|---|---:|---:|---:|---:|---:|'])
        for tag,value in data['models'].items():
            raw=value['raw'];p=value['calibrated']['common_FA_budget1'];star=' **선정**' if tag==data['selected_tag'] else ''
            lines.append(f'| {label(tag)}{star} | {pct(raw["next5_on_common_currently_up_links"]["ap"])} | {pct(raw["any3to7_on_common_support"]["ap"])} | {pct(p["precision"]) if p else "운영점 없음"} | {pct(p["event_recall_3s"]) if p else "운영점 없음"} | {num(p["false_or_late_per_drone_hour"]) if p else "운영점 없음"} |')
        lines.extend(['','FA1 운영점은 보정 자료에서 고정한 문턱이다. 새 확인에서 예산을 초과하는지도 이 표의 실제 오경보 값으로 판단한다. 확인 자료에 맞춰 문턱을 낮추지 않았다.',''])
        selected=data['models'][data['selected_tag']]['calibrated'];p=selected['alarms']
        lines.extend([f'선정 후보의 기본 참고 문턱: 정밀도 {pct(p["precision"])} ({p["timely_correct_first_alerts"]:,}/{p["first_alerts"]:,}개 경보), 선행 탐지율 {pct(p["event_recall_3s"])} ({p["event_hits_3s"]:,}/{p["eligible_directed_link_events"]:,}개 사건), 오경보·늦은 경보 {p["false_or_late_per_drone_hour"]:.2f}/h. 소수 경보의 높은 적중률은 충분한 신뢰도 근거가 아니다.',''])
        for c in data['contrasts']:
            lines.append(f'- {label(c["reference"])} → {label(c["candidate"])}: 5초 AP {c["pooled_ap5_difference"]*100:+.2f}%p, 3~7초 AP {c["pooled_ap37_difference"]*100:+.2f}%p.')
        lines.append('')
        u=data['selected_vs_reference_uncertainty']
        if u:
            for metric,title2 in [('macro_spatial_group_ap5_difference','5초'),('macro_spatial_group_ap37_difference','3~7초')]:
                m=u['metrics'][metric]
                if m['point'] is not None:lines.extend([f'공간 그룹 평균 {title2} AP 차이 {m["point"]*100:+.2f}%p, 짝지은 재표집95% 구간 [{m["percentile95"][0]*100:+.2f}, {m["percentile95"][1]*100:+.2f}]%p.',''])
    lines.extend(['재표집10,000회의 통계량은 유효 공간 그룹별 AP 평균 차이다. 전체 관측을 통합한 AP 차이의 신뢰구간과 구분한다. 양성이 없는 비행은 AP 정의 불가로 표시하되 오경보 집계에는 포함한다.',''])
    diagnostic=ARTIFACTS/'service_diagnostics.json'
    if diagnostic.exists():
        d=read(diagnostic);c=d['outcomes']
        lines.extend(['## 고찰: 순위 성능과 선행 경보 사이의 간격','',
            f'모델 고정 후 선택45비행만 진단했다. 서비스 사건 {c["eligible_events"]}건 중 {c.get("risk_below_entry_threshold",0)}건은3~7초 전 위험이 고정 경보 문턱에 이르지 않았다. 입력 자격이 없던 사건은 {c.get("no_qualified_history",0)}건, 제때 최초 경보한 사건은 {c.get("timely_first_alert",0)}건이다.','',
            f'최초 경보 탐지율 {pct(d["first_alert_event_recall"])}와 사전 고위험 포착률 {pct(d["any_high_risk_event_coverage"])}가 같다. 경보가 이미 켜져 있어 새 경보로 세지 않았다는 집계 문제만으로 낮은 탐지율을 설명하기 어렵다. 이 진단으로 모델·문턱을 다시 바꾸지 않았다.','',
            '공유 계획의 확정 구간은 발행 후3초다. 현재1Hz 예측 시점에서는 남은 확정 구간이 더 짧다. 따라서3~7초 선행 경보의 대부분은 잠정 경로 가정과 학습한 보정에 의존한다.8초 좌표가 있다는 사실을8초 뒤 실제 움직임을 알고 있다는 뜻으로 해석하지 않는다.','',
            '현재 이진 연결 상태와 서비스 사건의 연관성은 진단값이다. 연속 신호 변화에 전조가 없다는 증거나 예측 가능성의 상한으로 사용하지 않는다. 추가 학습 시간을 늘리는 효과와 실제 장비의 성능은 이번 결과로 입증하지 않았다.',''])
    lines.extend([
        '## 4. 검증과 한계','',
        f'이전 납품 파일 {audit["parent_delivery_files_preserved"]}개와 원래 의존 파일 {audit["original_dependencies_preserved"]}개를 보존했다. 원래 최종 지도32개는 사용하지 않았다. {audit["source_flights"]}개 비행에서 실제 수신 여부와 패킷 정답을 검사했다.','',
        f'공유한 확정 이동 구간 {audit["committed_plan_points_checked"]:,}점을 실제 발행 이동과 대조했다. 좌표당 최대 차이는 {audit["maximum_committed_coordinate_quantization_error_m"]:.5f}m이며0.25m 양자화 범위 이내다.3초 이후 잠정 경로는 아래 선택 자료 오차로 구분한다.','',
        '| 계획 시점 이후 | 선택 자료의 송신 경로 위치 RMSE |','|---|---:|'])
    for row in audit['tentative_plan_diagnostic']:lines.append(f'| {row["horizon_s"]}초 | {row["selection_plan_position_rmse_m"]:.2f}m |')
    lines.extend(['','이 위치 오차는 선택 자료의 사후 진단이다. 미래 실제 위치를 모델 입력으로 제공한 결과가 아니다.','',
        f'실제 내보낸 두 API를 {runtime["frames"]:,}프레임에서 검사했다. 링크 최대 확률 차이 {runtime["max_abs_probability_error"]["link"]:.8g}, 서비스 {runtime["max_abs_probability_error"]["service"]:.8g}, 최초 경보 판단 차이 {runtime["first_alert_mismatches"]}개다. 필수 RF 결측은 위험도None을 반환한다.','',
        f'CPU 실시간 실행과GPU 일괄 실행의float32 연산 차이를 분리 검사했다. 실제 그래프 입력은 완전히 일치했고 확률 허용오차는 {runtime.get("numerical_tolerance",.0001):.5f}로 기록했다. 최초2e-5 검사 중단과 추가 수치 검증 이력도 보존했다. 모델·저장 예측·정답·경보 문턱은 변경하지 않았다.',''])
    for task in ('link','service'):
        latency=runtime['cpu_ms'][task];lines.append(f'- {task} 데스크톱 추론: 첫 {latency["cold"]:.2f}ms, 이후 중앙값 {latency["warm_median"]:.2f}ms. 실제 탑재 장치 지연 인증은 아니다.')
    lines.extend(['','환경은 기존2.4GHz·고도50m·단순화한 건물 전파·확률 채널·이상적TDMA 모델이다. 실제 무선 칩, 채널 경쟁, 안테나 자세, 실측 비행 자료에 대한 검증은 포함하지 않았다. 새42비행은 기존14확인지도의 새 난수 실현이다. 새로운 지형·실제 장비의 성능으로 일반화하지 않는다.','',
        '연구용 자동 경보와 PPO 자동 교체는 활성화하지 않았다. 배포 준비 완료·실사용 성능 확보로 표시하지 않는다.','',
        '## 근거 파일','',
        '- `artifacts/protocol.json`, `data_plan.json`, `training_plan.json`, `tree_control_plan.json`: 입력·표본·학습·대조 설계.','',
        '- `artifacts/freeze.json`, `confirmation_access.json`: 모델·문턱 고정과 새 확인 비행 접근 시각.','',
        '- `artifacts/final_results.json`: 모든 후보의 원시/보정 지표, 조건별 결과와 대조 효과.','',
        '- `artifacts/audit.json`, `runtime_checks.json`: 보존·수신·계획 시각·정답·실시간 검사.','',
        '- `artifacts/research_candidates.json`, `runtime.py`: 선정 후보와 연구용 실행 인터페이스.',''])
    (ROOT/'RESULTS_20261102.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
