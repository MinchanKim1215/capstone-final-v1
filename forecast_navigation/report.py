"""Formal Korean results; no inference of practical safety from AP."""
from . import ROOT,ARTIFACTS
from .contracts import read,sha,PARENT
from .figures import LABELS


def label(tag):return LABELS.get(tag,tag).replace('\n',' ')
def pct(v):return '정의 불가' if v is None else f'{v*100:.2f}%'
def number(v):return '정의 불가' if v is None else f'{v:.2f}'


def main():
    result=read(ARTIFACTS/'final_results.json');gate=read(ARTIFACTS/'gate_result.json');audit=read(ARTIFACTS/'audit.json');run=read(ARTIFACTS/'runtime_checks.json')
    lines=['# 캡스톤 디자인 II — 예측 알고리즘 개선·학습 결과','',
        '작성일: 2026-11-03. 공유 경로·시간 그래프·물리 잔차 학습(V11), 경로 보완 대조 실험(V12). 기존 코드·모델 보존.','',
        '## 판단 요약','',
        '실사용 신뢰성은 AP 상승만으로 판단하지 않는다. 정밀도90%·사건의3초 선행 탐지율80%·오경보/늦은 경보1회/드론·시간 이하·보정 경보50개 이상을 함께 확인한다. 아래 확인 결과가 목표를 충족하지 않으면 실사용 성능 확보로 결론내리지 않는다.','',
        '| 과제 | 비교 기준 | 경보 기준 사전 선정 | 5초 AP 변화 | 3~7초 AP 변화 | 기본 경보 목표 충족 |','|---|---|---|---:|---:|---|']
    for task,title in [('link','물리 링크 단절'),('service','패킷 서비스 장애')]:
        d=result['tasks'][task];c=d['selected_vs_reference'];m=d['models'][d['selected_tag']]
        lines.append(f'| {title} | {label(d["reference_tag"])} | {label(d["selected_tag"])} | {c["pooled_ap5_difference"]*100:+.2f}%p | {c["pooled_ap37_difference"]*100:+.2f}%p | {"충족" if m["calibrated"]["current_fold_goal_met"] else "미달"} |')
    lines.extend(['','경보 기준 선정과 위험 순위 우수 후보는 다를 수 있다. 선택 자료에서 미리 정한 순위 우수 후보의 확인 결과도 함께 표시한다.','',
        '| 과제 | 사전 순위 우수 후보 | 5초 AP | 3~7초 AP | 기존 강한 기준 대비 AP 변화 |','|---|---|---:|---:|---:|'])
    for task,title in [('link','물리 링크 단절'),('service','패킷 서비스 장애')]:
        d=result['tasks'][task];tag=d['ranking_champion'];m=d['models'][tag]['raw'];c=d['ranking_vs_reference']
        lines.append(f'| {title} | {label(tag)} | {pct(m["next5_on_common_currently_up_links"]["ap"])} | {pct(m["any3to7_on_common_support"]["ap"])} | {c["pooled_ap5_difference"]*100:+.2f} / {c["pooled_ap37_difference"]*100:+.2f}%p |')
    lines.extend(['','후보는 선택 자료로 선정하고, 확률·경보 문턱을 보정 자료에서 고정했다. 이후 새42비행을 생성했다. 확인 결과를 보고 다시 후보를 고르지 않았다. 물리 과제는 기존 선정 후보 외에 기존 순위 우수 후보도 사전 비교 기준으로 포함했다. 약한 기준만 골라 향상률을 계산하지 않았다.','',
        '## 1. 구성과 학습','',
        '- **계획 공유:** 각 드론이 이미 발행한3초 이동과 잠정8초 경로를1Hz 비컨에 포함한다. 실제 수신에 성공한 메시지만 사용한다.128바이트·기존 비컨 전송 성공·기존TDMA 슬롯 조건을 유지했다.','',
        '- **시간 그래프:** 노드와 링크 관계를 유지한다. 현재 및 과거3초의 네 시점에 메시지 전달과GRU를 적용한다.','',
        '- **물리 계산과 학습 보정:** 공개 지도와 받은 경로로0~8초 전파 여유를 계산한다. 모델이 남은 오차를 보정하고 링크 단절과 패킷 서비스 장애를 별도로 예측한다. 전체 무선 패킷 엔진의 미래를 아는 구조는 아니다.','',
        '- **경로 보완:**3초 이후 잠정 경로에 현재 임무 경로·회피·커버리지·회복 상태를 반영했다. 현재 내비게이션 상태와 공개 임무만 사용했다. 실제 미래PPO 명령·RF·간섭·패킷 정답은 입력하지 않았다. 이후 제어가 달라지면 잠정 경로도 틀릴 수 있다.','',
        f'- 추가 학습 전 선택45비행 경로 오차를 검사했다. 명령 발행 직후부터3~7초의 위치RMSE {gate["original_rmse_m"]:.2f}m → {gate["mission_preview_rmse_m"]:.2f}m, {gate["relative_rmse_reduction"]*100:.2f}% 감소. 비행·시점을 같은 비중으로 집계했다. 사전 기준30%를 통과했다. **위치 오차 감소율이며 예측 성능 향상률이 아니다.**','',
        '- 원래 경로/보완 경로 각각 난수2개로 추가6,000회씩 학습했다. 네 모델의 초기 가중치·훈련94,080표본·정답·미니배치 순서·학습률·선택11,520표본을 맞췄다.0회 초기 상태도 체크포인트 선택에 포함했다. 단순히 더 오래 학습한 효과와 경로 보완 효과를 구분했다.','',
        '- 기존 모델, 두 경로 모델 및 기존 모델과의50% 혼합을 비교했다. 물리6후보·서비스5후보를 모두 평가했다. 새42비행은 알려진14지도의 새 난수 실현이다. 미지의 지형·실제 장비 검증과 구분한다.','',
        '- 이번 새 예측기 학습은 지도학습이다. PPO는 자료 생성 정책으로 고정했다. 새로운 PPO 재배치 정책을 학습하거나 자동 교체하지 않았다.',''])
    for task,title in [('link','물리 링크 단절'),('service','패킷 서비스 장애')]:
        d=result['tasks'][task];lines.extend([f'## 2. {title} — 동일한 새42비행','',
            '| 후보 | 5초 AP | 3~7초 AP | FA1 정밀도 | 3초 선행 탐지율 | 오경보·늦은 경보/h |','|---|---:|---:|---:|---:|---:|'])
        for tag,m in d['models'].items():
            a=m['raw'];p=m['calibrated']['common_FA_budget1'];mark=' **경보 기준 선정**' if tag==d['selected_tag'] else ''
            if tag==d['ranking_champion']:mark+=' **순위 우수**'
            lines.append(f'| {label(tag)}{mark} | {pct(a["next5_on_common_currently_up_links"]["ap"])} | {pct(a["any3to7_on_common_support"]["ap"])} | {pct(p["precision"]) if p else "운영점 없음"} | {pct(p["event_recall_3s"]) if p else "운영점 없음"} | {number(p["false_or_late_per_drone_hour"]) if p else "운영점 없음"} |')
        lines.extend(['','FA1: 보정 자료에서 오경보·늦은 경보1회/h 이하, 경보50개 이상인 문턱이다. 확인 비행에서는 이 예산을 초과할 수 있다. 운영점 없음은 해당 보정 기준을 만족하는 문턱이 없다는 뜻이다.',''])
        lines.extend(['| 참고 확률 지표: 5초 | AUROC | 보정 Brier | 보정 ECE | 양성/전체 관측 |','|---|---:|---:|---:|---:|'])
        for tag in dict.fromkeys([d['reference_tag'],d['ranking_champion']]):
            m=d['models'][tag];a=m['raw']['next5_on_common_currently_up_links'];b=m['calibrated']
            lines.append(f'| {label(tag)} | {pct(a["auroc"])} | {b["calibrated_next5"]["brier"]:.5f} | {b["reliability_next5"]["ece"]:.5f} | {a["positives"]:,}/{a["samples"]:,} |')
        lines.extend(['','AUROC도 정답률이 아니다. Brier는 확률 오차, ECE는 구간별 예측 확률과 실제 발생률 차이의 요약이다. 낮을수록 좋지만 희귀 사건에서 전체 평균이 낮다는 이유만으로 고위험 확률의 신뢰성을 보장하지 않는다.',''])
        p=d['models'][d['selected_tag']]['calibrated']['alarms']
        lines.extend([f'선정 후보 기본 문턱: 정밀도 {pct(p["precision"])} ({p["timely_correct_first_alerts"]:,}/{p["first_alerts"]:,}개 경보),3초 선행 탐지율 {pct(p["event_recall_3s"])} ({p["event_hits_3s"]:,}/{p["eligible_directed_link_events"]:,}개 사건), 오경보·늦은 경보 {number(p["false_or_late_per_drone_hour"])}/h.',''])
        c=d['contrasts'][0]
        lines.extend([f'동일 학습량의 원래 경로 → 보완 경로:5초 AP {c["pooled_ap5_difference"]*100:+.2f}%p,3~7초 AP {c["pooled_ap37_difference"]*100:+.2f}%p. 이 차이가 경로 보완의 직접 대조 결과다.',''])
        lines.extend(['| 간섭 조건 | 기준 비행 평균 3~7초 AP | 사전 순위 우수 후보 비행 평균 3~7초 AP | AP 정의 비행/전체 |','|---|---:|---:|---:|'])
        for kind,title2 in [('clear','추가 방해 없음'),('intermittent','간헐 방해'),('continuous','지속 방해')]:
            g=d['condition_groups'][kind];a=g[d['reference_tag']];b=g[d['ranking_champion']]
            lines.append(f'| {title2} | {pct(a["macro_flight_ap37"])} | {pct(b["macro_flight_ap37"])} | {b["ap37_defined_flights"]}/{b["flights"]} |')
        lines.extend(['','조건별 AP는 사건이 있는 비행의 AP 평균이다. 위 전체 관측 통합AP와 가중 방식이 다르다. 사건이 없는 비행은 AP 정의 불가로 표시하며 오경보 계산에는 포함한다.',''])
        u=d['selected_vs_reference']['uncertainty']
        if u:
            for metric,title2 in [('macro_spatial_group_ap5_difference','5초'),('macro_spatial_group_ap37_difference','3~7초')]:
                z=u['metrics'][metric]
                if z['point'] is not None:lines.extend([f'공간 그룹 평균 {title2} AP 차이 {z["point"]*100:+.2f}%p, 짝지은10,000회 재표집95% 구간 [{z["percentile95"][0]*100:+.2f}, {z["percentile95"][1]*100:+.2f}]%p. 전체 관측 통합AP의 신뢰구간과 구분한다.',''])
        if d['ranking_champion']!=d['selected_tag']:
            u=d['ranking_vs_reference']['uncertainty']
            if u:
                for metric,title2 in [('macro_spatial_group_ap5_difference','5초'),('macro_spatial_group_ap37_difference','3~7초')]:
                    z=u['metrics'][metric]
                    if z['point'] is not None:lines.extend([f'사전 순위 우수 후보의 공간 그룹 평균 {title2} AP 차이 {z["point"]*100:+.2f}%p, 짝지은 재표집95% 구간 [{z["percentile95"][0]*100:+.2f}, {z["percentile95"][1]*100:+.2f}]%p.',''])
    v11=read(PARENT/'final_results.json');lines.extend(['## 3. 앞선 세 가지 개선(V11)의 결과','',
        'V11은 그래프6개·같은 표본의 트리 대조4개를 새로 학습했다. 별도의 새42비행으로 확인했다. 아래 수치는V11 내부의 같은 비행 비교다. V12의 새 비행 수치와 직접 빼지 않는다.',''])
    for task,title in [('link','물리'),('service','서비스')]:
        d=v11['tasks'][task];old=d['models'][d['reference_tag']]['raw'];new=d['models'][d['selected_tag']]['raw']
        lines.append(f'- {title}:5초 AP {pct(old["next5_on_common_currently_up_links"]["ap"])} → {pct(new["next5_on_common_currently_up_links"]["ap"])},3~7초 AP {pct(old["any3to7_on_common_support"]["ap"])} → {pct(new["any3to7_on_common_support"]["ap"])}.')
    lines.extend(['','## 4. 검증·고찰','',
        f'- {audit["source_flights"]}비행에서 원본 자료와 보완 경로의 결합을 검사했다. 비컨 {audit["beacon_messages_roundtrip_checked"]:,}개의 원래 경로를128바이트 확장 패킷에서 정확히 복원했다. 기존 RF·성공한 수신·실제 이동·패킷 결과는 유지했다.','',
        f'- 원래 코드·참조 가중치 {audit["original_dependencies_preserved"]}개 보존. 원래 최종 지도32개 미사용. 새 확인 비행은V11 확인 난수를 재사용하지 않았다.','',
        f'- 내보낸 실제 두API를 {run["frames"]:,}프레임에서 실행했다. 두 경로 입력은 일괄 평가와 정확히 일치했다. 최대 확률 차이: 링크 {run["max_abs_probability_error"]["link"]:.8g}, 서비스 {run["max_abs_probability_error"]["service"]:.8g}. 최초 경보 차이 {run["first_alert_mismatches"]}건. RF 결측은None을 반환한다.','',
        '- 실시간 사전 검사에서는 훈련용 희소 경로 캐시를 전체 비행 정답처럼 읽은 검사 오류를 수정했다. 같은 훈련 비행의 과거 관측으로 공개 경로를 재계산한 뒤 입력·예측 일치를 검증했다. 모델·정답·확률·경보 문턱을 수정한 조치는 아니다.','',
        '- 경로 오차 감소와 AP 개선, 실제 선행 경보를 분리해서 판단한다. AP가 올라도 사건 대부분을3초 전에 탐지하지 못하면 신뢰 가능한 경보로 볼 수 없다. 긴 학습 시간만으로 전조 정보나 실제 무선 환경의 차이가 해결되지는 않는다.','',
        '- 현재 환경은2.4GHz·고도50m·단순화한 건물 전파·확률 채널·이상적TDMA다. 실제 모듈·안테나 자세·채널 경쟁·실측자료 검증이 없다. 데스크톱 처리 시간은 탑재 장치 성능 인증이 아니다.','',
        '- 보완 경로는 시뮬레이터의 임무 제어기가 보유한 현재 계획 상태에서 산출한다. 수신 모델에는 실제 성공한 비컨으로 전달한다. 분산된 실제 드론이 동일한 제어 상태를 확보·동기화하는 통합 구현은 별도 검증이 필요하다.','',
        '## 용어','',
        '| 용어 | 의미 |','|---|---|',
        '| AP | 위험 순위에 따른 정밀도·재현율 곡선의 요약. 예측 정확도와 구분. |',
        '| %p | 두 백분율의 절대 차이. |',
        '| 정밀도 | 발생한 경보 중 제때 맞은 경보 비율. |',
        '| 3초 선행 탐지율 | 실제 사건 중 발생3초 이상·7초 미만 전에 유효 최초 경보한 비율. 늦은 경보로 보충하지 않음. |',
        '| 물리 링크 단절 | 특정 두 드론 사이의 무선 연결 상태 단절. |',
        '| 서비스 장애 | 최근1초의 기한 내 패킷 전달률90% 미만이1초 이상 지속하는 사건. 최소20개 패킷 조건. |',
        '| RMSE | 위치 예측 오차의 제곱 평균 제곱근. 단위m. |','',
        '## 근거','',
        '- `artifacts/protocol.json`, `comparison_spec.json`, `training_plan.json`: 사전 설계.','',
        '- `artifacts/freeze.json`, `confirmation_access.json`: 모델·문턱 고정과 확인 시작 시각.','',
        '- `artifacts/final_results.json`: 모든 후보·조건·대조 효과와 재표집 구간.','',
        '- `artifacts/audit.json`, `runtime_checks.json`: 무결성·실시간 실행 검증.','',
        '- `artifacts/figures/`: 두 과제의 지표와 실제 학습 곡선.','',
        f'결과 데이터 SHA256: `{sha(ARTIFACTS/"final_results.json")}`.',''])
    (ROOT/'RESULTS_20261103.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
