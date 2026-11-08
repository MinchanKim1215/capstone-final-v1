"""Formal combined report; all numbers taken from matched frozen outcomes."""
from . import ROOT,ARTIFACTS,GRAPH
from .contracts import read,sha
NAMES={'frozen_v13':'기존 탐지 우선 설정','retuned_v13':'기존 모델 문턱만 재보정','v15_global_event':'집계 이력 추가 사건 학습',
       'v15_peer_bce':'개별 관측 이력 일반 학습','v15_peer_event':'개별 관측 이력 사건 학습','v15_global_event_blend50':'기존·집계 이력 50% 혼합','v15_peer_event_blend50':'기존·개별 이력 50% 혼합',
       'v14_short_bce':'짧은 그래프 일반 학습','v14_short_event':'짧은 그래프 사건 학습','v14_long_event':'20초 그래프 사건 학습',
       'v14_short_event_blend50':'기존·짧은 그래프 50% 혼합','v14_long_event_blend50':'기존·20초 그래프 50% 혼합'}
def pct(v):return '-' if v is None else f'{v*100:.2f}%'

def main():
    r=read(ARTIFACTS/'final_results.json');g=read(GRAPH/'final_results.json');selected=r['selected_tag'];models=dict(g['models']);models.update(r['models'])
    a=models['frozen_v13']['calibrated']['alarms'];b=models[selected]['calibrated']['alarms'];control=models['retuned_v13']['calibrated']['alarms'];diff=r['selected_vs_frozen'];ci=diff['uncertainty']['metrics']['calibrated_recall_difference']['percentile95'] if diff['uncertainty'] else [0,0]
    f=read(ARTIFACTS/'freeze.json');policy=read(next(x['policy'] for x in f['candidates'] if x['candidate']['tag']==selected));runtime=read(ARTIFACTS/'runtime_checks.json');audit=read(ARTIFACTS/'audit.json');p=read(ARTIFACTS/'protocol.json')
    for tag in ('frozen_v13','retuned_v13'):
        if g['models'][tag]['calibrated']['alarms']!=r['models'][tag]['calibrated']['alarms']:raise ValueError('The supposed matched baselines differ')
    lines=['# 캡스톤 디자인 II — 선행 탐지율 개선 학습·검증 종합 결과','',
        '작성일:2026-11-03. 두 단계 총12개 신규 모델 학습. 대상은 첫 중계드론의 패킷 서비스 장애다.','',
        f'최종 사전 선정안: **{NAMES[selected]}**. 같은42비행에서 선행 탐지율 **{pct(a["event_recall_3s"])} → {pct(b["event_recall_3s"])} ({diff["event_recall_difference"]*100:+.2f}%p)**. 경보 정밀도 {pct(a["precision"])} → {pct(b["precision"])}. 실제 오경보·늦은 경보 {a["false_or_late_per_drone_hour"]:.2f} → {b["false_or_late_per_drone_hour"]:.2f}회/드론·h.','',
        f'탐지 사건: {a["event_hits_3s"]}/{a["eligible_directed_link_events"]}건 → {b["event_hits_3s"]}/{b["eligible_directed_link_events"]}건. 같은FA4 절차의 문턱 대조군은 {pct(control["event_recall_3s"])}다. 선정안과 기존 고정 설정의 탐지율 차이95% 공간 그룹 재표집 구간은 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]%p다.','',
        ('최종 선정안은 기존 모델의 문턱 재보정이다. 새 모델 학습의 성능 향상으로 해석하지 않는다.' if selected=='retuned_v13' else '신규 후보는 선택 자료에서 선정했다. 확인 성능을 보고 후보나 문턱을 다시 고르지 않았다.'), '',
        '기존 개발 목표인 정밀도90%·탐지율80%·오경보1회/h는 미달이다. 이번 사전 선정안은 기존 고정 설정보다 탐지율이 낮아 현재 탐지 우선 프로필로 자동 채택하지 않는다.','',
        '## 1. 진행 내용','',
        '- 1단계: 드론별34항목·방향성 링크별4항목의 짧은 이력과20초 이력을 비교하고 그래프 자체를 학습했다. 일반/사건/20초 사건 학습을 각 두 난수·2400업데이트로 비교했다. 새 그래프가 선택 자료에서 기존 모델을 앞서지 못해 미채택했다.',
        '- 2단계: 집계 과정에서 사라질 수 있는 개별 RF 관측을 유지했다. 각 상대 드론의 관측48항목·관측 마스크48항목·현재 발행 의도9항목·측정 에너지6항목·에너지 유효성6항목·역할2항목을 사용했다. 총119항목·1초 간격20표본이다.',
        '- 상대 드론별 시간 기억을 공통64차원GRU가 학습하고, 현재 그래프 표현을 질의로 사용해 중요한 상대의 정보를 모은다. 집계 이력 사건 학습, 개별 이력 일반 학습, 개별 이력 사건 학습을 같은 초기 상태·배치 순서·4000업데이트·각 두 난수로 비교했다.',
        '- 이미 학습된V13 예측과V12 그래프 표현을 고정했다. 새 보정층의 초기 출력은0이며 기존 위험도에서 출발한다. 미래RF·미래 간섭·실제 미래 명령·패킷 정답은 입력하지 않았다.128바이트 실제 수신 비컨과 로컬 관측을 유지했다.',
        '- 훈련245비행에서148,949시점을 구성했다. 최적화에는125,440개 자연 시간 표본과2,393개 완전한 사건 전 구간을 사용했다. 사건 전3~7초 중 적어도 한 번 위험을 높이는 손실과 자연 표본의 오경보 억제 근사를 결합했다. 근사 손실이 실제 정밀도를 보장하지 않는다.',
        '- 사용자 선택인 탐지 우선·오경보 약4회/h를 모델 선택과 보정에 반영했다. 선택45비행과 보정45비행을 분리했다. 원래 개발 목표인 정밀도90%·탐지율80%·오경보1회/h와 구분한다.',
        '- 기존PPO 자료 생성 정책은 고정했다.12개 신규 모델은 예측기 지도학습이다. PPO나 실제 장비를 자동 교체하지 않았다.','',
        '## 2. 같은42비행의 전체 비교','',
        '| 후보 | 선행 탐지율 | 탐지 사건 | 정밀도 | 맞은 최초 경보/전체 | 실제 오경보/h |','|---|---:|---:|---:|---:|---:|']
    for tag,m in models.items():
        x=m['calibrated']['alarms'];label=NAMES[tag]+(' **최종 사전 선정**' if tag==selected else '')
        lines.append(f'| {label} | {pct(x["event_recall_3s"])} | {x["event_hits_3s"]}/{x["eligible_directed_link_events"]} | {pct(x["precision"])} | {x["timely_correct_first_alerts"]}/{x["first_alerts"]} | {x["false_or_late_per_drone_hour"]:.2f} |')
    lines+=['','기존 고정 설정은 사용자가 선택한 문턱0.33을 보존했다. 나머지는 동일FA4 보정 절차다. 이전31.61%(55/174건)는 다른42비행이므로 이번 결과에서 직접 빼지 않는다. 경보 비용도 함께 판단한다.','',
        '| 후보 | 5초 AP | 3~7초 AP | 3~7초 AUROC | 보정 Brier |','|---|---:|---:|---:|---:|']
    for tag,m in r['models'].items():lines.append(f'| {NAMES[tag]} | {pct(m["raw"]["next5_on_common_currently_up_links"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["auroc"])} | {m["calibrated"]["calibrated_any3to7"]["brier"]:.5f} |')
    lines+=['','AP·AUROC는 위험 순위 지표다. 정답률이 아니다. AUROC가 높아도 드문 사건의 선행 탐지율과 경보 정밀도는 낮을 수 있다.','',
        '## 3. 비교 독립성과 검증','',
        f'- 2단계 계획 고정 시각: {p["created_at"]}. 당시1단계 확인 입력이24/42회 처리된 상태였고, 확인 성능 요약 파일은0개였다. 실제 첫 확인 성능 요약 생성 시각: {audit["first_V14confirmation_summary_at"]}.',
        '- 확인 비행은 총42회다. 두 단계가 같은 알려진14지도·세 간섭 조건의 비행을 공유한다.84회로 중복 계산하지 않는다. 비행은1단계 모델 고정 후 생성됐으며,2단계 모델 고정 후 새로 생성한42회로 주장하지 않는다.',
        '- 2단계 설계·훈련·모델 선택·문턱 보정에는1단계 확인 성능을 사용하지 않았다.2단계 모델·소스·문턱 고정 후 확인 성능 열람을 허용했다. 같은 관측·정답·입력 자격·최초 경보·사건 분모·3≤선행시간<7초를 유지했다.',
        f'- {audit["source_flights"]}비행의 개별 관측 이력을 원본에서 다시 계산해 정확히 대조했다. 원래 의존 파일{audit["original_dependencies_preserved"]}개, 이전V13 납품 파일{audit["previous_delivery_files_preserved"]}개를 보존했다. 원래 최종32지도는 사용하지 않았다.',
        f'- 실제 실행API: {runtime["frames"]:,}프레임·{runtime["tested_predictions"]:,}개 위험도 검사. 개별 관측 이력 차이0, 최초 경보 판단 차이{runtime["first_alert_mismatches"]}건, 최대 확률 차이{runtime["errors"]["probability"]:.8g}. RF 결측은None이다.',
        f'- 데스크톱CPU 추론 중앙값{runtime["cpu_ms"]["warm_median"]:.2f}ms·최대{runtime["cpu_ms"]["warm_max"]:.2f}ms. 탑재 장치 지연 인증이나1Hz 실시간 보장과 구분한다.',
        f'- 선정 문턱:{policy["threshold"]}. 보정 규칙:{policy["rule"]}. 보정 경보:{policy["selected"]["first_alerts"]}개. 보정FA4는 이후 비행에서도4회/h 이하라는 하드 상한이 아니다.','',
        '## 4. 한계와 고찰','',
        '- 선택 자료에서 우위를 보이지 않은 구조를 새 확인 결과만으로 채택하지 않았다. 확인 결과의 최고 행을 사후 선택한 수치를 최종 선정 성능으로 제시하지 않는다.',
        '- 긴 이력, 그래프 직접 학습, 개별 관측 보존을 대조했다. 이들 구조의 효과는 지표와 비용으로 판단한다. 긴 학습 시간만으로 충분한 전조가 생긴다는 근거는 없다.',
        '- 1단계 선택 자료에서 누락의 주원인은3~7초 전 위험 점수 미달이었다. 경보가 이미 켜져 있어 새 최초 경보가 없던 경우는 소수였다. 이 사실만으로 관측 가능한 전조가 없다고 단정하지 않는다.',
        '- 알려진14지도의 새 무선 실현 결과다. 실제 드론·통신모듈·안테나·혼잡·미지 지형·분산 경로 동기화의 실측 검증은 없다. 실사용 신뢰 확보와 시뮬레이션 개선을 구분한다.','',
        '## 용어·근거','',
        '- 서비스 장애: 최근1초의 마감 경과 패킷20개 이상에서250ms 내 도착률90% 미만 상태가1초 지속되기 시작한 사건.',
        '- 선행 탐지율: 실제 사건 중 발생3초 이상·7초 미만 전에 유효 최초 경보한 비율.',
        '- 경보 정밀도: 최초 경보 중 정해진 선행 구간에 맞은 경보 비율.',
        '- FA/h: 오경보와 늦은 경보를 드론·시간으로 나눈 값. 팀 전체 경보 수와 구분한다.',
        '- 근거: 두 실험의 protocol/training_plan/freeze/confirmation_access/final_results/audit/runtime_checks.json과 실제 학습 이력.',
        f'- 1단계 결과 SHA256: `{sha(GRAPH/"final_results.json")}`.',f'- 2단계 결과 SHA256: `{sha(ARTIFACTS/"final_results.json")}`.','']
    (ROOT/'RESULTS_20261103.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
