"""Concise formal Korean report generated from exact archived outcomes."""
from . import ROOT,ARTIFACTS
from .contracts import read,sha

NAMES={'frozen_v13':'기존 탐지 우선 설정','retuned_v13':'기존 모델 문턱만 재보정','v14_short_bce':'짧은 그래프 일반 학습','v14_short_event':'짧은 그래프 사건 학습',
       'v14_long_event':'20초 그래프 사건 학습','v14_short_event_blend50':'기존·짧은 그래프 50% 혼합','v14_long_event_blend50':'기존·20초 그래프 50% 혼합'}
def pct(x):return '-' if x is None else f'{x*100:.2f}%'

def main():
    r=read(ARTIFACTS/'final_results.json');selected=r['selected_tag'];models=r['models'];old=models['frozen_v13']['calibrated']['alarms'];new=models[selected]['calibrated']['alarms'];retuned=models['retuned_v13']['calibrated']['alarms']
    delta=r['selected_vs_frozen']['event_recall_difference']*100;ci=r['selected_vs_frozen']['uncertainty']['metrics']['calibrated_recall_difference']['percentile95'] if r['selected_vs_frozen']['uncertainty'] else [0,0]
    diagnosis=read(ARTIFACTS/'selection_diagnosis.json')['models'][selected];causes=diagnosis['causes']
    f=read(ARTIFACTS/'freeze.json');pol=read(next(x['policy'] for x in f['candidates'] if x['candidate']['tag']==selected));audit=read(ARTIFACTS/'audit.json');runtime=read(ARTIFACTS/'runtime_checks.json')
    text=['# 캡스톤 디자인 II — 20초 드론·링크 그래프 학습 결과','', '작성일: 2026-11-03. 대상: 첫 중계드론의 패킷 서비스 장애. 새42비행·알려진14지도·세 간섭 조건의 동일 관측 비교.','',
        '새 그래프6개를 실제 학습했으나 별도 선택 자료에서 기존 예측기를 앞서지 못했다. 사전 선정안은 기존 모델의 문턱 재보정이다. 선정안의 지표 변화는 새 신경망 학습의 성능 향상으로 해석하지 않는다.','',
        f'사전 선정안: **{NAMES[selected]}**. 선행 탐지율 **{pct(old["event_recall_3s"])} → {pct(new["event_recall_3s"])} ({delta:+.2f}%p)**. 경보 정밀도 {pct(old["precision"])} → {pct(new["precision"])}. 실제 오경보·늦은 경보 {old["false_or_late_per_drone_hour"]:.2f} → {new["false_or_late_per_drone_hour"]:.2f}회/드론·h.','',
        f'탐지 사건 {old["event_hits_3s"]}/{old["eligible_directed_link_events"]}건 → {new["event_hits_3s"]}/{new["eligible_directed_link_events"]}건. 같은FA4 보정 절차를 적용한 기존 모델의 탐지율은 {pct(retuned["event_recall_3s"])}다.','',
        f'기존 고정 설정 대비 탐지율 차이의 공간 그룹 짝지은 재표집95% 구간: [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]%p. 제한된 지리 표본의 불확실성이다.','',
        '## 1. 변경 내용과 채택 근거','',
        '- 이전 모델은 20초 관측을 드론·역할별 통계480개로 집계했다. 개별 드론·링크 그래프는 짧은 이력으로 계산한 채 고정했다.',
        '- 이번 모델은 드론별34항목과 방향성 링크별4항목의1초 간격20표본을 유지한다. 드론 간 관계를 계산하는 그래프와 시간 기억을 함께 학습했다.',
        '- 현재 발행 경로와 관측 기반 물리 예측을 유지했다. 실제 받은128바이트 비컨만 사용했다. 전송량 증가, 미래RF·미래 간섭 일정·정답 입력은 없다.',
        '- 기존V12 모델의 같은 초기 가중치에서 일반 학습, 사건 학습,20초 이력 사건 학습을 각각 두 난수로2400회 업데이트했다. 총6개 모델을 실제 학습했다.0회 초기 모델도 선택 후보에 포함했다.',
        '- 훈련245비행에서148,949시점을 구성했다. 최적화에는 그중 자연 시간 표본125,440개와 완전한 사건 전 구간2,393개를 사용했다. 사건 전3~7초 기회 중 적어도 한 번 위험을 높이는 손실과 정상 구간 오경보 억제 손실을 결합했다.',
        '- 사용자가 선택한 탐지 우선·오경보 약4회/h를 이번 학습의 모델 선택 기준으로 반영했다. 선택45비행에서 모델·체크포인트를 정하고 보정45비행에서 확률과 문턱을 정했다. 확인42비행은 모든 모델·소스·문턱 고정 후 새 난수로 생성했다.',
        '- PPO 자료 생성 정책과 기존 파일을 보존했다. 이번6개 모델은 지도학습이다. 재배치 성과 향상이나 실제 장비 성능으로 보고하지 않는다.','',
        '시간 기억과 그래프 결합의 설계 동기는 [Rossi 등,2020](https://arxiv.org/abs/2006.10637)을 참고했다. 본 구현은 고정 간격 스냅숏GRU이며 해당 논문의TGN을 그대로 재현한 것은 아니다. 사건 탐지 손실은 [Rath·Hughes,2022](https://proceedings.mlr.press/v151/rath22a.html)의 탐지율·정밀도 제약 관점을 참고한 자체 근사다. 연구 성능의 자동 이전을 전제하지 않는다.','',
        '## 2. 같은 새42비행의 전체 후보','', '| 후보 | 선행 탐지율 | 탐지 사건 | 정밀도 | 맞은 최초 경보/전체 | 실제 오경보/h |', '|---|---:|---:|---:|---:|---:|']
    for tag,m in models.items():
        a=m['calibrated']['alarms'];label=NAMES[tag]+(' **사전 선정**' if tag==selected else '')
        text.append(f'| {label} | {pct(a["event_recall_3s"])} | {a["event_hits_3s"]}/{a["eligible_directed_link_events"]} | {pct(a["precision"])} | {a["timely_correct_first_alerts"]}/{a["first_alerts"]} | {a["false_or_late_per_drone_hour"]:.2f} |')
    text+=['','기존 고정 설정은 사용자가 선택한V13의 문턱0.33을 보존했다. 나머지 후보는 같은FA4 보정 절차를 적용했다. 확인 결과로 후보·문턱을 다시 선택하지 않았다. 이전 보고의31.61%(55/174건)는 다른 비행 결과이므로 이번 수치와 직접 빼지 않는다.','',
        '| 후보 | 5초 AP | 3~7초 AP | 3~7초 AUROC | 보정 Brier(3~7초) |','|---|---:|---:|---:|---:|']
    for tag,m in models.items():text.append(f'| {NAMES[tag]} | {pct(m["raw"]["next5_on_common_currently_up_links"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["auroc"])} | {m["calibrated"]["calibrated_any3to7"]["brier"]:.5f} |')
    text+=['','AP와AUROC는 위험 순위 지표다. 정답률이나 사건 선행 탐지율로 해석하지 않는다. AUROC가 높아도 드문 사건의 경보 정밀도·선행 탐지율은 낮을 수 있다.','', '## 3. 학습 요인과 간섭 조건','']
    for d in r['contrasts']:text.append(f'- {NAMES[d["reference"]]} → {NAMES[d["candidate"]]}: 탐지율 {d["event_recall_difference"]*100:+.2f}%p, 정밀도 {d["precision_difference"]*100:+.2f}%p, 오경보 {d["FA_per_hour_difference"]:+.2f}/h. 각 보정 문턱의 실제 비용을 함께 비교한다.')
    text+=['','| 간섭 조건 | 기존 탐지율 | 선정안 탐지율 | 선정안 정밀도 | 선정안 오경보/h |','|---|---:|---:|---:|---:|']
    for kind,label in [('clear','추가 방해 없음'),('intermittent','간헐 방해'),('continuous','지속 방해')]:
        a=r['condition_alarms'][kind]['frozen_v13'];b=r['condition_alarms'][kind][selected];text.append(f'| {label} | {pct(a["event_recall_3s"])} | {pct(b["event_recall_3s"])} | {pct(b["precision"])} | {b["false_or_late_per_drone_hour"]:.2f} |')
    text+=['','## 4. 선택 자료의 누락 원인','',
        f'모델·문턱 고정 후 별도 선택45비행의{diagnosis["events"]}개 사건을 사후 분석했다. 확인 비행은 이 분석에 사용하지 않았다. 선정안은{causes.get("timely_first_alert",0)}개 사건을 제때 경보했다.{causes.get("risk_below_threshold",0)}개는3~7초 전 점수가 문턱 미만이었고,{causes.get("already_active_without_new_alert",0)}개는 이미 켜진 경보 때문에 새 최초 경보가 없었으며,{causes.get("no_qualified_input",0)}개는 입력 자격이 없었다. 경보 상태 초기화만 바꾸는 것이 주된 해결책이라는 근거는 없다. 이 분석으로 관측 가능한 전조가 없다고 단정하지 않는다.','',
        '## 5. 판정과 검증 범위','',
        f'- 선정안 보정 문턱: {pol["threshold"]}. 보정 규칙: {pol["rule"]}. 보정 최초 경보: {pol["selected"]["first_alerts"]}개. 보정FA4는 이후 비행에서도4회/h 이하라는 보장이 아니다.',
        '- 원래 개발 목표인 정밀도90%·탐지율80%·오경보1회/h는 별도 목표로 유지한다. 이번에는 사용자 선택에 따라 약4회/h 비용에서 탐지율을 우선했다. 두 기준의 달성 여부를 혼합하지 않는다.',
        f'- 전체 {audit["source_flights"]}비행의 자료·정답·입력시각·해시를 검사했다. 원래 의존 파일 {audit["original_dependencies_preserved"]}개와V12·V13 납품 파일을 보존했다. 원래 최종32지도는 사용하지 않았다.',
        f'- 실제 실행API를 {runtime["frames"]:,}프레임에서 검사했다. 드론·링크20표본 입력 차이0, 최초 경보 판단 차이 {runtime["first_alert_mismatches"]}건. 최대 확률 차이 {runtime["errors"]["probability"]:.8g}. RF 결측은None을 반환한다.',
        f'- 데스크톱CPU 추론 중앙값 {runtime["cpu_ms"]["warm_median"]:.2f}ms, 최대 {runtime["cpu_ms"]["warm_max"]:.2f}ms. 새 비행 동시 수집 부하에서 측정했다.1Hz 실시간 보장이나 실제 탑재 장치 지연 인증과 구분한다.',
        '- 알려진14지도의 새 무선 실현 검증이다. 미지 지형·실제 드론·통신모듈·안테나·혼잡·분산 경로 동기화의 실측 검증은 없다. 자동PPO 교체와 신뢰 경보는 활성화하지 않았다.',
        '- 사건의 전조가 관측되지 않는 구간은 긴 학습만으로 해결되지 않는다. 반대로 이번 모델 결과만으로 예측 가능성의 상한을 단정하지 않는다.','',
        '## 용어·근거','',
        '- 서비스 장애 사건: 최근1초의 마감 경과 패킷이20개 이상인 시점에서,250ms 내 도착률이90% 미만인 상태가1초 지속되기 시작한 사건. 기존 정답 정의를 유지했다.',
        '- 선행 탐지율: 실제 사건 중 발생3초 이상·7초 미만 전에 유효 최초 경보한 비율.',
        '- 경보 정밀도: 발생한 최초 경보 중 정해진 선행 구간에 맞은 경보 비율.',
        '- FA/h: 오경보와 늦은 경보를 드론·시간으로 나눈 값. 기체가 여러 대면 팀 전체 경보 수와 구분한다.',
        '- 공동학습과 구분: 이번에는 예측기 내부의 그래프·시간 기억을 함께 지도학습했다. PPO 정책을 동시에 재학습하지 않았다.',
        '- 근거 파일: artifacts/protocol.json,training_plan.json,freeze.json,confirmation_access.json,final_results.json,selection_diagnosis.json,audit.json,runtime_checks.json.','',
        f'결과 SHA256: `{sha(ARTIFACTS/"final_results.json")}`.','']
    (ROOT/'RESULTS_20261103.md').write_text('\n'.join(text),encoding='utf-8')

if __name__=='__main__':main()
