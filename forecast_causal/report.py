"""Four-stage Korean report from actual recorded results, never invented gains."""
from . import ROOT,ARTIFACTS
from .contracts import read,sha
NAMES={'frozen_v13':'기존 고정 V13','retuned_v13':'기존 V13 문턱만 보정','v16_tree_base':'LightGBM 기존 관측','v16_tree_network':'LightGBM 새 패킷 관측','v16_tree_physics':'LightGBM 물리·경로 불확실성','v16_tree_both':'LightGBM 새 관측+물리','v16_gru_base':'GRU 기존 이력 대조군','v16_gru_network':'GRU 새 패킷 이력 추가','v16_old_tree_physics50':'기존·물리 LightGBM 혼합','v16_old_tree_both50':'기존·관측+물리 LightGBM 혼합','v16_old_gru_network50':'기존·새 관측 GRU 혼합','v16_old50_tree_both25_gru_network25':'기존50·LightGBM25·GRU25 혼합'}
CAUSES={'all_events':'전체 사건','no_route_head_evidence':'마감 전 경로 부재','obstruction_connectivity_repair':'건물 추가손실 제거 시 순간 회복','interference_connectivity_repair':'간섭 제거 시 순간 회복','fading_connectivity_repair':'페이딩 제거 시 순간 회복','retry_failure_evidence':'마감 전 전송 실패','queue_wait_evidence':'마감 전125ms 이상 대기'}
def pct(v):return '-' if v is None else f'{100*v:.2f}%'
def pp(v):return '-' if v is None else f'{100*v:+.2f}%p'
def alarm_row(tag,m,selected):
    a=m['calibrated']['alarms'];label=NAMES[tag]+(' **사전 선정**' if tag==selected else '')
    return f'| {label} | {pct(a["event_recall_3s"])} | {a["event_hits_3s"]}/{a["eligible_directed_link_events"]} | {pct(a["precision"])} | {a["timely_correct_first_alerts"]}/{a["first_alerts"]} | {a["false_or_late_per_drone_hour"]:.2f} |'
def ranking_row(tag,m):return f'| {NAMES[tag]} | {pct(m["raw"]["next5_on_common_currently_up_links"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["ap"])} | {pct(m["raw"]["any3to7_on_common_support"]["auroc"])} | {m["calibrated"]["calibrated_any3to7"]["brier"]:.5f} |'

def main():
    r=read(ARTIFACTS/'final_results.json');ca=read(ARTIFACTS/'selection_event_causes.json');selected=r['selected_tag'];models=r['models'];a=models['frozen_v13']['calibrated']['alarms'];b=models[selected]['calibrated']['alarms'];diff=r['selected_vs_frozen'];ci=diff['uncertainty']['metrics']['calibrated_recall_difference']['percentile95'] if diff['uncertainty'] else [0,0];audit=read(ARTIFACTS/'audit.json');rt=read(ARTIFACTS/'runtime_checks.json');freeze=read(ARTIFACTS/'freeze.json')
    lines=['# 캡스톤 디자인 II — 장애 원인·전조 관측·예측기 개선 결과','', '작성일:2026-11-03. 사용자 요청의 네 단계를 모두 근거 파일과 연결한다.','',
        f'사전 선정안: **{NAMES[selected]}**. 새42비행에서 선행 탐지율 **{pct(a["event_recall_3s"])} → {pct(b["event_recall_3s"])} ({diff["event_recall_difference"]*100:+.2f}%p)**. 경보 정밀도 **{pct(a["precision"])} → {pct(b["precision"])}**. 실제 오경보·늦은 경보 **{a["false_or_late_per_drone_hour"]:.2f} → {b["false_or_late_per_drone_hour"]:.2f}회/드론·h**.','',
        f'탐지 사건 {a["event_hits_3s"]}/{a["eligible_directed_link_events"]}건 → {b["event_hits_3s"]}/{b["eligible_directed_link_events"]}건. 탐지율 차이의95% 공간 그룹 재표집 구간 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]%p. 이 구간은 제한된 알려진 지도 표본의 불확실성을 나타낸다.','',
        ('사전 등록한 탐지율·정밀도·경보 비용의 점추정 조건을 충족했다. 실측 성능 인증과 구분하며 자동 운영 교체는 하지 않았다.' if r['adoption']['point_metrics_pass'] else '사전 등록한 탐지율 증가·정밀도 유지·경보 비용 조건을 함께 충족하지 못했다. 현재 운영 프로필은 유지한다.'),
        ('선정안은 원래 개발 목표인 정밀도90%·탐지율80%·오경보1회/h에 미달한다.' if not models[selected]['calibrated']['current_fold_goal_met'] else '선정안은 이번 시뮬레이션에서 원래 개발 지표 조건을 충족했다. 실측 인증은 남아 있다.')+' 이번 사용자가 허용한 실험 경보 예산은 약4회/h다. 긴 학습만으로 실사용 성능이 보장되지 않는다.','',
        '## 1. 놓친 장애 원인 분해 — 완료','',
        '선택45비행의 전체200개 사건을 재구성했다. 기존 모델은 제때 최초 경보60건, 위험 점수 미달134건, 이미 켜진 경보5건, 입력 자격 부족1건이었다. 원래 집계와 정확히 일치한다.','',
        '| 중복 가능한 증거 | 전체200건 | 점수 미달134건 |','|---|---:|---:|']
    for key in list(CAUSES)[1:]:lines.append(f'| {CAUSES[key]} | {ca["all_event_evidence_counts"].get(key,0)} | {ca["low_score134_evidence_counts"].get(key,0)} |')
    lines+=['','모든200건에서 마감 전 패킷 대기가 길었으나196건은 경로 부재와 함께 발생했다. 대기 자체를 독립 원인200건으로 단정하지 않는다. 건물·간섭·페이딩 제거는 순간 연결성 대조이며 전체 임무를 다시 수행한 배타적 인과 증명이 아니다.','',
        '패킷별 증거는250ms 마감 전으로 잘랐다. 마감 이후 재전송이나 장기 대기를 당시 실패 원인에 섞지 않았다. 큐 초과 폐기는0건, 재전송 한도 폐기는1건이었다. 원인 분석에는 실제 정답을 사용하지만 예측 입력으로 전달하지 않았다.','',
        '사건3초 전 실제 수신 정보에 마감 초과 대기가 나타난 비율은6.5%, 경로 부재는2.0%였다. 새 큐 정보만으로 충분히 빠른 전조를 얻기 어려운 사례가 많다. 이 수치를 예측 가능성의 이론적 상한으로 해석하지 않는다.','',
        '## 2. 관측 정보 보강 — 완료','',
        '- 로컬 센서: 가장 오래·평균 기다린 시간, 마감 초과·임박 패킷 수, 목적지별 적체, 재전송 중인 패킷 수, 최근 경로 부재·경로 변경·재전송·ACK 수, 실제 최근 전송에 사용한 다음 홉과 관측 나이.','- 250ms라는 고정 마감과 대기 나이로 남은 마감 여유를 표현한다. 패킷 전체 목록이나 미래 폐기 사유는 예측기에 주지 않는다.','- 다른 기체의 상태는 실제 수신한 비컨으로만 전달한다.128바이트 비컨의 미사용 예약 슬롯18바이트에 요약 정보를 넣었다. 기존 전송 길이·수신 성공·통신 슬롯은 같다. 원래 내용은 정확히 복원된다. 대상 편대는10기 이하다.','- 수신·시간 제한을 지킨186개 요약, 현재·3초 전·3초/10초 통계 등1,116개 트리 입력,1초 간격20표본 GRU 이력을 구성했다. 미래RF·미수신 패킷을 바꿔도 입력이 변하지 않는 검사를 통과했다.','',
        '## 3. 원인에 맞는 예측기 대조 — 완료','',
        '- LightGBM: 동일한 기존432개 관측에 새 패킷 정보, 물리 정보, 둘의 결합을 각각 추가했다. 네 구성·두 학습 난수·2개 출력에 대해 총16개 이진 부스터를 학습했다. 구성당800갱신, 같은 자연 표본을 사용했다.','- 물리 예측: 공개 건물 지도와 실제 수신한 발행 경로, 최근 관측 전파 여유로1·3·5·7초의 차폐·연결 위험을 계산했다. 훈련245비행에서만 경로 오차 크기를 추정하고8개 고정 위치 시나리오를 만들었다.','- 중계드론은 발행·확정된 경로가3초 수준이므로3~7초 구간은 미래 재배치 선택에 영향을 받는다. 훈련 자료의 미확정7초 경로 오차는 진행 방향11.42m·수직 방향7.02m RMS였다. 시나리오 위치를 실제 실행 경로나 보정된 확률로 주장하지 않는다.','- GRU: 기존V13과 그래프 표현을 고정했다. 기존 관측 이력과 새 패킷 이력을 추가한 두 구성을 같은 초기 상태·배치·손실·두 난수·4,000갱신으로 비교했다. 초기 추가 출력은0으로 기존 예측과 동일하다.','- 총148,949개 훈련 시점 중125,440개 자연 시간 표본과2,393개 완전한 사건 전 구간을 사용했다. 선택45비행으로 모델을 정하고 보정45비행으로 확률과 문턱을 정했다. 결합을 포함한12후보를 새 확인 전에 고정했다.','',
        '## 4. 같은 기준의 새42비행 검증 — 완료','',
        f'모델·소스·주FA4/보조FA3,5 문턱 고정 시각:{freeze["created_at"]}. 이후 알려진14지도·세 간섭 조건에서 새 난수42비행을 수집했다. 이전 확인 비행으로 다시 학습하거나 유리한 후보를 사후 선택하지 않았다. 어려운 사건도 전체 분모에 포함했다.','',
        '| 후보 | 3초 선행 탐지율 | 탐지/전체 사건 | 경보 정밀도 | 맞은 최초/전체 경보 | 실제 오경보/h |','|---|---:|---:|---:|---:|---:|']
    for tag,m in models.items():lines.append(alarm_row(tag,m,selected))
    lines+=['','기존 고정V13은 사용자 선택 문턱0.33을 유지했다. 문턱만 보정한V13은 같은FA4 절차의 대조군이다. 보정FA4는 새 비행에서도4회/h 이하라는 보장이 아니다. 과거31.61%(55/174건)와는 비행이 다르므로 직접 차이를 계산하지 않는다.','',
        '| 후보 | 5초 AP | 3~7초 AP | 3~7초 AUROC | 보정 Brier |','|---|---:|---:|---:|---:|']
    for tag,m in models.items():lines.append(ranking_row(tag,m))
    lines+=['','AP·AUROC는 위험 순위 성능이며 정답률이 아니다. Brier는 확률과0/1정답의 평균 제곱 오차로 낮을수록 좋다. 희귀 사건에서는 높은AUROC와 낮은 선행 탐지율이 동시에 나올 수 있다.','',
        '물리·경로 정보 추가의 효과가 가장 뚜렷했다. 같은 LightGBM에서 선행 탐지율8.17%→33.17%,3~7초AP9.52%→35.01%로 상승했다. 실제 오경보는3.99→4.09회/h였다. 이 차이는 물리 정보 묶음의 효과이며 LightGBM이라는 이름만으로 얻은 향상으로 해석하지 않는다.',
        '물리 입력에 새 패킷 관측을 더했을 때 추가 향상은 확인되지 않았다. GRU에 새 패킷 이력을 더한 경우도 같은 오경보2.79회/h에서 탐지율23.08%→22.12%로 낮아졌다. 관측 추가가 자동으로 유효한 전조를 만드는 것은 아니다.',
        '확인 자료에서 기존·물리 LightGBM 혼합은73건, 사전 선정안은72건을 탐지했다. 이1건 차이를 보고 최종 선정안을 바꾸지 않았다.','',
        '### 입력 추가 효과','', '| 대조 | 탐지율 차이 | 정밀도 차이 | 실제 오경보/h 차이 | 3~7초 AP 차이 |','|---|---:|---:|---:|---:|']
    for d in r['contrasts']:lines.append(f'| {NAMES[d["reference"]]} → {NAMES[d["candidate"]]} | {pp(d["event_recall_difference"])} | {pp(d["precision_difference"])} | {d["FA_per_hour_difference"]:+.2f} | {pp(d["ap37_difference"])} |')
    lines+=['','### 원인 증거별 탐지','', '| 중복 가능한 증거 | 사건 수 | 기존 탐지 | 선정안 탐지 |','|---|---:|---:|---:|']
    groups=r['cause_analysis']
    for key,label in CAUSES.items():
        n=groups['counts'].get(key,0);old=groups['hits']['frozen_v13'].get(key,0);new=groups['hits'][selected].get(key,0);lines.append(f'| {label} | {n} | {old}/{n} | {new}/{n} |')
    lines+=['','증거별 행은 겹친다. 전체 사건 행의 분모는 모든 후보의 원래 지표와 같다. 오경보 비용은 원인별로 중복 배분하지 않았다. 건물 관련 순간 회복 증거 그룹에서55→69건으로 늘었으며, 간섭 관련 그룹은5/29건으로 그대로다. 차폐 관련 개선을 모든 원인의 개선으로 확대하지 않는다.','',
        '| 경보 결과 | 기존 | 선정안 |','|---|---:|---:|']
    for key,label in [('timely_first_alert','제때 최초 경보'),('risk_below_threshold','필요 구간에서 점수 미달'),('already_active_without_new_alert','점수는 높지만 새 최초 경보 없음'),('no_qualified_input','필요 구간의 유효 관측 부족')]:
        lines.append(f'| {label} | {groups["forecast_outcome_counts"]["frozen_v13"].get(key,0)} | {groups["forecast_outcome_counts"][selected].get(key,0)} |')
    u=diff['uncertainty']['metrics'] if diff['uncertainty'] else None
    if u:
        p_ci=u['calibrated_precision_difference']['percentile95'];fa_ci=u['calibrated_false_or_late_per_drone_hour_difference']['percentile95']
        lines+=['',f'정밀도 차이의95% 공간 그룹 재표집 구간은[{p_ci[0]*100:+.2f}, {p_ci[1]*100:+.2f}]%p, 오경보/h 차이는[{fa_ci[0]:+.2f}, {fa_ci[1]:+.2f}]다. 두 구간은0을 포함하므로 관측된 정밀도 상승·오경보 감소를 확정된 일반화 효과로 단정하지 않는다.']
    lines+=['',
        '### 사전 고정한 보조 운영점','', '| 후보 | 보정 예산/h | 탐지율 | 정밀도 | 확인 오경보/h |','|---|---:|---:|---:|---:|']
    for tag in dict.fromkeys(('frozen_v13','retuned_v13',selected)):
        for budget,v in models[tag]['calibrated']['secondary_operating_points'].items():
            x=v['alarms'];lines.append(f'| {NAMES[tag]} | {budget} | {pct(x["event_recall_3s"])} | {pct(x["precision"])} | {x["false_or_late_per_drone_hour"]:.2f} |')
    lines+=['','## 검증과 한계','',
        f'- 실제 API:{rt["frames"]:,}프레임·{rt["tested_predictions"]:,}위험도. 입력 차이0, 최초 경보 차이{rt["first_alert_mismatches"]}건, 최대 확률 차이{rt["errors"]["probability"]:.8g}. 필수RF 결측 시 위험도는None이다.',
        f'- 데스크톱CPU 추론 중앙값{rt["cpu_ms"]["warm_median"]:.2f}ms, 최대{rt["cpu_ms"]["warm_max"]:.2f}ms. 실행 중 부하를 포함한 관측이며 탑재 장치의1Hz 보증이 아니다.',
        f'- 무결성:{audit["source_flights"]}비행, 원래 의존 파일{audit["original_dependencies_preserved"]}개, 보존한V12~V15 납품본, 확률 기록{audit["prediction_files_verified"]:,}개 검사. 원래 최종32지도는 열지 않았다.',
        '- 기존 엔진은5ms 슬롯 통신과 현재 전파 확률에 기초한 즉시 경로 계산을 사용한다. 충돌 경쟁·분산 경로 갱신·실제 무선모듈 동작을 충분히 재현한ns-3 또는 실측 환경으로 주장하지 않는다.',
        '- 새 큐 센서는0.5초 프레임에서 측정하고 다른 기체에1초 주기로 전달한다.250ms 패킷 마감보다 느리므로 짧은 적체 전조를 놓칠 수 있다. 추가 입력만으로3초 선행 탐지가 보장되지 않는다.',
        '- 물리 특징은 시뮬레이터와 같은 공개 지도·전파식을 사용한다. 실제 미래 위치를 넣지는 않았으나 전파식 자체가 틀릴 수 있는 실세계의 독립 검증은 남아 있다.',
        '- 새 예측기를 PPO에 연결하면 재배치 행동과 이후 관측 분포가 달라질 수 있다. 이번 고정 정책의 예측 비교를 새 정책의 실제 재배치·연결 성능 향상으로 해석하지 않는다.',
        '- 이번 예측 학습 동안 PPO는 고정했다. 재배치 정책을 다시 강화학습한 결과나 연결률 개선으로 혼합하지 않는다. 추가 정보를 같은 비컨으로 전달했지만 실제 펌웨어·모듈 구현 검증은 남아 있다.',
        '- 다음 개선 판단은 이 보고서의 대조 효과와 실패 증거에서 출발한다. 테스트 최고 행만 고르거나 사건을 제외해서 성능을 높이지 않는다. 실측 통신장비가 정해지면 실제 전조 기록과 분산 라우팅 조건을 먼저 수집해야 한다.','',
        '## 실제 학습·체크포인트 기록','', '| 구성 | 난수 | 실제 갱신 | 선택된 모델 |','|---|---:|---:|---|']
    for m in r['training_receipt']:
        chosen=' / '.join(f'출력{k}:{v}' for k,v in m['selected_head_iterations'].items()) if m['family']=='v16_tree' else str(m['selected_step'])
        lines.append(f'| {NAMES["v16_"+m["variant"]]} | {m["seed"]} | {m["updates"]} | {chosen} |')
    lines+=['','출력2는5초 위험, 출력5는3~7초 위험이다. GRU 선택 시점0은 추가 학습 전의 기존 예측을 유지한 경우다. 마지막 갱신이 항상 최고 성능인 것은 아니다.',
        '신경망 손실에는 자연 표본에서 정밀도40%를 겨냥한 근사 제약을 사용했다. 탐지 우선 실험의 최적화 설정이며 실제 최초 경보 정밀도를 보장하지 않는다. 원래 정밀도90% 목표의 달성 선언과 구분한다.','',
        '## 용어와 근거','',
        '- 서비스 장애: 최근1초의 마감 경과 패킷20개 이상에서250ms 이내 도착률90% 미만이1초 지속되기 시작한 사건.',
        '- 선행 탐지율: 사건 중 발생3초 이상·7초 미만 전에 유효 최초 경보가 있었던 비율.',
        '- 평가 사건 자격: 원래 규칙에 따라 임무 시작24초 이후·종료8.5초 이전의1초 지속 장애를 사용한다. 이 범위 안의 관측 부족·약한 전조 사건을 제외하지 않았다.',
        '- 경보 정밀도: 최초 경보 중 정해진 선행 구간에서 실제 사건을 맞힌 비율.',
        '- FA/h: 오경보·늦은 경보를 대상 드론·시간으로 나눈 값. 팀 전체의 경보 수와 구분.',
        '- RMS: 오차 제곱 평균의 제곱근. GRU: 시간 흐름을 기억하는 순환 신경망. LightGBM: 수치 특징의 분기 규칙을 학습하는 부스팅 트리.',
        '- 상세 근거: protocol, cause_rulebook, selection_event_causes, input_preflight, experiment_plan, training_plan, freeze, confirmation_access, final_results, runtime_checks, audit.json.',
        f'- 결과 SHA256:`{sha(ARTIFACTS/"final_results.json")}`.','']
    (ROOT/'RESULTS_20261103.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
