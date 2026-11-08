"""Readable Korean report from immutable completed-comparison artifacts."""
from pathlib import Path
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now
from .calibration import verify_freeze


def pct(x):return '해당 없음' if x is None else f'{x*100:.2f}%'
def pp(a,b):return f'{(b-a)*100:+.2f}%p'
def number(x):return '해당 없음' if x is None else f'{x:.2f}'


def main():
    verify_freeze();result=read(ARTIFACTS/'final_results.json');done=read(ARTIFACTS/'completed_comparison.json')
    audit=read(ARTIFACTS/'audit.json');runtime=read(ARTIFACTS/'selected_runtime_checks.json')
    stress=read(ARTIFACTS/'stress'/done['selected_tag']/'results.json')
    choice=read(done['selection_decision']);old=result['baseline']['raw'];new=result['selected']['raw']
    oc=result['baseline']['calibrated'];nc=result['selected']['calibrated'];op=read(done['baseline_alarm']);np=read(done['selected_alarm'])
    text=['# 캡스톤 디자인 II — 예측 성능 개선 실험 결과',f'작성: {now()}. 기존 보존·별도 학습·모델 동결·개발 확인 평가 완료.',
        '\n## 판정',f"선정 모델: **{done['selected_tag']}**. 기존 비교 기준: **v7 64회 최선 체크포인트**. 기존 176회 모델도 보존했다.",
        f"같은 확인 42비행에서 5초 AP는 {pct(old['next5_on_common_currently_up_links']['ap'])}→{pct(new['next5_on_common_currently_up_links']['ap'])}, "
        f"3~7초 AP는 {pct(old['any3to7_on_common_support']['ap'])}→{pct(new['any3to7_on_common_support']['ap'])}다.",
        f"개발 경보 목표 동시 충족: **{'충족' if nc['trusted_simulation_alarm_enabled'] else '미충족'}**. 실제 기체·통신모듈 검증: **미실시**. 실사용 검증 완료로 판정하지 않는다.",
        'AP는 경보 적중률 또는 사건 발견률이 아니다. 실제 경보 정밀도·사건 재현율·오경보는 아래 별도 표를 따른다.',
        '\n[한눈에 보는 비교 그림](artifacts/figures/confirmation_summary.png) · [확률 신뢰도 그림](artifacts/figures/probability_reliability.png)',
        '\n## 1. 무엇을 바꿨는가',
        '1. 현재 연결 링크의 첫 단절뿐 아니라 회복 후 재단절도 학습하도록 시간 구간별 정답을 추가했다. 예측 출력은 매초 갱신한다.',
        '2. PPO 보상 경사를 분리하고 예측 정답을 직접 학습했다. 기존 PPO의 비행은 고정하여 예측기끼리 같은 관측에서 비교했다.',
        '3. 통신 변화·결측·관측 경과 시간·같은 관측자의 주변 링크를 입력에 반영했다.',
        '4. 공개 명목 경로와 받은 과거 위치로 진행도를 추정하고, 이후 이동에 대한 지도 손실 대안을 추가했다. 실제 미래 위치·미래 PPO 명령은 입력하지 않았다.',
        '5. 유휴 채널 에너지 측정 시제품과 수신 비콘 전달 경로를 추가했다. 센서는 가상이며 하드웨어 지원이 확인된 상태가 아니다.',
        '6. LightGBM·잔차 Transformer·연속 RF 보조 정답·앙상블을 실제 학습·비교했다. 추가 복잡성이 성능을 높이지 못한 후보도 결과에 남겼다.',
        '\n## 2. 독립 확인 성적 — 같은 정답·같은 관측',
        '아래 AP·AUROC·Brier는 전체 공통 관측을 합친 값이다. 5초 지표는 현재 연결된 링크에 한정하여 기존 예측기와 같은 정답으로 비교했다.',
        '\n| 지표 | 기존 v7 | 새 후보 | 차이 |\n|---|---:|---:|---:|']
    for key,label,metric in [('next5_on_common_currently_up_links','5초 AP','ap'),('any3to7_on_common_support','3~7초 AP','ap'),
                             ('next5_on_common_currently_up_links','5초 AUROC','auroc'),('any3to7_on_common_support','3~7초 AUROC','auroc')]:
        a=old[key][metric];b=new[key][metric];text.append(f'| {label} | {pct(a)} | {pct(b)} | {pp(a,b)} |')
    for key,label in [('next5_on_common_currently_up_links','5초 Brier ↓'),('any3to7_on_common_support','3~7초 Brier ↓')]:
        a=old[key]['brier'];b=new[key]['brier'];text.append(f'| {label} | {a:.5f} | {b:.5f} | {b-a:+.5f} |')
    text.extend(['\n**원래 공통 문턱 0.5의 최초 경보**. 선행 3~7초, 이탈 문턱은 진입 문턱의 절반. 반복 단절과 관측 부족 사건도 사건 분모에 포함한다.',
        '\n| 경보 지표 | 기존 | 새 후보 |\n|---|---:|---:|'])
    for field,label in [('precision','경보 정밀도'),('event_recall_3s','3초 이상 선행 사건 재현율')]:
        text.append(f"| {label} | {pct(old['strict_alarms_at_half'][field])} | {pct(new['strict_alarms_at_half'][field])} |")
    for field,label in [('false_or_late_per_drone_hour','오경보·늦은 경보/중계 드론·시간'),('first_alerts','최초 경보 수'),('eligible_directed_link_events','평가 방향 링크 사건 수')]:
        a=old['strict_alarms_at_half'][field];b=new['strict_alarms_at_half'][field]
        av=number(a) if field=='false_or_late_per_drone_hour' else f'{a:,}'
        bv=number(b) if field=='false_or_late_per_drone_hour' else f'{b:,}'
        text.append(f"| {label} | {av} | {bv} |")
    text.extend(['\n**보정 자료에서 정한 문턱을 그대로 적용한 확인 성적**. 위 0.5 표와 다른 운영점이다. 경보를 줄여 얻은 높은 정밀도와 놓친 사건을 함께 읽어야 한다.',
        '\n| 경보 운영점 | 기존 | 새 후보 | 개발 목표 예 |\n|---|---:|---:|---:|',
        f"| 고정 보정 문턱 | {op['threshold']:.4f} | {np['threshold']:.4f} | 보정 자료에서 결정 |",
        f"| 경보 정밀도 | {pct(oc['alarms']['precision'])} | {pct(nc['alarms']['precision'])} | 90% 이상 |",
        f"| 3초 이상 선행 사건 재현율 | {pct(oc['alarms']['event_recall_3s'])} | {pct(nc['alarms']['event_recall_3s'])} | 80% 이상 |",
        f"| 오경보·늦은 경보/중계 드론·시간 | {number(oc['alarms']['false_or_late_per_drone_hour'])} | {number(nc['alarms']['false_or_late_per_drone_hour'])} | 1 이하 |",
        f"| 실제 최초 경보 수 | {oc['alarms']['first_alerts']:,} | {nc['alarms']['first_alerts']:,} | 최소 50 |",
        f"| 전체 목표 충족 | {'충족' if oc['trusted_simulation_alarm_enabled'] else '미충족'} | {'충족' if nc['trusted_simulation_alarm_enabled'] else '미충족'} | 모든 조건 동시 |",
        '\n개발 목표는 교수 지정치 또는 실사용 인증 기준이 아니다. 에너지 후보의 보정 경보는 신선한 자기·이웃 측정이 있을 때만 허용했다. 이 조건 때문에 경보를 낼 수 없는 실제 단절도 사건 분모에 남겼다.',
        '각자 보정 자료에서 고정한 운영점이다. 기존 모델은 연구용 정밀도 80%·오경보 5회/시간 이하 조건도 만족하지 못해 보정 확률 0.5 부근으로 돌아간 값을 표시했다. 새 후보의 낮은 오경보와 낮은 재현율을 모두 보고한다. 이 표를 동일 오경보 예산에서의 성능 우위로 읽으면 안 된다.',
        '\n고정밀·낮은 탐지율 보조 운영점도 보정 자료에서 미리 정했다. 확인 결과는 다음과 같다. 이 지점의 존재는 전체 목표 충족을 뜻하지 않는다.'])
    for label,value in [('기존',oc.get('high_precision_partial_operating_point')),('새 후보',nc.get('high_precision_partial_operating_point'))]:
        text.append(f"- {label}: "+(f"정밀도 {pct(value['precision'])}, 사건 재현율 {pct(value['event_recall_3s'])}, 오경보·늦은 경보 {number(value['false_or_late_per_drone_hour'])}/중계 드론·시간, 경보 {value['first_alerts']:,}개." if value else '보정 자료에서 정밀도·오경보·최소 경보 조건을 만족하는 운영점 없음.'))
    text.extend(['\n확률 보정의 별도 확인이다. Brier·ECE는 작을수록 좋다. ECE는 확률 구간을 나누는 방법에 따라 변한다. 표본 단위 확률 신뢰도이며 최초 경보 정밀도와 다르다.',
        '\n| 모델·출력 | 원래 Brier | 보정 후 Brier | 보정 후 ECE |\n|---|---:|---:|---:|'])
    for label,raw,cal in [('기존',old,oc),('새 후보',new,nc)]:
        for output,raw_key,cal_key,rel_key in [('5초','next5_on_common_currently_up_links','calibrated_next5_on_common_currently_up_links','reliability_next5'),
                                              ('3~7초','any3to7_on_common_support','calibrated_any3to7_on_common_support','reliability_any3to7')]:
            text.append(f"| {label} {output} | {raw[raw_key]['brier']:.5f} | {cal[cal_key]['brier']:.5f} | {cal[rel_key]['ece']:.5f} |")
    text.extend(['\n## 3. 개선의 근거 — 모델 선택 자료만 사용',
        '모델 선택 15지도×3조건=45비행. 7개 학습 후보와 기존 모델, 최고 트리·신경망의 가중 평균 3개를 비교했다. 확인 성적을 이용한 재선택은 하지 않았다.',
        '\n| 후보 | 5초 AP | 3~7초 AP | 선택 점수 |\n|---|---:|---:|---:|'])
    for item in sorted(choice['candidates'],key=lambda x:x['score'],reverse=True):
        s=read(item['summary']);text.append(f"| {item['tag']} | {pct(s['next5_on_common_currently_up_links']['ap'])} | {pct(s['any3to7_on_common_support']['ap'])} | {item['score']:.5f} |")
    text.extend(['\n선택 점수는 두 AP의 산술평균이다. 1위와 75% 트리 혼합의 점수 차이는 약 0.00016으로 작다. 특정 모델 계열의 일반적 우월성을 뜻하지 않는다. 초기 TCN의 낮은 성능은 별도 예비 기록에 보존했다. 이를 최종 확인 성적처럼 섞지 않았다. 기존 대비 전체 차이에는 정답·특징·학습 방식 변경이 함께 포함되므로 어느 한 변경만의 인과 효과라고 단정하지 않는다.',
        '\n## 4. 지역·간섭 조건·불확실성',
        '다음 AP는 양성 사건이 있어 AP를 정의할 수 있는 비행별 AP의 평균이다. 양성 없는 비행의 AP를 0으로 채우지 않았다. 비행 수는 구조화 결과의 defined_flights로 함께 기록했다. 위 전체 관측 통합 AP와 집계 방식이 다르다.',
        '\n| 구분 | 지도 수 | 기존 5초 AP | 새 5초 AP | 기존 3~7초 AP | 새 3~7초 AP |\n|---|---:|---:|---:|---:|---:|'])
    labels={'legacy_geographies':'기존 지도','added_geographies':'추가 지도','clear':'무간섭','intermittent':'한 구간 간섭','continuous':'지속 간섭'}
    for key,group in result['groups'].items():
        a=group['baseline'];b=group['selected'];text.append(f"| {labels[key]} | {a['regions']} | {pct(a['macro_flight_ap5'])} | {pct(b['macro_flight_ap5'])} | {pct(a['macro_flight_ap37'])} | {pct(b['macro_flight_ap37'])} |")
    ci=result['paired_uncertainty'];text.append(f"\n확인 지도는 {ci['regions']}개이며 인접·중첩을 묶은 공간 그룹은 {ci['geography_clusters']}개다. 같은 공간 그룹의 비행과 방향 링크를 함께 묶어 10,000회 짝지은 재표집을 수행했다. 통계량은 공간 그룹별 AP의 평균 차이다.")
    for key,label in [('macro_spatial_group_ap5_difference','5초 AP'),('macro_spatial_group_ap37_difference','3~7초 AP')]:
        value=ci['metrics'][key]
        if value['point'] is not None:
            lo,hi=value['percentile95'];text.append(f"- {label}: 평균 차이 {value['point']*100:+.2f}%p, 재표집 95% 구간 [{lo*100:+.2f}, {hi*100:+.2f}]%p. AP 정의 가능한 공간 그룹 {value['defined_spatial_groups']}개.")
    o=result['event_observability'];text.extend([f"\n확인 사건 {o['events']:,}개 중 기존 관측 자격으로 경보 기회가 있었던 비율은 {pct(o['possible_recall_ceiling']['legacy'])}, 센서 자격까지 요구하면 {pct(o['possible_recall_ceiling']['energy_qualified'])}다. **완벽한 예측기를 가정한 입력 자격 상한**이며 달성 재현율이 아니다.",
        '\n## 5. 측정 오차와 실행 검증',
        '선택 자료의 기존 지도 1개와 추가 지도 1개, 각 세 간섭 조건에서 수행한 소규모 진단이다. 실제 장비의 오차 분포나 전 지역 강건성 성적이 아니다.',
        '\n| 측정 조건 | 비행 평균 5초 AP | 비행 평균 3~7초 AP | 고정 보정 경보 사건 재현율 |\n|---|---:|---:|---:|'])
    stress_labels={'clean':'원래 관측','energy_missing':'에너지 센서 없음','energy_noise_2p5db':'에너지 추가 잡음 σ=2.5dB','position_bias_2m':'고정 위치 편차: 축별 σ=2m'}
    for profile in stress['profiles']:
        text.append(f"| {stress_labels[profile['profile']]} | {pct(profile['macro_case_ap5'])} | {pct(profile['macro_case_ap37'])} | {pct(profile['calibrated_alarms']['event_recall_3s'])} |")
    text.extend(['\n위치 편차 시 현재 지도 손실·공개 경로·거리 등 관련 특징을 같은 오차 좌표에서 다시 계산했다. 깨끗한 지도 기하를 몰래 재사용하지 않았다. 물리적 비행과 단절 정답은 고정했으므로 전체 제어계 오차 시험은 아니다.',
        f"\n실행 검사: 배치·실시간 확률 최대 차이 {runtime['batch_stream_max_abs_probability_error']:.3g}. 필수 RF 관측 결측은 위험도 `None`. 에너지 누락 시 센서 자격 경보 차단. 미검증 신뢰 경보 비활성. CPU 첫 호출 {runtime['cpu_ms']['cold']:.2f}ms, 이후 중앙값 {runtime['cpu_ms']['warm_median']:.2f}ms. 데스크톱 결과이며 탑재 장치 지연 인증이 아니다.",
        f"\n보존·출처 감사: 원시 기록 {audit['raw_archives_hash_verified']:,}개 해시, 기존 의존 파일 {audit['original_dependencies_hash_verified']:,}개 해시, 학습·개발 공간 그룹 분리, 원시/캐시/정답 대조, 저장 예측의 새 추론 재현을 통과했다. 기존 최종 평가 지도 32개는 계속 봉인한다.",
        '\n## 6. 부족한 점과 다음 개선',
        '1. **경보 목표의 동시 충족**: AP 상승만으로 높은 선행 탐지율과 낮은 오경보가 동시에 확보되지 않는다. 보정으로 순위 성능을 새로 만들 수 없다.',
        '2. **미래 이동 의도**: 공개 명목 경로는 도움이 됐다. 이후 실제 PPO 재배치·보호 정지는 알 수 없다. 실제 수신 가능한 몇 초 뒤 의도를 명확히 하고, 제어 계획 변경·지연 정보를 함께 학습할 필요가 있다.',
        '3. **실측 무선 환경**: 기체·모듈·주파수를 정한 뒤 RSSI/SINR·유휴 에너지·채널 점유율·송수신 시각을 기록해야 한다. 현재 2.4GHz·50m·고정 속성의 가정은 다른 하드웨어에 자동 적용되지 않는다.',
        '4. **위치·자세·지도 오차**: GNSS 정확도, 고도 기준과 지형, 기체 자세에 따른 안테나 이득·기체 차폐, 건물 높이·수목·이동 장애물을 검증해야 한다. 추정 오차를 확률 또는 품질 정보로 입력하는 구성이 필요하다.',
        '5. **서비스 단절 정답**: 현재 정답은 이론적 홉 성공확률 문턱이다. 실제 서비스에서는 연속 패킷 유실·PDR·지연 기준과 연결해야 한다. 단절 정의를 바꾸면 기존 성적과 직접 비교하지 않고 새 시험으로 보고해야 한다.',
        '6. **새 제어계 확인**: 이번 예측기가 비행을 바꾸지는 않았다. 기존 연결 유지율·완주 시간의 상승을 새로 달성했다고 주장하지 않는다. PPO에 교체 연결한 뒤에는 동일 조건의 별도 폐루프 평가가 필요하다.',
        '\n## 7. 결과의 범위와 보존',
        '이번 훈련은 245개 지도·319비행·535,200표본이다. 모델 선택 45비행, 보정 45비행, 독립 개발 확인 42비행을 사용했다. 초기 검증 조건 치우침은 확인 전 88비행을 보완하여 바로잡았다.',
        '초기 예측기에는 과거 검증 비행 8개의 학습 이력이 있다. 기존 지역 전체가 완전히 처음 보는 지형이었다고 표현하지 않는다. 추가 지도 결과와 기존 지도 결과를 분리했다. 이번 확인 자료는 기존 봉인 32개 최종 시험과 다르다.',
        '\n## 8. 용어와 근거 파일',
        '- AP: 여러 문턱의 정밀도·재현율을 요약한 순위 지표. 경보 적중 확률과 다름. AUROC도 정확도와 다름.',
        '- 경보 정밀도: 최초 경보 중 3~7초 후 실제 단절과 일치한 비율.',
        '- 사건 재현율: 모든 지속 단절 중 3초 이상 먼저 최초 경보가 있었던 비율.',
        '- Brier: 확률 오차, 낮을수록 좋음. 희귀 사건에서는 다른 지표와 함께 판정.',
        '- %p: 백분율 차이. 60%→65%는 +5%p.',
        '\n설계·환경·문헌 근거: [METHODS.md](METHODS.md). 실행 방법: [QUICK_START.md](QUICK_START.md).',
        '\n| 근거 | 파일 |\n|---|---|',
        '| 전체 구조화 성적·조건별 성적·불확실성 | `artifacts/final_results.json` |',
        '| 모델 선택 기록 | `artifacts/completion/selection_decision.json` |',
        '| 모델·보정·평가 코드 동결 | `artifacts/model_alarm_freeze.json` |',
        '| 동일 관측 비교의 정확한 원시 결과 경로 | `artifacts/completed_comparison.json` |',
        '| 원본 보존·자료·추론 재현 감사 | `artifacts/audit.json` |',
        '| 실시간 출력 검증 | `artifacts/selected_runtime_checks.json` |',
        f"| 측정 오차 진단 | `artifacts/stress/{done['selected_tag']}/results.json` |"])
    chunks=[];last_table=False
    for block in text:
        table=block.lstrip().startswith('|')
        chunks.append(('\n' if table and last_table else '\n\n')+block)
        last_table=table
    (ROOT/'RESULTS_20261102.md').write_text(''.join(chunks).lstrip()+'\n',encoding='utf-8')
    bundle=dict(created_at=now(),mode='research_only',model=done['selected_model'],model_sha256=sha(done['selected_model']),
        alarm_policy=done['selected_alarm'],alarm_policy_sha256=sha(done['selected_alarm']),
        freeze=str(ARTIFACTS/'model_alarm_freeze.json'),freeze_sha256=sha(ARTIFACTS/'model_alarm_freeze.json'),
        runtime_api='forecast_rebuild.runtime_stable.StableLocalForecast',runtime_adapter_sha256=sha(ROOT/'runtime_stable.py'),
        production_ready=False,hardware_validated=False,automatic_ppo_replacement=False)
    write(ARTIFACTS/'research_candidate.json',bundle)
    print(ROOT/'RESULTS_20261102.md')


if __name__=='__main__':main()
