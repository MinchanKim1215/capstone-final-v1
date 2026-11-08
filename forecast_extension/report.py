"""Korean report: continuation, new experience and fixed fresh-flight results."""
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now,OLD_TREE,OLD_NEURAL,OLD_SAMPLES
from .evaluate import verify_freeze,model_pin
from .runtime import runtime_dependencies


def pct(x):return '정의 불가' if x is None else f'{x*100:.2f}%'
def pp(a,b):return f'{(b-a)*100:+.2f}%p'
def num(x):return '없음' if x is None else f'{x:.2f}'


def main():
    frozen=verify_freeze();result=read(ARTIFACTS/'final_results.json');audit=read(ARTIFACTS/'audit.json');runtime=read(ARTIFACTS/'runtime_checks.json')
    stress=read(ARTIFACTS/'stress_results.json');choice=read(ARTIFACTS/'selection_decision.json');models=result['models']
    old_tag=read(OLD_TREE)['tag'];neural_tag=read(OLD_NEURAL)['tag'];selected=result['selected_tag'];old=models[old_tag];winner=models[selected]
    names={old_tag:'기존 최선 LightGBM',neural_tag:'기존 Transformer(4,750회)',
        'same_data_20000':'B: 동일 자료 연장 Transformer','new_data_20000':'C: 새 경험·오차 추가 Transformer',
        'same_data_final20000':'B: 고정20,000회 종점','new_data_final20000':'C: 고정20,000회 종점',
        'new_data_tree':'새 자료 LightGBM','new_data_tree_no_energy':'새 자료 LightGBM·센서 없음','frozen_v7_64':'기존 v7 공동학습 예측기',
        'v9_blend_25':'트리25% 혼합','v9_blend_50':'트리50% 혼합','v9_blend_75':'트리75% 혼합'}
    new_counts=read(ARTIFACTS/'samples/new_only/manifest.json')['counts'];old_counts=read(OLD_SAMPLES/'manifest.json')['counts'];samples=old_counts['train']+new_counts['train']
    b=read(ARTIFACTS/'models/same_data_20000/model.json');c=read(ARTIFACTS/'models/new_data_20000/model.json')
    chosen_meta=read(result['selected_model']) if result['selected_model']!='baseline' else {}
    composition=' + '.join(f"{names.get(read(member['path'])['tag'],read(member['path'])['tag'])} {member['weight']*100:.0f}%" for member in chosen_meta.get('members',[]))
    budget=winner['calibrated']['common_FA_budget1']
    text=['# 캡스톤 디자인 II — 연장 학습·새 경험 추가 비교 보고서',f'작성: {now()}. 세 단계의 학습·고정 비교·새 비행 확인·보존 감사 완료.',
        '\n## 판정',('이번20,000회 연장과 새 경험 추가로 개발 경보 목표를 충족했다.' if winner['calibrated']['trusted_simulation_alarm_enabled'] else '**이번20,000회 연장과 새 경험 추가만으로는 개발 경보 목표를 충족하지 못했다.** 순위 지표 상승과 실제 선행 경보 성능을 구분해야 한다.'),
        f"확인 전에 선정한 연구 후보: **{names.get(selected,selected)}** (`{selected}`).",
        ('후보 구성: '+composition+'.') if composition else '',
        f"같은 새 확인42비행에서 기존 최선 트리 대비 선정 후보의 5초 AP 차이는 {pp(old['raw']['next5_on_common_currently_up_links']['ap'],winner['raw']['next5_on_common_currently_up_links']['ap'])}, "
        f"3~7초 AP 차이는 {pp(old['raw']['any3to7_on_common_support']['ap'],winner['raw']['any3to7_on_common_support']['ap'])}다.",
        f"전체 개발 경보 목표의 보정·확인 동시 충족: **{'충족' if winner['calibrated']['trusted_simulation_alarm_enabled'] else '미충족'}**. 실제 장비 검증: **미실시**. 실사용 자동 경보를 활성화하지 않았다.",
        (f"보정에서 FA1/h 이하로 정한 운영점의 새 확인 결과: 경보 정밀도 **{pct(budget['precision'])}**, 3초 이상 선행 사건 재현율 **{pct(budget['event_recall_3s'])}**, 오경보·늦은 경보 **{num(budget['false_or_late_per_drone_hour'])}/중계드론·시간**. 전체 평가 사건{budget['eligible_directed_link_events']:,}개 중{budget['event_hits_3s']:,}개에 최초 경보를 제때 냈다." if budget else '보정 자료에서 FA1/h 조건의 최소 경보 수를 만족하는 운영점이 없었다.'),
        'AP는 여러 문턱의 정밀도·재현율을 요약한 순위 지표다. 실제 최초 경보 정밀도와 선행 사건 재현율은 별도 표를 따른다.',
        '\n[학습 곡선](artifacts/figures/learning_curves.png) · [새 비행 비교 그림](artifacts/figures/confirmation_comparison.png)',
        '\n## 1. 같은 자료에서 학습을 늘린 결과',
        '같은 5,000회 체크포인트의 가중치·옵티마이저에서 출발하여 20,000회까지 15,000회 추가 업데이트했다. 학습률은 250회 준비 구간 뒤 낮아지는 새 계획을 적용했다. 따라서 단순 벽시계 시간만의 효과가 아니라 **같은 자료의 연장 학습과 학습률 계획 조정** 결과다.',
        f"기존 훈련 표본 {old_counts['train']:,}개를 보존했다. B의 최고 고정 검증 점수는 {b['selected_step']:,}회, 마지막 학습은 {b['completed_steps']:,}회다. 마지막 모델과 최고 모델을 구분했다.",
        '\n| 고정 검증 표본의 업데이트 | 5초 AP | 3~7초 AP | AP 평균 |\n|---|---:|---:|---:|']
    history=read(ARTIFACTS/'models/same_data_20000/history.json');top=max(history,key=lambda r:r['selection']['score'])
    for row in [history[0],top,history[-1]]:
        text.append(f"| {row['step']:,}회 | {pct(row['selection']['next5']['ap'])} | {pct(row['selection']['any3to7']['ap'])} | {pct(row['selection']['score'])} |")
    text.extend(['\n원래 선택45비행 전체에서의 비교다. 체크포인트 선택에 사용한 개발 자료이므로 독립 시험 성적으로 표현하지 않는다.',
        '\n| 모델 | 전체 공통5초 AP | 전체 공통3~7초 AP | 원래0.5 사건R3 | 원래0.5 오경보·늦은 경보/h |\n|---|---:|---:|---:|---:|'])
    for path in result['stage1']['summaries']:
        s=read(path);a=s['strict_alarms_at_half'];text.append(f"| {names.get(s['model_tag'],s['model_tag'])} | {pct(s['next5_on_common_currently_up_links']['ap'])} | {pct(s['any3to7_on_common_support']['ap'])} | {pct(a['event_recall_3s'])} | {num(a['false_or_late_per_drone_hour'])} |")
    text.extend(['\n## 2. 추가한 경험과 관측 오차',
        f"새 비행377개: 훈련245, 선택45, 보정45, 확인42. 기존245개 훈련 지도에 새 비행을 하나씩 추가했다. 훈련 원시 비행은 기존319+새245=564개다. 새·기존 훈련 표본 합계는 **{samples:,}개**다.",
        '훈련84비행의 일부 표본에 관측 오차를 추가했다. 63비행에는 축별 표준편차1·2·4m 위치 편차를, 21비행에는 센서 오차만 추가했다. 에너지 측정에는 추가2.5dB 잡음 또는 누락을 사용했다. 이 수치는 장비 실측 분포가 아닌 시험 가정이다.',
        '관측 위치를 바꾸면 거리·반경속도·지도 손실·현재 명령 오프셋·공개 경로 특징을 같은 오차 좌표에서 다시 계산했다. 수신하지 않은 이웃 명령과 자기 자신에 대한 빈 항목은 유지했다. 단절 정답과 실제 비행 물리는 변경하지 않았다.',
        f"C도 같은5,000회 가중치·옵티마이저에서 같은15,000회 추가 업데이트를 수행했다. 최고 체크포인트는 {c['selected_step']:,}회다. 새 자료로 바뀐 미니배치 표집 난수도 다르므로 B/C 차이 전체를 자료 증가만의 순수 인과 효과로 단정하지 않는다.",
        f"같은 새 확인42비행에서 정확한20,000회 종점끼리 비교한 C−B 차이는 5초 AP {pp(models['same_data_final20000']['raw']['next5_on_common_currently_up_links']['ap'],models['new_data_final20000']['raw']['next5_on_common_currently_up_links']['ap'])}, 3~7초 AP {pp(models['same_data_final20000']['raw']['any3to7_on_common_support']['ap'],models['new_data_final20000']['raw']['any3to7_on_common_support']['ap'])}다. 이 순위 개선만으로 경보 탐지율 개선을 단정할 수 없다. 고정 운영점의 정밀도·사건 재현율·오경보는5절에 모두 표시했다.",
        '새 자료 LightGBM 두 후보는 에너지 사용 여부를 비교했다. 모든 예측기는 고정된 v7 PPO 비행의 동일 관측에서 평가했다. 이번에 PPO 재배치기를 다시 학습하거나 새 연결 유지율을 달성한 것은 아니다.',
        '\n## 3. 마지막 평가의 범위와 선택 규칙',
        '선택90비행은 기존45+새45다. 보정45와 확인42는 새 난수·기존과 반대 진행 방향의 비행이다. **확인 지도14개는 이미 알려진 지도**이므로 새 지형 일반화로 표현하지 않는다. 원래 봉인된 최종 지도32개는 계속 보존했다.',
        '초기 예측기에는 과거 검증 비행8개의 학습 노출 이력이 있다. 이번 표본의 공간 역할 분리는 이번 최적화 자료의 분리이며, 사전학습까지 포함한 완전 미노출 지형 성능을 뜻하지 않는다.',
        f"확인 전에 모든 후보·가중치·문턱·평가 코드를 고정했다. 기본 후보와 같은20,000회 종점 대조, 트리/신경망 혼합3개를 포함한 {len(models)}개 후보를 모두 한 번 확인했다. 확인 성적으로 모델을 다시 선택하지 않았다.",
        '모델 선택은 개발 목표 전체를 만족하면 선행 사건 재현율이 큰 후보, 그 조건을 만족하는 후보가 없으면 정밀도90%·오경보1회/중계드론·시간 이하·경보50개 이상 조건에서 사건 재현율이 큰 후보를 우선했다. 해당 운영점도 없으면 두 AP 평균으로 선택했다. AP 최고 후보는 별도로 기록했다.',
        f"적용 선택 규칙: `{choice['rule']}`. AP 최고 후보: `{choice['ranking_champion']}`. 선정 연구 후보: `{selected}`.",
        '고정 검증 표본에서 고른 최고 중간 모델과 정확한20,000회 종점을 모두 후보에 넣었다. 마지막 후보 선택은 전체 선택90비행으로 수행했으므로, 혼합 후보가 사용하는 신경망은 최고 중간 모델과 다를 수 있다.',
        '\n## 4. 고정 후보의 새 확인42비행 성적',
        '같은 관측·같은 정답·공통 관측 지원에서 비교한다. 5초 AP는 현재 연결된 링크에 한정한다. 아래 AP는 전체 관측 통합 값이다. 이전v8 보고서의 다른 난수·방향 성적과 직접 차감하지 않는다.',
        '\n| 모델 | 5초 AP | 3~7초 AP | 5초 AUROC | 원래0.5 사건R3 | 원래0.5 오경보·늦은 경보/h |\n|---|---:|---:|---:|---:|---:|'])
    for tag,model in models.items():
        raw=model['raw'];a=raw['strict_alarms_at_half'];label=names.get(tag,tag)+(' **선정**' if tag==selected else '')
        text.append(f"| {label} | {pct(raw['next5_on_common_currently_up_links']['ap'])} | {pct(raw['any3to7_on_common_support']['ap'])} | {pct(raw['next5_on_common_currently_up_links']['auroc'])} | {pct(a['event_recall_3s'])} | {num(a['false_or_late_per_drone_hour'])} |")
    text.extend(['\n## 5. 오경보 예산을 맞추려 한 고정 운영점',
        '각 모델의 **보정 자료에서 오경보·늦은 경보1회/h 이하와 경보50개 이상**을 만족한 지점 중 사건 재현율이 가장 큰 문턱을 정했다. 이 문턱을 새 확인에 그대로 적용했다. 확인에서 실제 오경보가1을 넘으면 초과로 표시한다. 확인에 맞춰 문턱을 다시 낮추거나 높이지 않았다.',
        '\n| 모델 | 확인 경보 정밀도 | 확인 선행 사건R3 | 확인 오경보·늦은 경보/h | 경보 수 | 예산 확인 |\n|---|---:|---:|---:|---:|---|'])
    for tag,model in models.items():
        p=model['calibrated']['common_FA_budget1'];label=names.get(tag,tag)
        if p is None:text.append(f'| {label} | 없음 | 없음 | 없음 | 없음 | 보정 운영점 없음 |');continue
        text.append(f"| {label} | {pct(p['precision'])} | {pct(p['event_recall_3s'])} | {num(p['false_or_late_per_drone_hour'])} | {p['first_alerts']:,} | {'유지' if p['false_or_late_per_drone_hour']<=1 else '초과'} |")
    p=winner['calibrated']['precision90_FA1_partial']
    text.append('\n선정 후보의 정밀도90%·FA1 보조 운영점 확인: '+(f"정밀도 {pct(p['precision'])}, 선행 사건R3 {pct(p['event_recall_3s'])}, 오경보·늦은 경보 {num(p['false_or_late_per_drone_hour'])}/h, 경보 {p['first_alerts']:,}개." if p else '보정 자료에 해당 조건과 최소 경보 수를 만족하는 문턱이 없었다.'))
    text.extend(['\n확률 자체의 오차도 같은 확인 관측에서 비교했다. Brier는 확률의 평균 제곱오차로 작을수록 좋다. ECE는 확률을10개 같은 너비 구간으로 나눈 평균 보정 오차다. 희귀 사건에서는 평균 오차가 작아도 높은 위험 확률 구간의 신뢰성이 보장되지는 않는다.',
        '\n| 후보·예측 범위 | 보정 전 Brier | 보정 후 Brier | 보정 후 ECE |\n|---|---:|---:|---:|'])
    for tag in dict.fromkeys([old_tag,selected]):
        model=models[tag]
        for raw_key,cal_key,reliability_key,label in [('next5_on_common_currently_up_links','calibrated_next5','reliability_next5','5초'),('any3to7_on_common_support','calibrated_any3to7','reliability_any3to7','3~7초')]:
            text.append(f"| {names.get(tag,tag)} · {label} | {model['raw'][raw_key]['brier']:.5f} | {model['calibrated'][cal_key]['brier']:.5f} | {model['calibrated'][reliability_key]['ece']*100:.2f}%p |")
    text.extend(['\n개발 전체 목표 예는 정밀도90%·3초 이상 선행 사건 재현율80%·오경보/늦은 경보1회/h 이하·경보50개 이상이다. 교수 지정치 또는 실사용 인증치가 아니다. 일부 조건만 충족했다고 전체 목표 통과로 표시하지 않는다.',
        '\n## 6. 불확실성과 관측 지원',
        '같은 지도의 방향 링크·인접 시각·세 간섭 조건은 독립 표본이 아니다. 인접·중첩 지도를 포함한12개 공간 그룹으로 묶어10,000회 짝지은 재표집을 수행했다. 통계량은 공간 그룹 AP 평균 차이다. 위 전체 통합 AP 차이의 신뢰구간과는 다르다.'])
    ci=result['paired_spatial_uncertainty_against_v8_tree']
    for key,label in [('macro_spatial_group_ap5_difference','5초 AP'),('macro_spatial_group_ap37_difference','3~7초 AP')]:
        item=ci['metrics'][key]
        if item['point'] is not None:
            low,high=item['percentile95'];text.append(f"- 선정 후보 대 기존 최선 트리 {label}: 평균 차이 {item['point']*100:+.2f}%p, 재표집95% 구간 [{low*100:+.2f}, {high*100:+.2f}]%p. 정의 가능한 공간 그룹 {item['defined_spatial_groups']}/{ci['geography_clusters']}개.")
    obs=result['event_observability'];text.append(f"\n입력 자격이 한 번이라도 있었던 사건 비율은 기존 지원 {pct(obs['possible_recall_ceiling']['legacy'])}, 에너지 지원까지 요구하면 {pct(obs['possible_recall_ceiling']['energy_qualified'])}다. 완벽한 예측기를 가정한 관측 자격 상한이며 달성 재현율이 아니다.")
    text.extend(['\n추가 범주 진단은 직전3초에 자기 송신 시도 또는 데이터 수신이 관측된 링크와 나머지를 구분한다. 실제 라우팅 핵심 링크 또는 전체 네트워크 단절을 직접 판정한 것이 아니다. 모델·문턱 선택에는 사용하지 않았다. AP는 양성 비율에 영향을 받으므로 범주 간 AP만으로 난이도나 실사용 성능을 단정하지 않는다.',
        '\n| 후보·범주 | 5초 질의 수 | 5초 양성 비율 | 5초 AP | 3~7초 AP |\n|---|---:|---:|---:|---:|'])
    for tag in dict.fromkeys([old_tag,selected]):
        for group,label in [('recent_data','최근 데이터 통신 관측'),('no_recent_data','최근 데이터 통신 없음')]:
            item=result['recent_data_diagnostic']['models'][tag][group]
            text.append(f"| {names.get(tag,tag)} · {label} | {item['next5']['samples']:,} | {pct(item['positive_fraction5'])} | {pct(item['next5']['ap'])} | {pct(item['any3to7']['ap'])} |")
    text.extend(['\n## 7. 측정 오차 진단과 실행 검증',
        '기존 지도1개·추가 지도1개의 새 선택 비행6개에서 비행당 최대6,000개 고정 질의로 비교했다. 전체 비행 경보 재현율 또는 독립 확인 성적이 아닌 선택 자료의 순위 강건성 진단이다.',
        '\n| 관측 조건 | 후보 | 질의 공통5초 AP | 질의 공통3~7초 AP |\n|---|---|---:|---:|'])
    for row in stress['profiles']:
        text.append(f"| {row['profile']} | {names.get(row['tag'],row['tag'])} | {pct(row['ap5'])} | {pct(row['ap37'])} |")
    text.extend([f"\n실시간 전체 비행 검사: {runtime['frames']:,}프레임, 지원 링크 예측 {runtime['tested_supported_link_predictions']:,}개. 배치 대비 최대 확률 차이 {runtime['max_abs_probability_error']:.3g}, 경보 판단 차이 {runtime['alarm_decision_mismatches']}개. 필수 RF 결측은 `None`, 미검증 자동 경보는 비활성이다.",
        f"데스크톱 CPU 첫 추론 {runtime['cpu_ms']['cold']:.2f}ms, 이후 중앙값 {runtime['cpu_ms']['warm_median']:.2f}ms. 탑재 장치 지연 인증이 아니다.",
        f"\n감사: 원시 물리 비행 아카이브 {audit['physical_archives_hash_verified']:,}개 해시, 원래 의존 파일 {audit['original_dependencies_preserved']:,}개 보존, 증강84비행의 정답·마스크·미수신 명령 보존, 훈련 인덱스의 train 역할, 공간 그룹 분리, 모든 고정 후보의 예측 캐시 재추론을 확인했다.",
        '\n## 8. 다음 개선과 해석 범위',
        '- 20,000회가 모델의 일반적인 학습 상한은 아니다. 이번 구조·학습률·자료·표집 조건 안에서의 결과다. 같은 자료 반복과 새 경험의 효과를 별도로 판단한다.',
        '- 고정된2.4GHz·50m 채널에서 새 비행과 관측 오차를 늘렸다. 실제 모듈의 변조·속도 적응·안테나·기체 자세·다중/이동 간섭·지형을 실측한 것은 아니다.',
        '- 현재 단절 정답은 이론적 양방향 홉 성공확률 문턱이다. 실제 PDR·연속 패킷 유실·지연의 서비스 단절 정답과 연결해야 한다.',
        '- 미래3초 이후의 실제 PPO 명령 변경과 전조 없는 새 간섭은 현재 이력만으로 정확한 시각까지 알 수 없다. 수신 가능한 계획·계획 변경·관측 품질 정보의 개선이 필요하다.',
        '- 새 예측기로 비행을 바꾸지 않았다. 실제 PPO 교체 뒤에는 별도 폐루프 비교가 필요하다. 기존 연결 유지율·완주 시간의 상승을 이번 성적으로 주장하지 않는다.',
        '\n## 근거와 실행 파일',
        '- 구조화 전체 성적: `artifacts/final_results.json`',
        '- 선택·고정·개봉 기록: `artifacts/selection_decision.json`, `artifacts/freeze.json`, `artifacts/confirmation_access.json`',
        '- 원본 보존 감사·실행 검사: `artifacts/audit.json`, `artifacts/runtime_checks.json`',
        '- 학습·실행 상태: 프로젝트 루트 `01_예측연장_현황.bat`',
        '- 동결 연구 후보: `artifacts/research_candidate.json`, `forecast_extension.runtime.load_candidate`',
        '- 정확한 원시 결과 경로와실험범위: `artifacts/completed_comparison.json`'])
    if audit['count_header_correction'].get('required'):
        text.append('- 확인 자료의 행 수 헤더가 이전 단계의0개로 복사된 메타데이터 오류를 실제42개로 정정했다. 원본 자료·원본 결과와 정정 기록은 `artifacts/metadata_correction/`에 보존했다. 모든 입력 행·모델·문턱·평가 수치는 동일하며, 정정 전후 해시 비교로 이를 검사했다.')
    chunks=[];last=False
    for block in text:
        table=block.lstrip().startswith('|');chunks.append(('\n' if table and last else '\n\n')+block);last=table
    (ROOT/'RESULTS_20261102.md').write_text(''.join(chunks).lstrip()+'\n',encoding='utf-8')
    entry=next(item for item in frozen['models'] if item['pin']['path']==frozen['selected_model'])
    write(ARTIFACTS/'research_candidate.json',dict(model=frozen['selected_model'],model_pin=model_pin(frozen['selected_model']),selected_tag=selected,
        policy=entry['policy'],policy_sha256=entry['policy_sha256'],freeze_sha256=sha(ARTIFACTS/'freeze.json'),runtime_sha256=sha(ROOT/'runtime.py'),
        runtime_dependencies=runtime_dependencies(),
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,mode='research_only',created_at=now()))
    print(ROOT/'RESULTS_20261102.md')


if __name__=='__main__':main()

