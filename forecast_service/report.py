"""Korean result report with task-matched baselines and unchanged denominators."""
from . import ARTIFACTS,ROOT
from .contracts import read,sha,now,V9
from .figures import NAMES


def pct(value):return '정의 불가' if value is None else f'{value*100:.2f}%'
def number(value):return '없음' if value is None else f'{value:.2f}'
def delta(a,b):return f'{(b-a)*100:+.2f}%p'


def main():
    result=read(ARTIFACTS/'final_results.json');audit=read(ARTIFACTS/'audit.json');runtime=read(ARTIFACTS/'runtime_checks.json')
    diagnosis=read(ARTIFACTS/'diagnostics/results.json');counts=diagnosis['event_outcomes'];total=sum(counts.values());manifest=read(ARTIFACTS/'samples/manifest.json')
    lines=['# 캡스톤 디자인 II — 사건 진단·패킷 서비스 예측 개선 보고서',f'작성: {now()}. 네 단계의 진단·구성·대조 학습·새 확인 평가와 감사를 완료했다.',
        '\n## 판정','기존 물리 링크 예측과 새 패킷 서비스 장애 예측은 정답과 사건 분모가 다르다. **두 과제의 AP·재현율을 직접 차감한 개선율은 사용하지 않는다.**']
    for task,label in [('link','기존 물리 링크 단절 과제'),('service','새 패킷 서비스 장애 과제')]:
        data=result['tasks'][task];old=data['models'][data['reference_tag']];new=data['models'][data['selected_tag']]
        lines += [f"- **{label}**: 확인 전에 선정한 `{data['selected_tag']}`. 같은 새42비행의 기준선 대비 5초 AP {delta(old['raw']['next5_on_common_currently_up_links']['ap'],new['raw']['next5_on_common_currently_up_links']['ap'])}, 3~7초 AP {delta(old['raw']['any3to7_on_common_support']['ap'],new['raw']['any3to7_on_common_support']['ap'])}. 개발 전체 경보 목표 {'충족' if new['calibrated']['trusted_simulation_alarm_enabled'] else '미충족'}."]
        if task=='link':lines.append('  운영 경보 기준에서 이전 혼합 모델이 그대로 선정됐다. 위0%p는 같은 모델을 기준선과 선정 후보로 표시한 결과다.')
        else:
            control=data['models']['v10_service_uniform']['raw']
            lines.append(f"  위 기준선은 기존 링크위험 최대값 집계다. 같은 서비스 정답을 직접 학습한 기본 대조군과 비교하면 5초 AP {delta(control['next5_on_common_currently_up_links']['ap'],new['raw']['next5_on_common_currently_up_links']['ap'])}, 3~7초 AP {delta(control['any3to7_on_common_support']['ap'],new['raw']['any3to7_on_common_support']['ap'])}다.")
        point=new['calibrated']['common_FA_budget1']
        if point:lines.append(f"  보정FA1/h 운영점의 확인: 정밀도{pct(point['precision'])}, 3초 이상 선행 최초 경보 재현율{pct(point['event_recall_3s'])}, 오경보·늦은 경보{number(point['false_or_late_per_drone_hour'])}/h. 사건{point['eligible_directed_link_events']:,}개 중{point['event_hits_3s']:,}개를 제때 경보했다.")
        else:lines.append('  보정 자료에서 FA1/h 이하·최소50개 경보를 함께 만족하는 운영점이 없었다.')
    lines += ['실제 드론·통신모듈 검증은 미실시다. 원래 코드·모델·보고서와 봉인 지도32개를 보존했다. 자동 PPO 교체와 실사용 경보는 활성화하지 않았다.',
        '\n[기존 링크 비교 그림](artifacts/figures/link_confirmation.png) · [새 서비스 비교 그림](artifacts/figures/service_confirmation.png) · [사건 타임라인](artifacts/diagnostics/timelines/INDEX.md)',
        '\n## 1. 실패 유형 진단',f"이전 고정 혼합 모델의 보정FA1 문턱0.77을 선택90비행에 적용했다. 전체 단절{total:,}개 중 위험 점수가 문턱에 이르지 않은 사건{counts['risk_below_entry_threshold']:,}개({pct(counts['risk_below_entry_threshold']/total)}), 경보 입력 자격이 없던 사건{counts['no_qualified_input']:,}개다.",
        f"최초 경보 재현율은{pct(diagnosis['groups']['all']['strict_recall'])}, 높은 위험 점수가 사전에 한 번이라도 있었던 비율은{pct(diagnosis['groups']['all']['high_risk_coverage_rate'])}다. 두 값의 차이가 작아, 기존 경보가 계속 켜져 있었다는 집계 문제만으로 낮은 탐지율을 설명하기 어렵다.",
        f"5초 이상 지속한 단절은{diagnosis['groups']['duration_ge5s']['events']:,}개다. 짧은 단절만 제외해서 해결된다고 판단하지 않았다. 최근 송수신이 관측된 링크는{diagnosis['groups']['recent_traffic']['events']:,}개이며, 해당 범주의 최초 경보 재현율은{pct(diagnosis['groups']['recent_traffic']['strict_recall'])}다.",
        '명령 변경·간섭 전환·전체 연결 하락과의 시간적 인접성은 events.csv에 기록했다. 이 연관성만으로 원인을 확정하지 않았다. 진단·수정에는 선택 자료만 사용했고, 이전 확인42비행을 새 학습 자료로 사용하지 않았다.',
        '\n## 2. 패킷 서비스 정답 구성','기존 과제는 이론적 양방향 홉 성공확률 문턱 아래 상태가1초 이상 지속된 물리 링크 단절의 시작을 예측한다. 이 정의와 경보 분모를 유지했다.',
        '새 과제는 수색 드론들이 생성한 패킷의 **250ms 내 전달률**이 최근1초 구간에서90% 미만인 상태가1초 이상 지속되는 시작을 예측한다. 각 시각에는 전달 마감시간이 이미 지난 패킷만 정답 계산에 포함했다. 마감시간250ms는 기존 시뮬레이터의 적시 전달 기준을 사용한 연구 가정이며, 실제 장비의 요구 규격이 아니다.',
        f"새 정답은 초기 구현부터250ms 경계에1e-10초(0.1나노초)의 수치 허용오차를 사용했다. 기존 엔진의 적시 전달 누계는 허용오차 없는 비교이므로 따로 검증했다. 경계에 걸린 패킷은{audit['roundoff_boundary_packets']:,}개/{audit['settled_packets_verified']:,}개다. 확인 비행의 기록 검사에서 이 두 조건을 혼용한 오류가 발생해, 같은42비행을 보존하고 누락된 패킷 기록만 정확 재생으로 복원했다. 모델·정답 계산식·문턱은 바꾸지 않았다.",
        '새 과제의 주 평가는 지정 중계드론1대(첫 중계드론)의 로컬 관측으로 수행했다. 다른 드론의 비공개 관측·미래 행동·미래 전파·정답 패킷 상태를 입력으로 주지 않았다. 이 출력은 향후 새로운 서비스 악화의 시작 위험이며 현재 서비스가 정상이라는 보증이 아니다.',
        '기존335비행을 같은 명령으로 재생했다. 전체 이동·물리 링크 시각열, 제어 경계의 로컬 관측·수신 명령, 최종 패킷 전달·유실·지연 결과가 원본과 일치했다. 새 확인 비행은 원래 정책으로 비행을 마친 뒤 패킷 정답만 복사해 기록했다. 패킷 로그 추가가 비행이나 정책에 영향을 주지 않도록 분리했다.',
        '\n## 3. 입력과 대조 학습',f"기존 링크 과제는 원래1,123,200표본·728개 특징을 그대로 사용했다. 서비스 과제는 기존 새훈련245비행과84개 관측오차 변형에서{manifest['counts']['train']:,}개 표본을 만들었다. 같은 물리 비행의 원본/변형 표본 위치는 겹치지 않으며 두 변형의 표본 할당 합계는 같다.",
        '서비스 기본 입력384개는 로컬 RF48항목·수신 명령9항목·선택적 에너지 측정·결측 여부의 현재/과거 요약과 공개 드론 수다. 추가48개 특징은 실제 수신한 수색 드론의 위치·속도·발행 명령·측정 나이와 공개 GIS로 계산한 현재·3초·7초의 직접 통신 가설이다. 미래 미발행 명령이나 실제 미래 위치는 사용하지 않았다.',
        '대조 순서: 링크 기본 가중치→사건 중심 가중치. 서비스 기본 입력/기본 가중치→같은 입력/사건 가중치→같은 가중치/통신 가설 입력 추가. 양성 가중치 상한8, 훈련 안에서 고른 어려운 정상 사례 가중치2를 사전 고정했다. 어려운 사례를 검증·확인 자료에서 채굴하지 않았다.',
        '각 변형을 두 난수2026110203·2026110213으로 학습하고 동일 비율 평균을 후보로 사용했다. 유리한 난수 하나를 고르지 않았다. 같은 최대800라운드·31리프·조기 종료60 조건을 사용했다. 조기 종료 때문에 실제 라운드 수는 모델마다 다르다.',
        '\n| 과제·변형·난수 | 선택 표본5초 AP | 선택 표본3~7초 AP |\n|---|---:|---:|']
    for item in read(ARTIFACTS/'trained_models.json')['candidates']:
        meta=read(item['model'])
        for member in meta['members']:
            m=read(member['path']);lines.append(f"| {item['task']} / {m['variant']} / {m['seed']} | {pct(m['selection']['2']['ap'])} | {pct(m['selection']['5']['ap'])} |")
    lines += ['上表는 조기 종료에 사용한 개발 표본 성적이다. 독립 확인 성적으로 해석하지 않는다.'.replace('上表','위 표'),
        '\n## 4. 모델·문턱 고정과 새 확인','기존 선택 자료에서 모델을 고르고 별도 보정45비행에서 확률과 문턱을 정했다. 전체 목표를 만족하는 후보가 있으면 사건 재현율, 그렇지 않으면 정밀도90%·오경보1회/h 이하·경보50개 이상 운영점의 재현율, 해당 후보도 없으면 두 AP 평균으로 선택했다.',
        '개발 전체 목표 예는 경보 정밀도90%·3초 이상 선행 최초 경보 재현율80%·오경보 및 늦은 경보1회/중계드론·시간 이하·경보50개 이상이다. 교수 지정값이나 실제 장비 인증 기준이 아니다.',
        '모든 후보·두 난수 평균·문턱·평가 구현을 고정한 뒤 새 난수의 확인42비행을 생성했다. 같은14개 기존 지도에서 세 간섭 조건을 사용했으므로 새로운 지형 일반화 성능은 아니다. 기준선 예측기와 자료 생성 정책에는 과거 검증 노출 이력이 있으며, 원래 최종 지도32개는 열지 않았다.',
        '기준선도 이번 새42비행에서 다시 평가했다. 따라서 이전 보고서의 다른 난수 비행에서 나온 기준선 수치와 달라질 수 있다. 비교는 같은 이번42비행 안에서 수행한다.',
        'AP는 정밀도·재현율을 여러 문턱에 걸쳐 요약한 순위 지표다. 5초 AP는 현재 정상인 공통 관측에 한정하고3~7초 AP는 전체 공통 관측에서 계산했다. 실제 경보 정밀도·사건 재현율은 따로 확인한다.']
    for task,label in [('link','5. 기존 물리 링크 단절 과제 — 원래 정의 유지'),('service','6. 새 패킷 서비스 장애 과제 — 별도 정답')]:
        data=result['tasks'][task];lines += ['\n## '+label,f"확인 전 선택: `{data['selected_tag']}`. 규칙: `{data['decision']['rule']}`.",
            '\n| 후보 | 5초 AP | 3~7초 AP | 보정FA1 운영점 정밀도 | 선행 사건 재현율 | 오경보·늦은 경보/h |\n|---|---:|---:|---:|---:|---:|']
        for tag,model in data['models'].items():
            raw=model['raw'];p=model['calibrated']['common_FA_budget1'];name=NAMES.get(tag,tag)+(' **선정**' if tag==data['selected_tag'] else '')
            tail=f"{pct(p['precision'])} | {pct(p['event_recall_3s'])} | {number(p['false_or_late_per_drone_hour'])}" if p else '운영점 없음 | 운영점 없음 | 운영점 없음'
            lines.append(f"| {name} | {pct(raw['next5_on_common_currently_up_links']['ap'])} | {pct(raw['any3to7_on_common_support']['ap'])} | {tail} |")
        lines.append('FA1 운영점은 보정 자료에서 정한 문턱을 그대로 적용한 결과다. 확인 오경보가1을 넘으면 예산 초과다. 운영점이 없으면 임의로 문턱을 바꾸지 않았다.')
        for key,value in data['one_change_ablations'].items():
            lines.append(f"- 동일 과제의 {NAMES.get(value['reference'],value['reference'])}→{NAMES.get(value['candidate'],value['candidate'])}: 통합5초 AP {value['pooled_ap5_difference']*100:+.2f}%p, 통합3~7초 AP {value['pooled_ap37_difference']*100:+.2f}%p.")
        uncertainty=data['selected_vs_reference_uncertainty']
        if data['selected_tag']==data['reference_tag']:lines.append('선정 후보와 기준선이 동일하므로 두 모델 간 신뢰구간을 제시하지 않는다. 새 대조 모델들의 차이는 위 표와 한 변수 대조 결과로 판단한다.')
        for key,label in ([] if data['selected_tag']==data['reference_tag'] else [('macro_spatial_group_ap5_difference','5초'),('macro_spatial_group_ap37_difference','3~7초')]):
            value=uncertainty['metrics'][key]
            if value['point'] is not None:
                lo,hi=value['percentile95'];lines.append(f"- 선정 후보 대 과제 기준선의 공간 그룹 평균{label} AP 차이 {value['point']*100:+.2f}%p, 짝지은 재표집95% 구간 [{lo*100:+.2f}, {hi*100:+.2f}]%p. 유효 그룹{value['defined_spatial_groups']}/{uncertainty['geography_clusters']}개.")
        selected=data['models'][data['selected_tag']];partial=selected['calibrated']['precision90_FA1_partial'];advisory=selected['calibrated']['alarms']
        lines.append(f"기본 보정 운영점: 정밀도{pct(advisory['precision'])}({advisory['timely_correct_first_alerts']:,}/{advisory['first_alerts']:,}개 경보), 선행 사건 재현율{pct(advisory['event_recall_3s'])}({advisory['event_hits_3s']:,}/{advisory['eligible_directed_link_events']:,}개 사건), 오경보·늦은 경보{number(advisory['false_or_late_per_drone_hour'])}/h.")
        if advisory['first_alerts']<50:lines.append('최소50개 경보 조건에 미달한다. 소수 경보의 높은 적중 비율을 일반적인 예측 신뢰성의 증거로 사용하지 않는다.')
        if partial:lines.append(f"보정P90·FA1 보조 문턱의 확인 결과: 정밀도{pct(partial['precision'])}, 선행 사건 재현율{pct(partial['event_recall_3s'])}, 오경보·늦은 경보{number(partial['false_or_late_per_drone_hour'])}/h. 보정 조건이 확인에서도 유지되는지 별도로 판단한다.")
        lines.append(f"선정 후보의 Brier(보정 전→후): 5초 {selected['raw']['next5_on_common_currently_up_links']['brier']:.5f}→{selected['calibrated']['calibrated_next5']['brier']:.5f}, 3~7초 {selected['raw']['any3to7_on_common_support']['brier']:.5f}→{selected['calibrated']['calibrated_any3to7']['brier']:.5f}. 작을수록 좋다.")
    physics_path=ARTIFACTS/'diagnostics/physics_components.json'
    if physics_path.exists():
        physics=read(physics_path);labels={'geometry_path_loss':'이동·지도에 따른 전파 손실 변화','random_shadow':'확률적 섀도잉 변화','interference_plus_noise':'간섭·잡음 전력 변화'}
        lines += ['\n## 추가 진단: 시뮬레이터 물리 항 분해',f"선택 자료90비행에서 비행당 최대12개, 총{physics['sampled_events']:,}개 단절 시작을 결정적으로 표집했다. 직전0.5초 대비 SINR 변화가 전파 손실·섀도잉·간섭 및 잡음의 변화 합으로 분해되는지 검사했다. 최대 식 오차는{physics['exact_sinr_decomposition_max_error_db']:.3g}dB다.",
            '\n| 물리 항 | 가장 큰 불리한 변화 항인 사례 | 그 항만 직전 상태로 유지하면 단절을 막는 사례 |\n|---|---:|---:|']
        for key,label in labels.items():lines.append(f"| {label} | {physics['largest_adverse_component'].get(key,0):,} | {physics['hold_previous_component_prevents_onset'].get(key,0):,} |")
        lines.append('마지막 열의 조건은 중복될 수 있다. 이는 표집한 사건의 시뮬레이터 내부 직전0.5초 분해이며, 전체 사건의 원인 분포나3초 전 예측 가능성의 상한이 아니다. 이 추가 진단으로 이미 고정한 모델·문턱을 변경하지 않았다.')
    lines += ['\n## 7. 검증과 해석 한계',f"원시 물리 아카이브{audit['physical_archives_verified']:,}개, 패킷 정답 아카이브{audit['packet_source_archives_verified']:,}개와 정산 패킷{audit['settled_packets_verified']:,}개를 검사했다. 기존 의존 파일{audit['original_dependencies_preserved']:,}개와 이전 납품 산출물을 보존했다. 훈련 역할·공간 분리·원본/변형 표본 불중복·마감시간 정답 직접 계산·모든 후보 저장 예측의 재계산 일치를 확인했다.",
        f"실제 내보낸 두 API를 한 비행{runtime['frames']:,}프레임에서 검증했다. 링크 확률 최대 차이{runtime['max_abs_probability_error']['link']:.3g}, 서비스 확률 최대 차이{runtime['max_abs_probability_error']['service']:.3g}, 경보 판단 차이{runtime['alarm_mismatches']}개다. 필수 RF 결측은None을 반환하며 자동 경보는 비활성이다."]
    for task,value in runtime['cpu_ms'].items():lines.append(f"- 데스크톱{task} 추론: 첫{value['cold']:.2f}ms, 이후 중앙값{value['warm_median']:.2f}ms. 탑재 장치 지연 인증이 아니다.")
    lines += ['공간 그룹 재표집10,000회의 통계량은 유효 그룹별 AP 평균 차이다. 전체 관측 통합 AP 차이의 신뢰구간과 다르다. 양성 사건이 없는 비행의 AP는 정의 불가로 처리하며 오경보 집계에서는 제외하지 않았다.',
        '환경은 기존2.4GHz·고도50m·이상적 TDMA·BPSK 성공확률·고정 간섭원 모델이다. 실제 Wi-Fi/IP 스택, 채널 경쟁·충돌, 실제 안테나/기체 자세, 실제 기체·통신모듈 측정 검증은 포함하지 않았다.',
        '새 서비스 정답이 임무 의미에 가까워졌다는 사실은 기존 물리 링크 예측이 향상됐다는 증거가 아니다. 과제별 대조군 결과를 따라야 한다. 조건을 바꾸어 숫자가 커진 것을 기존 성능 향상으로 표시하지 않았다.',
        'PPO는 자료 생성 정책으로 고정했다. 이번 결과를 새 재배치 정책의 연결 유지율·임무 시간 향상으로 해석하지 않는다. 미발행 미래 명령과 전조 없는 간섭은 현재 관측에 없는 정보이므로 정확한 발생 시각을 보장할 수 없다.',
        '\n## 근거와 재사용','- `artifacts/diagnostics/`: 전체 사건 분해와 타임라인.','- `artifacts/experiment_plan.json`: 진단 뒤·학습 전에 고정한 대조 계획.',
        '- `artifacts/replay_manifest.json`, `service_cache/`: 원본과 일치한 패킷 정답 복원.',
        '- `artifacts/freeze.json`, `confirmation_access.json`: 후보·문턱 고정 뒤 새 비행을 연 기록.',
        '- `artifacts/final_results.json`: 과제별 모든 모델 성적·간섭 조건·짝지은 비교.',
        '- `artifacts/audit.json`, `runtime_checks.json`: 보존·정답·예측·실행 검사.',
        '- `artifacts/research_candidates.json`: 두 연구 후보의 로더·가중치·정책 해시. `forecast_service.runtime_aligned.load_link/load_service` 사용. 확률은20초부터, 경보는 원래 평가 규칙과 같은21초부터 시작한다. 동결 원본을 유지한 별도 시작 시각 어댑터다.',
        '- 프로젝트 루트 `02_사건분석_서비스예측_현황.bat`: 현재 상태.']
    chunks=[];last_table=False
    for value in lines:
        is_table=value.lstrip().startswith('|');chunks.append(('\n' if is_table and last_table else '\n\n')+value);last_table=is_table
    (ROOT/'RESULTS_20261102.md').write_text(''.join(chunks).lstrip()+'\n',encoding='utf-8');print(ROOT/'RESULTS_20261102.md')


if __name__=='__main__':main()
