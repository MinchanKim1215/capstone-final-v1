from . import ARTIFACTS
from .contracts import read


def main():
    print('예측 개선 실험 현황\n')
    for title,name in [('전체 작업','research_status.json'),('간섭 조건 균형 자료','balanced_collection_status.json'),
                       ('기본 후보 학습','campaign/status.json'),('연속 통신 정답 후보','models/expanded_dense_residual_energy/status.json'),
                       ('공개 경로 특징 준비','planning_status.json'),('동일 조건 비교','benchmarks/balanced_status.json'),
                       ('선택·보정·독립 확인','completion_status.json'),('측정 오차·보존 검사','post_checks_status.json')]:
        path=ARTIFACTS/name
        if not path.exists():continue
        a=read(path);state=a.get('state','확인 중')
        labels={'completed':'완료','training_completed':'학습 완료','running':'진행 중','failed':'오류'}
        parts=[labels.get(state,str(state))]
        if name=='benchmarks/balanced_status.json' and a.get('state')=='waiting_for_balanced_collection' and (ARTIFACTS/'balanced_collection_status.json').exists() and read(ARTIFACTS/'balanced_collection_status.json')['state']=='completed':
            parts=['비교 입력 준비 중']
        if 'complete' in a:parts.append(f"{a['complete']}/{a['total']}")
        if 'stage' in a:parts.append(a['stage'])
        if name=='campaign/status.json' and (ARTIFACTS/'completed_comparison.json').exists():parts=['학습·비교 완료']
        print(title+': '+' / '.join(parts))
    ready=ARTIFACTS/'final_results.json'
    if ready.exists():
        a=read(ready);print('\n최종 선택 후보:',a.get('selected_model_tag','-'))
        print('실사용 검증:', '완료' if a.get('production_ready') else '미완료 - 연구용 결과')
        print('선행 경보 목표:', '충족' if a['selected']['calibrated']['trusted_simulation_alarm_enabled'] else '미충족')
        print('상세 결과: forecast_rebuild/RESULTS_20261102.md')
    print('\n기존 v5·v6·v7 코드와 모델은 보존됩니다.')


if __name__=='__main__':main()
