from . import ARTIFACTS
from .contracts import read


def main():
    print('예측 학습 연장·새 경험 추가 실험\n')
    overall=read(ARTIFACTS/'status.json') if (ARTIFACTS/'status.json').exists() else {}
    for title,path in [('전체 작업','status.json'),('같은 자료 연장','models/same_data_20000/status.json'),
        ('새 경험 수집','collection_status.json'),('자료·오차 조건 준비','preparation_status.json'),
        ('새 자료 학습','models/new_data_20000/status.json'),('최종 비교','evaluation_status.json'),
        ('실행 검사·결과 보고','post_status.json')]:
        file=ARTIFACTS/path
        if not file.exists():continue
        value=read(file);parts=[value.get('state','확인 중')]
        if path=='evaluation_status.json' and (ARTIFACTS/'completed_comparison.json').exists():parts=['completed']
        if 'stage' in value:parts.append(value['stage'])
        if 'step' in value:parts.append(str(value['step'])+'/'+str(value.get('target',20000)))
        if 'selected_step' in value:parts.append('선택 체크포인트 '+str(value['selected_step']))
        if 'model' in value:parts.append(value['model'])
        if 'scope' in value:parts.append(value['scope'])
        if 'complete' in value:parts.append(str(value['complete'])+'/'+str(value['total']))
        print(title+': '+' / '.join(parts))
    if overall.get('state')=='completed':
        print('\n학습·비교·감사 완료. 결과: forecast_extension/RESULTS_20261102.md')
        print('실제 장비 검증 미실시. 자동 실사용 경보 비활성.')
    print('\n기존 v8 코드·가중치·보고서와 봉인된 최종 지도 32개를 보존합니다.')


if __name__=='__main__':main()
