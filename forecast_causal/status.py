"""Read-only status and next action for the explicit four-stage follow-up."""
from . import ARTIFACTS,WORKSPACE
from .contracts import read
def main():
    p=ARTIFACTS/'status.json'
    if not p.exists():print('장애 원인·전조 개선 작업 준비 중');return
    a=read(p);print('장애 원인·전조 정보 개선');print('상태:',a['state'],'단계:',a['stage']);print('갱신:',a['updated_at'])
    for k in ('complete','total','variant','step','error','next_action'):
        if k in a:print(k,':',a[k])
    p=ARTIFACTS/'milestones.json';m=read(p) if p.exists() else {}
    for i in range(1,5):print(f'{i}단계:', '완료' if str(i) in m else '진행/대기')
    print('전체 기록:',WORKSPACE/'PROJECT_FOLLOWUP.md')
    if a['state']=='completed' and (ARTIFACTS/'final_results.json').exists():
        result=read(ARTIFACTS/'final_results.json');tag=result['selected_tag'];old=result['models']['frozen_v13']['calibrated']['alarms'];new=result['models'][tag]['calibrated']['alarms']
        print('사전 선정:',tag)
        print('선행 탐지율:',f'{old["event_recall_3s"]*100:.2f}% -> {new["event_recall_3s"]*100:.2f}%')
        print('정밀도:',old['precision'],'->',new['precision'],'오경보/h:',old['false_or_late_per_drone_hour'],'->',new['false_or_late_per_drone_hour'])
        print('결과:',ARTIFACTS.parent/'RESULTS_20261103.md')
    else:print('기존 모델 보존. 새 비행 검증 전 성능 향상으로 표시하지 않음.')
if __name__=='__main__':main()
