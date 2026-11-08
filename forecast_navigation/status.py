"""Read-only progress and completed matched results."""
from . import ARTIFACTS
from .contracts import read


def main():
    print('캡스톤 디자인 II — 경로 보완·동일 조건 학습 비교')
    path=ARTIFACTS/'status.json'
    if not path.exists():print('상태 기록 없음');return
    for key,value in read(path).items():print(key,':',value)
    for name in ('evaluation_status.json','power_guard.json'):
        path=ARTIFACTS/name
        if path.exists():print(name,read(path))
    path=ARTIFACTS/'final_results.json'
    if path.exists():
        for task,d in read(path)['tasks'].items():
            m=d['models'][d['selected_tag']]
            print(task,'경보 기준 후보=',d['selected_tag'],'AP5=',m['raw']['next5_on_common_currently_up_links']['ap'],
                'AP3~7=',m['raw']['any3to7_on_common_support']['ap'],'경보목표충족=',m['calibrated']['current_fold_goal_met'])
            ranking=d['models'][d['ranking_champion']];c=d['ranking_vs_reference']
            print(task,'사전 순위 우수 후보=',d['ranking_champion'],
                'AP5=',ranking['raw']['next5_on_common_currently_up_links']['ap'],
                'AP3~7=',ranking['raw']['any3to7_on_common_support']['ap'],
                '기존 강한 기준 대비 차이(%p)=',round(c['pooled_ap5_difference']*100,2),round(c['pooled_ap37_difference']*100,2))


if __name__=='__main__':main()
