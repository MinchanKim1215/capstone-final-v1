"""Compact read-only study status."""
from . import ARTIFACTS
from .contracts import read


def main():
    path=ARTIFACTS/'status.json'
    print('캡스톤 디자인 II — 예정 경로·시간 그래프·물리 보정 실험')
    if not path.exists():print('아직 상태 기록이 없습니다.');return
    value=read(path)
    for key,v in value.items():print(key,':',v)
    if (ARTIFACTS/'final_results.json').exists():
        result=read(ARTIFACTS/'final_results.json')
        for task,data in result['tasks'].items():
            selected=data['models'][data['selected_tag']]
            print(task,'selected=',data['selected_tag'],'AP5=',selected['raw']['next5_on_common_currently_up_links']['ap'],
                'AP37=',selected['raw']['any3to7_on_common_support']['ap'],'full_goal=',selected['calibrated']['current_fold_goal_met'])


if __name__=='__main__':main()
