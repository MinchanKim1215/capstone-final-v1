"""Read-only current phase and actual completed event-recall outcomes."""
from . import ARTIFACTS
from .contracts import read


def main():
    def pct(value):return '정의 불가' if value is None else f'{value*100:.2f}%'
    print('캡스톤 디자인 II — 선행 탐지율 개선 실험')
    for name in ('status.json','evaluation_status.json','power_guard.json'):
        p=ARTIFACTS/name
        if p.exists():print(name,read(p))
    p=ARTIFACTS/'final_results.json'
    if p.exists():
        d=read(p);print('확인 전 선정 모델:',d['selected_tag'])
        for tag,m in d['models'].items():
            a=m['calibrated']['alarms'];print(tag,'선행 탐지율=',pct(a['event_recall_3s']),'정밀도=',pct(a['precision']),
                '오경보/h=',round(a['false_or_late_per_drone_hour'],2),'탐지 사건=',a['event_hits_3s'],'/',a['eligible_directed_link_events'])
        print('기존 정밀도90%·탐지율80%·FA1 목표 충족:',d['models'][d['selected_tag']]['calibrated']['current_fold_goal_met'])
        other=ARTIFACTS/'secondary_operating_points.json'
        if other.exists():
            values=read(other)['models'];print('다음은 보조 운영점이며 기존FA1 목표와 구분합니다.')
            for tag in dict.fromkeys(['frozen_parent',d['selected_tag']]):
                for budget in ('3','5'):
                    point=values[tag][budget]
                    if point:
                        a=point['confirmation'];print(tag,'보정FA예산=',budget,'탐지율=',pct(a['event_recall_3s']),'정밀도=',pct(a['precision']),'실제오경보/h=',round(a['false_or_late_per_drone_hour'],2))


if __name__=='__main__':main()
