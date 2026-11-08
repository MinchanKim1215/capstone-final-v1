"""Read-only Korean training and evaluation status."""
from . import ARTIFACTS
from .contracts import read

def main():
    p=ARTIFACTS/'status.json'
    if not p.exists():print('개별 관측 실험 준비 중');return
    s=read(p);print('개별 관측 이력 보존 예측 학습');print('상태:',s['state'],' 단계:',s.get('stage'));print('갱신:',s.get('updated_at'))
    for k in ('complete','total','variant','seed','step','best_step','error'):
        if k in s:print(k,':',s[k])
    path=ARTIFACTS/'final_results.json'
    if path.exists():
        r=read(path)
        for tag in ('frozen_v13','retuned_v13',r['selected_tag']):
            a=r['models'][tag]['calibrated']['alarms'];print(tag,f'선행 탐지율{a["event_recall_3s"]*100:.2f}% 정밀도{a["precision"]*100:.2f}% 오경보{a["false_or_late_per_drone_hour"]:.2f}/드론h')
    print('연구용. 기존PPO·실제 장비 자동교체 없음.')

if __name__=='__main__':main()
