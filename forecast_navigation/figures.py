"""Scientific figures from completed, frozen confirmation evidence only."""
import os
from . import ARTIFACTS
os.environ['MPLCONFIGDIR']=str(ARTIFACTS/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
from .contracts import read,write,sha

LABELS={'v11_selected':'기존 선정','v11_ranking_champion':'기존 순위 우수',
    'v12_original':'원래 경로\n추가 학습','v12_mission':'보완 경로\n추가 학습',
    'v12_original_blend50':'기존·원래 경로\n50% 혼합','v12_mission_blend50':'기존·보완 경로\n50% 혼합'}


def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    result=read(ARTIFACTS/'final_results.json');folder=ARTIFACTS/'figures';folder.mkdir(exist_ok=True)
    for task,data in result['tasks'].items():
        tags=list(data['models']);x=np.arange(len(tags));fig,axes=plt.subplots(1,3,figsize=(16,5.5))
        colors=['#255bad' if t==data['selected_tag'] else '#218676' if t==data['ranking_champion'] else '#8ba0bc' for t in tags]
        for ax,field,title in [(axes[0],'next5_on_common_currently_up_links','5초 AP'),(axes[1],'any3to7_on_common_support','3~7초 AP')]:
            values=[data['models'][t]['raw'][field]['ap']*100 for t in tags];ax.bar(x,values,color=colors)
            for j,v in enumerate(values):ax.text(j,v+.6,f'{v:.2f}',ha='center',fontsize=9)
            ax.set_ylim(0,max(values)*1.25+3);ax.set_title(title+' (%)');ax.set_xticks(x,[LABELS[t] for t in tags],rotation=35,ha='right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        legend=[Patch(color='#255bad',label='경보 기준 사전 선정')]
        if data['ranking_champion']!=data['selected_tag']:legend.append(Patch(color='#218676',label='선택 자료 순위 우수'))
        axes[0].legend(handles=legend,fontsize=8)
        ax=axes[2]
        for j,tag in enumerate(tags):
            point=data['models'][tag]['calibrated']['common_FA_budget1']
            if point is None:ax.text(j,1,'운영점\n없음',ha='center',va='bottom',color='#9a3a3a',fontsize=8)
            else:
                v=point['event_recall_3s']*100;ax.bar(j,v,color=colors[j]);ax.text(j,v+1,f'{v:.2f}\nP {point["precision"]*100:.1f}%\nFA {point["false_or_late_per_drone_hour"]:.2f}/h',ha='center',fontsize=8)
        ax.axhline(80,color='#b33d35',ls='--',lw=1,label='탐지율 목표80%');ax.set_ylim(0,105);ax.set_xlim(-.6,len(tags)-.4)
        ax.set_title('고정 FA1 문턱의 3초 선행 탐지율 (%)');ax.set_xticks(x,[LABELS[t] for t in tags],rotation=35,ha='right');ax.legend(fontsize=8)
        title='물리 링크 단절' if task=='link' else '패킷 서비스 장애'
        fig.suptitle(f'{title} — 동일한 새42비행 · 첫 중계드론 · 경보 기준 선정: {LABELS[data["selected_tag"]].replace(chr(10)," ")}',fontsize=14)
        fig.text(.5,.015,'AP는 예측 순위 지표. FA1 문턱은 확인 전 보정 자료에서 고정. 두 과제의 정답·분모는 다름.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.08,1,.92]);fig.savefig(folder/(task+'_confirmation.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for path in (ARTIFACTS/'models').glob('v12_*_s*/history.json'):
        data=read(path);meta=read(path.parent/'model.json');h=data['history'];profile=data['profile'];seed=meta['seed']
        for ax,field in zip(axes.ravel(),('link5','link37','service5','service37')):
            ax.plot([v['step'] for v in h],[v['score'][field]['ap']*100 for v in h],'-' if seed==2026110205 else '--',
                color='#245bad' if profile=='mission' else '#b47c29',label=('보완' if profile=='mission' else '원래')+' 경로 / '+str(seed))
            ax.set_title(field+' 선택 표본 AP (%)');ax.set_xlabel('추가 학습 갱신 횟수');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('같은 초기 가중치·학습량의 실제 이력 — 선택 자료 지표');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(folder/'training_curves.png',dpi=150);plt.close(fig)
    write(folder/'provenance.json',dict(final_results_sha256=sha(ARTIFACTS/'final_results.json'),training_plan_sha256=sha(ARTIFACTS/'training_plan.json')))


if __name__=='__main__':main()
