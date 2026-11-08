"""Export scientific charts with explicit cohort and operating-point labels."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from . import ARTIFACTS
from .contracts import read,write,sha,OLD_TREE


def main():
    results=read(ARTIFACTS/'final_results.json');folder=ARTIFACTS/'figures';folder.mkdir(exist_ok=True)
    font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for tag,label,color in [('same_data_20000','동일 자료 연장','#64748b'),('new_data_20000','새 경험·관측 오차 추가','#0e7490')]:
        history=read(ARTIFACTS/'models'/tag/'history.json');steps=[r['step'] for r in history]
        for ax,key,title in zip(axes,['next5','any3to7'],['5초 AP','3~7초 AP']):
            ax.plot(steps,[r['selection'][key]['ap']*100 for r in history],color=color,label=label,linewidth=1.8)
            ax.set_title(title+' · 고정 선택 표본');ax.set_xlabel('누적 학습 업데이트');ax.set_ylabel('AP (%)');ax.grid(alpha=.15)
    for ax in axes:
        values=np.concatenate([line.get_ydata() for line in ax.lines])
        low=max(0.,np.floor((values.min()-2)/5)*5);high=min(100.,np.ceil((values.max()+2)/5)*5)
        if high-low<15:
            middle=(low+high)/2;low=max(0.,middle-7.5);high=min(100.,middle+7.5)
        ax.set_ylim(low,high)
    for ax in axes:ax.legend(frameon=False,fontsize=9)
    fig.suptitle('같은 5,000회 모델에서 출발 · 동일 추가 15,000회 예산',fontsize=14)
    fig.tight_layout();fig.savefig(folder/'learning_curves.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
    tags=[read(OLD_TREE)['tag'],'same_data_20000','new_data_20000','new_data_tree',results['selected_tag']]
    tags=list(dict.fromkeys(tags));names={read(OLD_TREE)['tag']:'기존 최선 트리','same_data_20000':'동일 자료 연장 Transformer','new_data_20000':'새 경험 Transformer','new_data_tree':'새 경험 트리','new_data_tree_no_energy':'새 경험 트리(센서 없음)',
        'frozen_v7_64':'기존 v7','expanded_residual_planned_energy':'기존 Transformer',
        'same_data_final20000':'연장 고정20k','new_data_final20000':'새 경험 고정20k'}
    x=np.arange(len(tags));fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for h,(metric,label,color) in enumerate([('next5_on_common_currently_up_links','5초 AP','#64748b'),('any3to7_on_common_support','3~7초 AP','#0e7490')]):
        values=[results['models'][tag]['raw'][metric]['ap']*100 for tag in tags]
        bars=axes[0].bar(x+(h-.5)*.36,values,width=.34,label=label,color=color)
        axes[0].bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3,fontsize=8)
    labels=[names.get(tag,'선정 혼합 후보') for tag in tags]
    axes[0].set_xticks(x,labels,rotation=25,ha='right',fontsize=9);axes[0].set_ylim(0,100);axes[0].set_ylabel('AP (%) · 경보 적중률과 다름');axes[0].legend(frameon=False)
    axes[0].set_title('고정 모델의 새 비행 확인 성적')
    ax=axes[1];ax.add_patch(Rectangle((80,90),20,10,color='#bbf7d0',alpha=.5));ax.text(90,96,'목표 영역',ha='center',fontsize=9,color='#166534')
    ax.axhline(90,color='#94a3b8',linestyle='--',linewidth=.8);ax.axvline(80,color='#94a3b8',linestyle='--',linewidth=.8)
    palette=['#64748b','#a16207','#0e7490','#7c3aed','#be123c']
    for i,tag in enumerate(tags):
        point=results['models'][tag]['calibrated']['common_FA_budget1']
        if not point or point['precision'] is None:continue
        ax.scatter(point['event_recall_3s']*100,point['precision']*100,s=65,color=palette[i%len(palette)],
            marker='o' if point['false_or_late_per_drone_hour']<=1 else 'x',label=names.get(tag,'선정 혼합 후보'))
    ax.set(xlim=(0,100),ylim=(0,100),xlabel='3초 이상 선행 사건 재현율 (%)',ylabel='최초 경보 정밀도 (%)')
    ax.set_title('보정에서 FA≤1/h로 정한 문턱 그대로 적용');ax.legend(frameon=False,fontsize=8,loc='lower right')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('새 난수·진행 방향의 확인 42비행 · 기존 지도 14개',fontsize=15)
    fig.text(.01,-.03,'X 표시는 새 확인에서 실제 오경보가 예산 1/h를 넘은 경우. 모든 후보·문턱은 확인 전에 고정. 실제 장비·새 지형·새 폐루프 성능 아님.',fontsize=9,color='#475569')
    fig.tight_layout();fig.savefig(folder/'confirmation_comparison.png',dpi=180,bbox_inches='tight',facecolor='white');fig.savefig(folder/'confirmation_comparison.svg',bbox_inches='tight',facecolor='white');plt.close(fig)
    write(folder/'provenance.json',dict(final_results_sha256=sha(ARTIFACTS/'final_results.json'),scope='Fresh-flight confirmation and fixed selection learning curves'))
    print(folder)


if __name__=='__main__':main()
