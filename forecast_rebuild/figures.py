"""Standalone scientific figures from the completed frozen confirmation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Rectangle
from . import ARTIFACTS
from .contracts import read,write,sha


def main():
    data=read(ARTIFACTS/'final_results.json');folder=ARTIFACTS/'figures';folder.mkdir(exist_ok=True)
    font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    old=data['baseline'];new=data['selected'];colors=['#64748b','#0e7490'];x=np.arange(2)
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),gridspec_kw={'wspace':.32})
    for index,(label,model) in enumerate([('기존 v7',old),('새 예측 후보',new)]):
        values=[model['raw']['next5_on_common_currently_up_links']['ap']*100,model['raw']['any3to7_on_common_support']['ap']*100]
        bars=axes[0].bar(x+(index-.5)*.33,values,width=.31,color=colors[index],label=label)
        axes[0].bar_label(bars,labels=[f'{value:.2f}%' for value in values],padding=4,fontsize=10)
    axes[0].set_xticks(x,['5초 단절 AP','3~7초 단절 AP']);axes[0].set_ylim(0,100)
    axes[0].set_ylabel('AP (%) · 경보 적중률과 다름');axes[0].legend(frameon=False)
    axes[0].set_title('순위 성능: 같은 관측·같은 정답',pad=14)
    ax=axes[1];ax.add_patch(Rectangle((80,90),20,10,color='#bbf7d0',alpha=.5))
    ax.axvline(80,color='#94a3b8',linestyle='--',linewidth=.8);ax.axhline(90,color='#94a3b8',linestyle='--',linewidth=.8)
    ax.text(90,96,'목표 영역',fontsize=9,color='#166534',ha='center')
    for index,(label,model) in enumerate([('기존',old),('새 후보',new)]):
        a=model['raw']['strict_alarms_at_half'];b=model['calibrated']['alarms']
        px=[a['event_recall_3s']*100,b['event_recall_3s']*100];py=[a['precision']*100,b['precision']*100]
        ax.plot(px,py,color=colors[index],alpha=.5,linewidth=1)
        ax.scatter(px[0],py[0],c=colors[index],marker='o',s=65,label=label+' · 원래 0.5')
        ax.scatter(px[1],py[1],c=colors[index],marker='D',s=58,label=label+' · 보정 고정 문턱')
    ax.set(xlim=(0,100),ylim=(0,100),xlabel='3초 이상 선행 사건 재현율 (%)',ylabel='최초 경보 정밀도 (%)')
    ax.legend(frameon=False,fontsize=9,loc='lower right');ax.set_title('경보 신뢰성: 정밀도·탐지율 동시 확인',pad=14)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('예측 개선 · 독립 개발 확인 14지도 / 42비행',fontsize=16,y=1.02)
    fig.text(.01,-.015,'동결된 기존 v7 비행에서 예측만 비교. 실제 장비 검증·새 폐루프 연결률 성적 아님. 목표에는 오경보·늦은 경보 ≤1회/중계드론·시간도 필요.',fontsize=9,color='#475569')
    fig.savefig(folder/'confirmation_summary.png',dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(folder/'confirmation_summary.svg',bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True,sharey=True)
    for ax,key,title in zip(axes,['reliability_next5','reliability_any3to7'],['5초 확률 보정','3~7초 확률 보정']):
        ax.plot([0,1],[0,1],color='#94a3b8',linestyle='--',label='확률=관측 빈도')
        for index,(label,model) in enumerate([('기존 v7',old),('새 후보',new)]):
            bins=model['calibrated'][key]['bins'];xx=[b['mean_prediction'] for b in bins];yy=[b['event_fraction'] for b in bins]
            ax.plot(xx,yy,marker='o',markersize=4,color=colors[index],label=label)
        ax.set_title(title);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.15)
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel('보정 후 예측 확률');ax.legend(frameon=False,fontsize=9)
    axes[0].set_ylabel('해당 확률 구간의 실제 사건 빈도')
    fig.suptitle('확률 신뢰도 · 고정 보정 적용 후의 독립 확인',fontsize=15)
    fig.tight_layout();fig.savefig(folder/'probability_reliability.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
    write(folder/'provenance.json',dict(results_sha256=sha(ARTIFACTS/'final_results.json'),confirmation_only=True,
        plots=['confirmation_summary.png','confirmation_summary.svg','probability_reliability.png']))
    print(folder)


if __name__=='__main__':main()
