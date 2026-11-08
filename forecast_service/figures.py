"""Standalone diagnostics and same-task confirmation comparisons."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import ARTIFACTS
from .contracts import read,write,sha

NAMES={'v9_blend_75':'이전 선정 혼합','expanded_lgbm_planned_energy':'이전 트리','v10_link_uniform':'동일 조건 재학습',
       'v10_link_targeted':'사건 중심 가중치','v10_service_v9proxy':'기존 링크위험 집계',
       'v10_service_uniform':'서비스 기본 입력','v10_service_targeted':'서비스 사건 가중치','v10_service_targeted_topology':'서비스 가중치+통신 가설'}


def main():
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':9})
    result=read(ARTIFACTS/'final_results.json');folder=ARTIFACTS/'figures';folder.mkdir(exist_ok=True)
    for task,data in result['tasks'].items():
        tags=list(data['models']);x=np.arange(len(tags));fig,axes=plt.subplots(1,2,figsize=(12,5))
        for h,(key,label,color) in enumerate([('next5_on_common_currently_up_links','5초 AP','#64748b'),('any3to7_on_common_support','3~7초 AP','#0e7490')]):
            values=[data['models'][tag]['raw'][key]['ap']*100 for tag in tags]
            bars=axes[0].bar(x+(h-.5)*.36,values,width=.34,color=color,label=label)
            axes[0].bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3,fontsize=8)
        axes[0].set_xticks(x,[NAMES.get(t,t) for t in tags],rotation=20,ha='right');axes[0].set_ylim(0,100);axes[0].set_ylabel('AP (%) · 경보 적중률과 다른 순위 지표')
        axes[0].legend(frameon=False);axes[0].set_title('같은 정답·같은 새 비행에서 비교')
        ax=axes[1];colors=['#64748b','#a16207','#0e7490','#be123c']
        for i,tag in enumerate(tags):
            point=data['models'][tag]['calibrated']['common_FA_budget1']
            if point is None or point['precision'] is None:continue
            ax.scatter(point['event_recall_3s']*100,point['precision']*100,color=colors[i%len(colors)],s=55,
                marker='o' if point['false_or_late_per_drone_hour']<=1 else 'x',label=NAMES.get(tag,tag))
        ax.axhline(90,linestyle='--',color='#94a3b8',linewidth=.7);ax.axvline(80,linestyle='--',color='#94a3b8',linewidth=.7)
        ax.set(xlim=(0,100),ylim=(0,100),xlabel='3초 이상 선행 최초 경보 재현율 (%)',ylabel='최초 경보 정밀도 (%)',title='보정에서 FA≤1/h로 정한 문턱 유지')
        if ax.get_legend_handles_labels()[0]:ax.legend(frameon=False,fontsize=8,loc='lower right')
        else:ax.text(.5,.5,'보정 자료에서 FA≤1/h와\n최소50개 경보를 함께 만족한\n운영점 없음',transform=ax.transAxes,ha='center',va='center',fontsize=12,color='#64748b')
        for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        title='기존 물리 링크 단절 과제' if task=='link' else '새 패킷 서비스 장애 과제 · 지정 중계드론의 로컬 관측'
        fig.suptitle(title,fontsize=14);fig.text(.01,-.04,'기존 지도14개의 새42비행. X: 확인 오경보 예산 초과. 두 과제는 정답·분모가 달라 과제 간 AP 차감 금지. 실제 장비 성능 미검증.',fontsize=8,color='#475569')
        fig.tight_layout();fig.savefig(folder/(task+'_confirmation.png'),dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
    diagnostic=read(ARTIFACTS/'diagnostics/results.json');counts=diagnostic['event_outcomes'];keys=list(counts)
    labels={'risk_below_entry_threshold':'경보 문턱 미만','no_qualified_input':'입력 자격 없음','timely_first_alert':'제때 최초 경보',
        'late_alert_only':'늦은 경보','prior_alarm_still_active':'기존 경보 지속','high_risk_without_new_first_alert':'고위험·신규 경보 없음'}
    fig,ax=plt.subplots(figsize=(8,4));values=[counts[k] for k in keys];bars=ax.barh([labels[k] for k in keys],values,color='#0e7490')
    ax.bar_label(bars,labels=[f'{v:,} ({v/sum(values)*100:.1f}%)' for v in values],padding=4,fontsize=8)
    ax.set_xlim(0,max(values)*1.32);ax.set_xlabel('원래 정의의 단절 사건 수');ax.invert_yaxis();ax.set_title('개발 선택90비행 · 기존 고정 모델과 문턱의 진단')
    fig.tight_layout();fig.savefig(folder/'miss_diagnosis.png',dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
    write(folder/'provenance.json',dict(final_results_sha256=sha(ARTIFACTS/'final_results.json'),diagnosis_sha256=sha(ARTIFACTS/'diagnostics/results.json')))
    print(folder)


if __name__=='__main__':main()
