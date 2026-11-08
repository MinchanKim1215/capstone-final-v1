"""Primary event-recall evidence, explicit precision/FA costs and training history."""
import os
from . import ARTIFACTS
os.environ['MPLCONFIGDIR']=str(ARTIFACTS/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from .contracts import read,write,sha
LABELS={'frozen_parent':'기존 모델·문턱','threshold_only_parent':'기존 모델\n문턱만 조정',
    'v13_bce':'일반 학습\n대조군','v13_event':'사건 중심\n학습','v13_memory_event':'20초 이력\n사건 학습',
    'v13_event_blend50':'기존·사건\n50% 혼합','v13_memory_event_blend50':'기존·20초 이력\n50% 혼합'}


def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    d=read(ARTIFACTS/'final_results.json');folder=ARTIFACTS/'figures';folder.mkdir(exist_ok=True);tags=list(d['models']);x=np.arange(len(tags))
    fig,axes=plt.subplots(1,3,figsize=(16,6));colors=['#245bad' if t==d['selected_tag'] else '#8fa1b9' for t in tags]
    for ax,key,title,target in [(axes[0],'event_recall_3s','3초 선행 사건 탐지율 (%)',80),(axes[1],'precision','경보 정밀도 (%)',90),(axes[2],'false_or_late_per_drone_hour','오경보·늦은 경보 / 드론·시간',1)]:
        values=[]
        for j,tag in enumerate(tags):
            a=d['models'][tag]['calibrated']['alarms'];v=a[key];v=v*100 if v is not None and key!='false_or_late_per_drone_hour' else v;values.append(v or 0)
            if v is None:ax.text(j,1,'정의 불가',ha='center',fontsize=8)
            else:
                ax.bar(j,v,color=colors[j]);text=f'{v:.2f}'
                if key=='event_recall_3s':text+=f'\n{a["event_hits_3s"]}/{a["eligible_directed_link_events"]}'
                ax.text(j,v+.9 if key!='false_or_late_per_drone_hour' else v+.03,text,ha='center',fontsize=8)
        ax.axhline(target,color='#b23c34',ls='--',lw=1,label='개발 목표'+str(target));ax.legend(fontsize=8)
        ax.set_ylim(0,105 if key!='false_or_late_per_drone_hour' else max(max(values)*1.35,target*1.7));ax.set_xlim(-.6,len(tags)-.4)
        ax.set_title(title);ax.set_xticks(x,[LABELS[t] for t in tags],rotation=35,ha='right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('선행 탐지율 직접 개선 — 동일한 새42비행 · 첫 중계드론 · 확인 전 모델·문턱 고정',fontsize=14)
    fig.text(.5,.015,'기존 모델·문턱은 현재 기준. 문턱만 조정한 대조군으로 학습 효과와 정책 효과를 구분. 정밀도·오경보도 함께 판단.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.93]);fig.savefig(folder/'event_recall_confirmation.png',dpi=160);plt.close(fig)
    secondary=read(ARTIFACTS/'secondary_operating_points.json');show=list(dict.fromkeys(['frozen_parent',d['selected_tag']]))
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));colors2=['#8b9bb4','#245bad']
    for j,tag in enumerate(show):
        points=secondary['models'][tag]
        for ax,key,title in [(axes[0],'event_recall_3s','3초 선행 사건 탐지율 (%)'),(axes[1],'precision','경보 정밀도 (%)')]:
            available=[(int(b),p['confirmation']) for b,p in points.items() if p is not None]
            xx=[b for b,p in available];yy=[p[key]*100 if p[key] is not None else np.nan for b,p in available]
            legend_label='기존 모델(보정 운영점)' if tag=='frozen_parent' else '20초 이력 모델(보정 운영점)'
            ax.plot(xx,yy,'o-',color=colors2[j],label=legend_label)
            for b,p in available:
                other=secondary['models'][show[1-j]].get(str(b)) if len(show)==2 else None
                above=other is None or (p[key] is not None and p[key]>=(other['confirmation'][key] or 0))
                if p[key] is not None:ax.annotate(f'{p[key]*100:.1f}%\n실제FA {p["false_or_late_per_drone_hour"]:.2f}/h',(b,p[key]*100),
                    xytext=(-6 if b==5 else 6,12 if above else -30),ha='right' if b==5 else 'left',textcoords='offset points',fontsize=8,
                    color=colors2[j],arrowprops=dict(arrowstyle='-',color=colors2[j],lw=.6))
            ax.set_title(title);ax.set_xlabel('보정 자료에서 고정한 FA 예산 (/h)');ax.set_xticks([1,3,5]);ax.set_xlim(.5,5.6);ax.set_ylim(0,105);ax.grid(alpha=.15);ax.legend(fontsize=8)
    for ax in axes:ax.text(1,3,'보정50경보\n운영점 없음',ha='center',va='bottom',fontsize=8,color='#9b4b45')
    fig.suptitle('보조 운영점 비교 — FA3/5는 기존 FA1 목표를 충족한 결과로 해석하지 않음');fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(folder/'recall_precision_tradeoff.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));palette={'bce':'#a67a2f','event':'#568778','memory_event':'#245bad'}
    for p in (ARTIFACTS/'models').glob('v13_*_s*/model.json'):
        m=read(p);h=read(p.parent/'history.json')['history'];style='-' if m['seed']==2026110301 else '--';label=m['variant']+' / '+str(m['seed'])
        axes[0].plot([v['step'] for v in h],[v['selection']['point']['event_recall_3s']*100 for v in h],style,color=palette[m['variant']],label=label)
        axes[1].plot([v['step'] for v in h],[v['selection']['point']['false_or_late_per_drone_hour'] for v in h],style,color=palette[m['variant']],label=label)
    for ax,title in [(axes[0],'선택 자료의 예산 내 선행 탐지율 (%)'),(axes[1],'해당 운영점의 오경보·늦은 경보 (/h)')]:ax.set_title(title);ax.set_xlabel('학습 갱신 횟수');ax.grid(alpha=.15)
    axes[0].legend(fontsize=7);fig.suptitle('실제 학습 이력 — 최종 확인 성능과 구분');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(folder/'training_curves.png',dpi=160);plt.close(fig)
    write(folder/'provenance.json',dict(final_results_sha256=sha(ARTIFACTS/'final_results.json'),secondary_results_sha256=sha(ARTIFACTS/'secondary_operating_points.json'),training_plan_sha256=sha(ARTIFACTS/'training_plan.json')))


if __name__=='__main__':main()
