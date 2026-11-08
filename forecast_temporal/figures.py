"""Scientific plots from archived matched outcomes; no hand-entered gains."""
import os
from . import ARTIFACTS
os.environ['MPLCONFIGDIR']=str(ARTIFACTS/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from .contracts import read,write,sha
from .report import NAMES

def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    d=read(ARTIFACTS/'final_results.json');folder=ARTIFACTS/'figures';folder.mkdir(exist_ok=True);tags=list(d['models']);y=np.arange(len(tags))
    colors=['#255daf' if t==d['selected_tag'] else '#c18535' if t=='retuned_v13' else '#8496ad' for t in tags]
    fig,axes=plt.subplots(1,3,figsize=(17,6.6),sharey=True)
    for ax,key,title in [(axes[0],'event_recall_3s','3초 선행 사건 탐지율 (%)'),(axes[1],'precision','경보 정밀도 (%)'),(axes[2],'false_or_late_per_drone_hour','실제 오경보·늦은 경보 / 드론·h')]:
        values=[]
        for j,tag in enumerate(tags):
            a=d['models'][tag]['calibrated']['alarms'];v=a[key];v=v*100 if v is not None and key!='false_or_late_per_drone_hour' else v;values.append(v or 0);ax.barh(j,v or 0,color=colors[j],height=.62)
            label='정의 불가' if v is None else f'{v:.2f}'
            if key=='event_recall_3s':label+=f'  ({a["event_hits_3s"]}/{a["eligible_directed_link_events"]})'
            ax.text((v or 0)+(.8 if key!='false_or_late_per_drone_hour' else .06),j,label,va='center',fontsize=9)
        ax.set_xlim(0,110 if key!='false_or_late_per_drone_hour' else max(6,max(values)*1.3));ax.set_title(title);ax.set_yticks(y,[NAMES[t]+(' [선정]' if t==d['selected_tag'] else '') for t in tags]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].invert_yaxis();axes[2].axvline(4,color='#a4423b',ls='--',lw=1);axes[2].text(.98,.015,'보정 예산4/h\n확인 상한 아님',transform=axes[2].transAxes,ha='right',va='bottom',color='#a4423b',fontsize=8)
    fig.suptitle('20초 드론·링크 그래프 — 같은 새42비행 · 첫 중계드론 · 모델·문턱 사전 고정',fontsize=14)
    fig.text(.5,.02,'기존 고정 설정은 문턱0.33 보존. 그 외 후보는 같은FA4보정 절차. 파랑: 확인 전 선정안. 탐지율과 경보 비용을 함께 비교.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(folder/'matched_confirmation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));palette={'short_bce':'#b78033','short_event':'#4f8e79','long_event':'#255daf'}
    for p in (ARTIFACTS/'models').glob('v14_*_s*/model.json'):
        m=read(p);history=read(p.parent/'history.json')['history'];style='-' if m['seed']==2026110314 else '--';label=m['variant']+' / '+str(m['seed'])
        for ax,key,scale in [(axes[0],'event_recall_3s',100),(axes[1],'false_or_late_per_drone_hour',1)]:ax.plot([a['step'] for a in history],[(a['selection']['point'][key] or 0)*scale for a in history],style,color=palette[m['variant']],label=label)
    axes[0].set_title('선택 자료: 예산 내 선행 탐지율 (%)');axes[1].set_title('해당 선택 운영점의 오경보·늦은 경보/h')
    for ax in axes:ax.set_xlabel('실제 학습 갱신 횟수');ax.grid(alpha=.15)
    axes[0].legend(fontsize=7);fig.suptitle('실제6개 학습 이력 — 선택 자료 수치이며 새 확인 성능과 구분');fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(folder/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8));show=list(dict.fromkeys(['frozen_v13','retuned_v13',d['selected_tag']]))
    for ax,(kind,label) in zip(axes,[('clear','추가 방해 없음'),('intermittent','간헐 방해'),('continuous','지속 방해')]):
        for i,tag in enumerate(show):
            a=d['condition_alarms'][kind][tag];v=a['event_recall_3s']*100;ax.bar(i,v,color='#255daf' if tag==d['selected_tag'] else '#8496ad');ax.text(i,v+1,f'{v:.1f}%\nFA {a["false_or_late_per_drone_hour"]:.2f}/h',ha='center',fontsize=9)
        ax.set_title(label);ax.set_ylim(0,105);ax.set_xticks(range(len(show)),[NAMES[t].replace(' ','\n',1) for t in show],rotation=12);ax.set_ylabel('선행 탐지율 (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('간섭 조건별 같은 비행 비교 — 소수 사건 표본의 불확실성 유지');fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(folder/'condition_recall.png',dpi=160);plt.close(fig)
    write(folder/'provenance.json',dict(final_results_sha256=sha(ARTIFACTS/'final_results.json'),training_plan_sha256=sha(ARTIFACTS/'training_plan.json')))

if __name__=='__main__':main()
