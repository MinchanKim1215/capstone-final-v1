"""Source-bound scientific figures for causes, controlled gains and exact training."""
import os
from . import ARTIFACTS
os.environ['MPLCONFIGDIR']=str(ARTIFACTS/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from .contracts import read,write,sha
from .report import NAMES,CAUSES

def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    r=read(ARTIFACTS/'final_results.json');folder=ARTIFACTS/'figures';folder.mkdir(exist_ok=True);tags=list(r['models']);y=np.arange(len(tags));colors=['#255daf' if t==r['selected_tag'] else '#bd8639' if t=='frozen_v13' else '#8496ad' for t in tags]
    fig,axes=plt.subplots(1,3,figsize=(18,8),sharey=True)
    for ax,key,title in zip(axes,('event_recall_3s','precision','false_or_late_per_drone_hour'),('3초 선행 사건 탐지율 (%)','경보 정밀도 (%)','실제 오경보·늦은 경보 / 드론·h')):
        values=[]
        for j,tag in enumerate(tags):
            a=r['models'][tag]['calibrated']['alarms'];v=a[key];v=v*100 if v is not None and key!='false_or_late_per_drone_hour' else v;values.append(v or 0);ax.barh(j,v or 0,color=colors[j],height=.62)
            label='정의 불가' if v is None else f'{v:.2f}'
            if key=='event_recall_3s':label+=f' ({a["event_hits_3s"]}/{a["eligible_directed_link_events"]})'
            ax.text((v or 0)+(.8 if key!='false_or_late_per_drone_hour' else .06),j,label,va='center',fontsize=9)
        ax.set_xlim(0,max(105,max(values)*1.25) if key!='false_or_late_per_drone_hour' else max(6,max(values)*1.3));ax.set_title(title);ax.set_yticks(y,[NAMES[t]+(' [선정]' if t==r['selected_tag'] else '') for t in tags]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].invert_yaxis();axes[2].axvline(4,color='#a4423b',ls='--');fig.suptitle('장애 원인·전조 관측 개선 — 모델·문턱 고정 후 새42비행 · 모든 사건 포함',fontsize=14)
    fig.text(.5,.025,'금색: 기존 고정 문턱0.33. 파랑: 확인 전 선정안. 나머지: 동일FA4 보정 절차. 보정 예산은 확인 비행의 상한 보장이 아님.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(folder/'matched_confirmation.png',dpi=150);plt.close(fig)
    ca=read(ARTIFACTS/'selection_event_causes.json');keys=[k for k in CAUSES if k!='all_events'];yy=np.arange(len(keys));fig,axes=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.3,1]})
    axes[0].barh(yy-.16,[ca['all_event_evidence_counts'].get(k,0) for k in keys],height=.3,label='전체200사건',color='#8496ad');axes[0].barh(yy+.16,[ca['low_score134_evidence_counts'].get(k,0) for k in keys],height=.3,label='점수 미달134사건',color='#255daf');axes[0].set_yticks(yy,[CAUSES[k] for k in keys]);axes[0].invert_yaxis();axes[0].set_xlim(0,215);axes[0].legend(fontsize=9);axes[0].set_title('마감 전 장애 증거: 중복 가능')
    timeline=ca['observed_precursor_timeline'];x=[int(k) for k in timeline]
    for key,label in [('fraction_reported_overdue','마감 초과 대기 보고'),('fraction_reported_no_route','경로 부재 보고')]:axes[1].plot(x,[timeline[str(t)][key]*100 for t in x],marker='o',label=label)
    axes[1].axvline(-3,color='#a4423b',ls='--',alpha=.6);axes[1].set_ylim(0,105);axes[1].set_xlabel('실제 사건 기준 상대 시각 (초)');axes[1].set_ylabel('해당 상태가 실제 수신된 사건 비율 (%)');axes[1].legend(fontsize=9);axes[1].set_title('예측 시점에 실제 받은 전조')
    for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('1단계 선택 자료 진단 — 순간 연결 회복은 전체 임무의 배타적 원인 증명이 아님',fontsize=12);fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(folder/'cause_and_precursors.png',dpi=150);plt.close(fig)
    fig,grid=plt.subplots(2,2,figsize=(13,9));treecolors={'tree_base':'#8496ad','tree_network':'#8b719c','tree_physics':'#4f8e79','tree_both':'#255daf'}
    for p in (ARTIFACTS/'models').glob('v16_tree*_s*/model.json'):
        m=read(p);history=read(p.parent/'history.json')['heads']['5'];style='-' if m['seed']==2026110316 else '--';label=m['variant']+' / '+str(m['seed'])
        grid[0,0].plot([v['iteration'] for v in history],[(v['selection_event']['point']['event_recall_3s'] or 0)*100 for v in history],style,color=treecolors[m['variant']],label=label)
        grid[0,1].plot([v['iteration'] for v in history],[(v['selection_ranking']['ap'] or 0)*100 for v in history],style,color=treecolors[m['variant']],label=label)
    grid[0,0].set_title('LightGBM 선택 운영점: 선행 탐지율 (%)');grid[0,1].set_title('LightGBM 선택 자료:3~7초 AP (%)');grid[0,0].legend(fontsize=6,ncol=2)
    axes=grid[1];palette={'gru_base':'#bd8639','gru_network':'#255daf'}
    for p in (ARTIFACTS/'models').glob('v16_gru*_s*/model.json'):
        m=read(p);history=read(p.parent/'history.json')['history'];label=m['variant']+' / '+str(m['seed']);style='-' if m['seed']==2026110316 else '--'
        for ax,key,scale in zip(axes,('event_recall_3s','precision'),(100,100)):ax.plot([v['step'] for v in history],[(v['selection']['point'][key] or 0)*scale for v in history],style,color=palette[m['variant']],label=label)
    axes[0].set_title('GRU 선택 운영점: 선행 탐지율 (%)');axes[1].set_title('같은 선택 운영점: 정밀도 (%)')
    for ax in grid.flat:ax.set_xlabel('실제 학습 갱신 횟수');ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('실제12개 학습 이력 — 선택 자료 수치이며 새 확인 성능과 구분',fontsize=12);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(folder/'training_curves.png',dpi=150);plt.close(fig)
    groups=r['cause_analysis'];keys=list(CAUSES);fig,ax=plt.subplots(figsize=(12,5.7));y=np.arange(len(keys));sel=r['selected_tag']
    for offset,tag,color in [(-.17,'frozen_v13','#bd8639'),(.17,sel,'#255daf')]:
        vals=[100*groups['hits'][tag].get(k,0)/max(1,groups['counts'].get(k,0)) for k in keys];ax.barh(y+offset,vals,height=.32,label=NAMES[tag],color=color)
        for i,k in enumerate(keys):ax.text(vals[i]+1,y[i]+offset,f'{groups["hits"][tag].get(k,0)}/{groups["counts"].get(k,0)}' if groups['counts'].get(k,0) else '사건 없음',va='center',fontsize=8)
    ax.set_yticks(y,[CAUSES[k] for k in keys]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('3초 선행 탐지율 (%) — 증거별 사건 중 제때 경보');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(fontsize=9,loc='lower right');ax.set_title('새42비행: 동일한 원인 증거 그룹 · 중복 가능 · 전체 사건 유지');fig.tight_layout();fig.savefig(folder/'cause_recall.png',dpi=150);plt.close(fig)
    write(folder/'provenance.json',dict(final_results_sha256=sha(ARTIFACTS/'final_results.json'),selection_causes_sha256=sha(ARTIFACTS/'selection_event_causes.json'),training_plan_sha256=sha(ARTIFACTS/'training_plan.json'),generator_sha256=sha(__file__),four_figures=True))

if __name__=='__main__':main()
