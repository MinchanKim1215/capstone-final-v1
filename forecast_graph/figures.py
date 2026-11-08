"""Standalone scientific comparison figures, with explicit absent operating points."""
import os
from . import ARTIFACTS
os.environ['MPLCONFIGDIR']=str(ARTIFACTS/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from .contracts import read,write,sha

LABELS={'previous_selected':'이전 선정 모델','v11_graph_local':'기존 관측\n그래프','v11_graph_shared':'공유 관측\n그래프',
    'v11_hybrid_shared':'물리·잔차\n그래프','v11_tree_existing_inputs':'기존 입력\n대조 트리','v11_tree_shared':'공유 입력\n트리','v11_hybrid_blend50':'이전·물리\n50% 혼합'}


def main():
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    results=read(ARTIFACTS/'final_results.json');folder=ARTIFACTS/'figures';folder.mkdir(exist_ok=True)
    for task,data in results['tasks'].items():
        tags=list(data['models']);x=np.arange(len(tags));fig,axes=plt.subplots(1,3,figsize=(16,5),gridspec_kw={'width_ratios':[1.25,1.25,1]})
        for ax,field,title in [(axes[0],'next5_on_common_currently_up_links','5초 AP'),(axes[1],'any3to7_on_common_support','3~7초 AP')]:
            values=[data['models'][t]['raw'][field]['ap']*100 for t in tags]
            colors=['#245bb5' if t==data['selected_tag'] else '#879bb5' for t in tags]
            ax.bar(x,values,color=colors)
            for k,v in enumerate(values):ax.text(k,v+.7,f'{v:.2f}',ha='center',fontsize=9)
            ax.set_ylim(0,max(values)*1.25+3);ax.set_title(title+' (%)');ax.set_xticks(x,[LABELS[t] for t in tags],rotation=45,ha='right')
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax=axes[2];points=[data['models'][t]['calibrated']['common_FA_budget1'] for t in tags]
        for k,p in enumerate(points):
            if p is None:ax.text(k,1,'운영점\n없음',ha='center',va='bottom',fontsize=8,color='#9a3a3a')
            else:
                v=p['event_recall_3s']*100;ax.bar(k,v,color='#245bb5' if tags[k]==data['selected_tag'] else '#879bb5')
                ax.text(k,v+1,f'{v:.2f}\nP {p["precision"]*100:.1f}%\nFA {p["false_or_late_per_drone_hour"]:.2f}/h',ha='center',fontsize=7)
        ax.axhline(80,color='#b23a33',linestyle='--',linewidth=1,label='개발 탐지율 목표80%')
        ax.set_ylim(0,105);ax.set_xlim(-.6,len(tags)-.4);ax.set_title('보정 FA1 운영점의 3초 선행 탐지율 (%)');ax.set_xticks(x,[LABELS[t] for t in tags],rotation=45,ha='right');ax.legend(fontsize=8)
        title='물리 링크 단절' if task=='link' else '패킷 서비스 장애'
        fig.suptitle(f'{title} — 새42비행 · 같은 첫 중계드론 · 선정: {LABELS[data["selected_tag"]].replace(chr(10)," ")}',fontsize=14)
        fig.text(.5,.015,'AP는 위험 순위 성능. 두 과제의 AP·사건 분모는 서로 다름. 운영점은 확인 전에 보정 자료에서 고정.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.08,1,.92]);fig.savefig(folder/(task+'_confirmation.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8));colors={'graph_local':'#909dad','graph_shared':'#c78828','hybrid_shared':'#245bb5'}
    for modelpath in (ARTIFACTS/'models').glob('v11_*_s*/model.json'):
        model=read(modelpath)
        if model['family']!='v11_graph':continue
        hist=read(modelpath.parent/'history.json')['history'];variant=model['variant'];style='-' if model['seed']==2026110204 else '--'
        for ax,field in zip(axes.ravel(),('link5','link37','service5','service37')):
            ax.plot([r['step'] for r in hist],[r['score'][field]['ap']*100 for r in hist],style,color=colors[variant],label=variant+' / '+str(model['seed']))
            ax.set_title(field+' 선택 표본 AP (%)');ax.set_xlabel('학습 갱신 횟수');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('실제 학습 이력 — 선택 표본 지표이며 최종 확인 성능과 구분');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(folder/'training_curves.png',dpi=150);plt.close(fig)
    write(folder/'provenance.json',dict(final_results_sha256=sha(ARTIFACTS/'final_results.json'),training_plan_sha256=sha(ARTIFACTS/'training_plan.json')))


if __name__=='__main__':main()

