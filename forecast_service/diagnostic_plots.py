"""Reproducible selection-case timelines, including false first alerts."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import ARTIFACTS
from .contracts import read,write,V9
from .diagnose import activation
from forecast_rebuild.dataset import Episode
from forecast_rebuild.targets import onsets
from forecast_rebuild.calibration import transform,alarm_support

NAMES={'timely_first_alert':'제때 최초 경보','no_qualified_input':'경보 입력 자격 없음','high_risk_without_new_first_alert':'고위험 점수·신규 경보 없음',
       'prior_alarm_still_active':'기존 경보 지속 중','late_alert_only':'늦은 경보','risk_below_entry_threshold':'낮은 위험 점수',
       'late_under3s':'오경보 분류: 선행3초 미만','early_7to12s':'오경보 분류: 선행7~12초','no_onset_within12s':'오경보 분류: 12초 내 단절 없음'}


def main():
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':9})
    folder=ARTIFACTS/'diagnostics';result=read(folder/'results.json');frozen=read(V9/'freeze.json')
    item=next(i for i in frozen['models'] if i['pin']['path']==frozen['selected_model']);summary=read(item['selection_summary']);policy=read(item['policy'])
    records={r['key']:r for r in read(summary['dataset_path'])['rows']};extra={};threshold=result['fixed_threshold']
    for case in summary['cases']:
        r=records[case['key']];ep=Episode(r);times=np.asarray(ep.ends)*.5;end=len(ep.links)*.5-8.5;use=np.flatnonzero((times>=21)&(times<=end))
        with np.load(case['predictions_path'],allow_pickle=False) as raw:p=transform(raw['prediction'],policy['parameters'])[...,5]
        support=alarm_support(ep,policy['require_energy']);_,first=activation(p[use],support[use],threshold);starts=onsets(ep.links)
        for observer in range(first.shape[1]):
            for peer in range(first.shape[2]):
                if peer==observer+r['n_search']:continue
                events=np.flatnonzero(starts[:,observer+r['n_search'],peer])*.5
                alarms=times[use][first[:,observer,peer]]
                for alarm in alarms:
                    if ((events>=alarm+3)&(events<alarm+7)).any():continue
                    future=events[events>=alarm];lead=float(future[0]-alarm) if len(future) else np.inf
                    name='late_under3s' if lead<3 else 'early_7to12s' if lead<12 else 'no_onset_within12s'
                    if name in extra:continue
                    ix=np.flatnonzero((times>=alarm-12)&(times<=alarm+12))
                    extra[name]=dict(row=dict(case=r['key'],region=r['region'],outcome=name,onset_s=alarm,observer=observer+r['n_search'],peer=peer),
                        threshold=threshold,time=(times[ix]-alarm).tolist(),risk=[float(z) if np.isfinite(z) else None for z in p[ix,observer,peer]],
                        supported=support[ix,observer,peer].astype(int).tolist(),up=ep.links[np.asarray(ep.ends)[ix]-1,observer+r['n_search'],peer].astype(int).tolist(),
                        first_alerts=(alarms[(alarms>=alarm-12)&(alarms<=alarm+12)]-alarm).tolist(),actual_onsets=(events[(events>=alarm-12)&(events<=alarm+12)]-alarm).tolist())
        if len(extra)==3:break
    write(folder/'false_alert_examples.json',list(extra.values()));examples=result['examples']+list(extra.values())
    images=folder/'timelines';images.mkdir(exist_ok=True)
    def draw(ax,item):
        t=item['time'];ax.plot(t,item['risk'],color='#0e7490',linewidth=1.7,label='위험 확률')
        ax.axhline(threshold,color='#64748b',linestyle='--',linewidth=.8,label='경보 문턱')
        ax.step(t,item['up'],where='post',color='#94a3b8',alpha=.5,label='직전 링크 상태')
        ax.fill_between(t,0,1,where=np.asarray(item['supported'])==0,color='#f59e0b',alpha=.12,label='입력 자격 없음')
        ax.axvline(0,color='#be123c',linewidth=.9)
        for e in item.get('actual_onsets',[]):ax.axvline(e,color='#be123c',linestyle=':',linewidth=.8)
        for alarm in item['first_alerts']:ax.scatter(alarm,.96,marker='v',s=28,color='#7c3aed',zorder=5)
        ax.set(ylim=(0,1.05),xlabel='사건 또는 경보 기준 상대 시간 (초)',ylabel='확률 / 직전 링크 상태')
        ax.set_title(NAMES[item['row']['outcome']]+' · '+item['row']['region'],fontsize=9);ax.grid(alpha=.12)
    lines=['# 개발 자료 사건 타임라인','주황 음영: 경보 입력 자격 없음. 보라 삼각형: 최초 경보. 빨간0초: 단절 사건 또는 오경보 시각. 점선 빨강: 실제 단절(오경보 사례).']
    chosen={}
    for i,item in enumerate(examples):
        fig,ax=plt.subplots(figsize=(8,3.2));draw(ax,item);ax.legend(fontsize=7,loc='lower right');fig.tight_layout()
        path=images/f'case_{i:02d}.png';fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
        lines.append(f"- [{NAMES[item['row']['outcome']]} / {item['row']['case']}]({path.name})")
        chosen.setdefault(item['row']['outcome'],item)
    (images/'INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(3,3,figsize=(15,10));items=list(chosen.values())
    for ax,item in zip(axes.flat,items):draw(ax,item)
    for ax in list(axes.flat)[len(items):]:ax.set_visible(False)
    fig.suptitle('선택 자료의 미탐·오경보 사례 · 모델/문턱 변경 없음',fontsize=15);fig.tight_layout()
    fig.savefig(folder/'timeline_overview.png',dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig)
    print(folder/'timeline_overview.png')


if __name__=='__main__':main()
