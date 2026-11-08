"""First-alert misses and false alerts on selection data, without refitting."""
from collections import Counter,defaultdict
import csv
import json
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import prepare,read,write,sha,now,status,V9
from forecast_rebuild.dataset import Episode
from forecast_rebuild.targets import onsets
from forecast_rebuild.calibration import transform,alarm_support
from forecast_extension.metrics import alert_counts,combine_alerts


def activation(score,support,threshold):
    p=np.asarray(score,np.float64);known=support&np.isfinite(p);ix=np.arange(len(p))[:,None,None]
    enter=known&(p>=threshold);leave=~known|(p<threshold*.5)
    active=np.maximum.accumulate(np.where(enter,ix,-1),axis=0)>np.maximum.accumulate(np.where(leave,ix,-1),axis=0)
    first=active.copy();first[1:] &= ~active[:-1]
    return active,first


def connected(links,ns):
    reach=np.asarray(links,bool).copy();ix=np.arange(reach.shape[1]);reach[:,ix,ix]=True
    for k in range(reach.shape[1]):reach |= reach[:,:,k,None]&reach[:,None,k,:]
    return reach[:,:ns,:ns].all((1,2))


def main():
    prepare();folder=ARTIFACTS/'diagnostics';folder.mkdir(exist_ok=True);status('stage1_selection_event_diagnostics')
    frozen=read(V9/'freeze.json');entry=next(i for i in frozen['models'] if i['pin']['path']==frozen['selected_model'])
    summary=read(entry['selection_summary']);policy=read(entry['policy']);threshold=policy['common_FA_budget1']['threshold']
    records={r['key']:r for r in read(summary['dataset_path'])['rows']};event_counts=Counter();false_counts=Counter();groups=defaultdict(Counter)
    examples=[];example_counts=Counter();case_counts=[];all_events=0
    fields=['case','region','condition','observer','peer','onset_s','duration_s','right_censored','outcome','strict_hit','high_risk_coverage','active_coverage','recent_traffic','command_change_near','jammer_switch_near','global_disconnect_coincident']
    with (folder/'events.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case_index,case in enumerate(summary['cases']):
            r=records[case['key']]
            if r['role']!='selection':raise ValueError('Diagnostic optimization must remain in selection')
            ep=Episode(r);times=np.asarray(ep.ends)*.5;end=len(ep.links)*.5-8.5;use=np.flatnonzero((times>=21)&(times<=end));ts=times[use]
            with np.load(case['predictions_path'],allow_pickle=False) as a:p=transform(a['prediction'],policy['parameters'])[...,5]
            support=alarm_support(ep,policy['require_energy']);active,first=activation(p[use],support[use],threshold)
            expected=alert_counts(ep.links,ep.ends,p,support,r['n_search'],threshold);case_counts.append(expected)
            starts=onsets(ep.links);ns=r['n_search'];net=connected(ep.links,ns)
            with np.load(r['array_path'],allow_pickle=False) as a:commands=a['commands']
            # Hidden schedules below are read for diagnostic annotations only.
            from adaptive_joint_expanded.world import World
            world=World(**r['spec'],deployment=r['deployment'],observer=None,max_steps=1)
            switches=[x for x in (world.jam_start,world.jam_end) if np.isfinite(x) and x>0];del world
            counted_hit=counted_events=counted_first=counted_correct=0
            for local in range(ep.values.shape[1]):
                for peer in range(ep.values.shape[2]):
                    observer=local+ns
                    if observer==peer:continue
                    events=np.flatnonzero(starts[:,observer,peer]);event_time=events*.5;alerts=ts[first[:,local,peer]];counted_first+=len(alerts)
                    for alarm in alerts:
                        future=event_time[event_time>=alarm];lead=float(future[0]-alarm) if len(future) else np.inf
                        correct=bool(((event_time>=alarm+3)&(event_time<alarm+7)).any());counted_correct+=int(correct)
                        if not correct:false_counts['late_under3s' if lead<3 else 'early_7to12s' if lead<12 else 'no_onset_within12s']+=1
                    for q in events[(event_time>=24)&(event_time<=end)]:
                        event=q*.5;window=(ts>event-7)&(ts<=event-3);hit=bool(first[window,local,peer].any())
                        eligible=bool(support[use[window],local,peer].any());high=bool((support[use[window],local,peer]&(p[use[window],local,peer]>=threshold)).any())
                        ongoing=bool(active[window,local,peer].any());late=bool(((alerts>event-3)&(alerts<=event+1)).any())
                        reason='timely_first_alert' if hit else 'no_qualified_input' if not eligible else 'high_risk_without_new_first_alert' if high else 'prior_alarm_still_active' if ongoing else 'late_alert_only' if late else 'risk_below_entry_threshold'
                        recovery=np.flatnonzero(ep.links[q:,observer,peer]);duration=(int(recovery[0]) if len(recovery) else len(ep.links)-q)*.5
                        v=np.asarray(ep.values[max(0,q-6):q,local,peer]);m=np.asarray(ep.masks[max(0,q-6):q,local,peer]);traffic=bool((np.where(m[:,5],v[:,5],0)+np.where(m[:,14],v[:,14],0)).sum()>0)
                        k=q//6;before=max(1,int((event-7)//3));changes=np.linalg.norm(np.diff(commands[before-1:min(len(commands),k+1)],axis=0),axis=-1)
                        changed=bool((changes[:,[observer,peer]]>1.).any()) if len(changes) else False
                        row=dict(case=r['key'],region=r['region'],condition=r['spec']['kind'],observer=observer,peer=peer,onset_s=event,duration_s=duration,
                            right_censored=not len(recovery),outcome=reason,strict_hit=hit,high_risk_coverage=high,active_coverage=ongoing,recent_traffic=traffic,
                            command_change_near=changed,jammer_switch_near=any(event-7<=s<=event for s in switches),global_disconnect_coincident=bool(net[q-1] and not net[q]))
                        writer.writerow(row);all_events+=1;counted_events+=1;counted_hit+=int(hit);event_counts[reason]+=1
                        labels=['all','duration_1to2s' if duration<2 else 'duration_2to5s' if duration<5 else 'duration_ge5s','recent_traffic' if traffic else 'no_recent_traffic',r['spec']['kind']]
                        for label in labels:
                            g=groups[label];g['events']+=1;g['first_alert_hits']+=int(hit);g['high_risk_coverage']+=int(high);g['active_coverage']+=int(ongoing);g[reason]+=1
                        if example_counts[reason]<3 and len(examples)<20:
                            keep=(times>=event-12)&(times<=event+8);ix=np.flatnonzero(keep)
                            sample=dict(row=row,threshold=threshold,time=(times[ix]-event).tolist(),risk=[float(z) if np.isfinite(z) else None for z in p[ix,local,peer]],
                                supported=support[ix,local,peer].astype(int).tolist(),up=ep.links[np.minimum(np.asarray(ep.ends)[ix]-1,len(ep.links)-1),observer,peer].astype(int).tolist(),
                                first_alerts=(alerts[(alerts>=event-12)&(alerts<=event+8)]-event).tolist())
                            examples.append(sample);example_counts[reason]+=1
            assert (counted_first,counted_correct,counted_events,counted_hit)==tuple(expected[k] for k in ('first_alerts','timely_correct_first_alerts','eligible_directed_link_events','event_hits_3s'))
            write(folder/'status.json',dict(state='running',complete=case_index+1,total=len(summary['cases']),events=all_events,updated_at=now()))
    result=dict(created_at=now(),scope='V9 selection90 only',model_tag=summary['model_tag'],fixed_threshold=threshold,policy_sha256=sha(entry['policy']),
        event_outcomes=dict(event_counts),false_alert_outcomes=dict(false_counts),groups={name:dict(values,strict_recall=values['first_alert_hits']/values['events'],high_risk_coverage_rate=values['high_risk_coverage']/values['events']) for name,values in groups.items()},
        preserved_original_counts=combine_alerts(case_counts),examples=examples,definition='Original >=1s theoretical physical-link outages; [3,7) first-alert rule unchanged.',
        caution='Traffic is locally observed activity; near-command/jammer/global-disconnect flags are associations, not causal attribution. No confirmation used for optimization.')
    write(folder/'results.json',result);write(folder/'status.json',dict(state='completed',complete=len(summary['cases']),total=len(summary['cases']),updated_at=now()))
    lines=['# 캡스톤 디자인 II — 개발 자료 미탐·오경보 진단',f'선택90비행. 고정 보정 문턱{threshold:.4f}. 기존 경보 정의·사건 분모 유지. 확인 자료 미사용.','\n| 사건 분류 | 사건 수 |\n|---|---:|']
    lines += [f'| {key} | {value:,} |' for key,value in event_counts.items()]
    lines += ['\n| 조건 | 사건 수 | 최초 경보 재현율 | 높은 위험 점수 사전 포착률 |\n|---|---:|---:|---:|']
    lines += [f"| {key} | {g['events']:,} | {g['strict_recall']*100:.2f}% | {g['high_risk_coverage_rate']*100:.2f}% |" for key,g in result['groups'].items()]
    lines += ['\n높은 위험 점수 사전 포착률은 진단값이며 최초 경보 재현율을 대체하지 않는다. 단절 지속시간·통신 관측·명령/간섭 인접성은 분해 기준이며 원인이 확정된 것은 아니다.','\n전체 사건: events.csv. 사건별 타임라인 원본: results.json의examples.']
    (ROOT/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(folder/'results.json')


if __name__=='__main__':main()
