"""Selection-only missed-event causes and honest fixed-rule threshold frontier."""
from collections import Counter
import numpy as np
from . import ARTIFACTS,PARENT
from forecast_navigation.contracts import read,write,sha,now
from forecast_navigation.episodes import cohort
from forecast_rebuild.calibration import transform,alarm_support
from forecast_rebuild.targets import onsets
from forecast_extension.metrics import alert_counts,combine_alerts


def starts(ep,p,support,threshold):
    times=ep.ends*.5;end=len(ep.links)*.5-8.5;active=np.zeros(p.shape[1:],bool);out=[]
    for t in np.flatnonzero((times>=21)&(times<=end)):
        new=support[t]&np.isfinite(p[t])&(p[t].astype(float)>=np.where(active,threshold*.5,threshold))
        out.append((float(times[t]),new&~active));active=new
    return out


def main():
    lookup={i['key']:i for i in read(PARENT/'data_manifest.json')['records'] if i['role']=='selection'}
    frozen=read(PARENT/'freeze.json');result={}
    for task in ('link','service'):
        entry=next(i for i in frozen['candidates'] if i['candidate']['task']==task and i['candidate']['tag']=='v12_mission_blend50')
        policy=read(entry['policy']);summary=read(entry['selection_summary']);data=[];causes=Counter();maximum=[];lead=[]
        for case in summary['cases']:
            ep=cohort(lookup[case['key']],task)
            with np.load(case['predictions_path'],allow_pickle=False) as a:p=transform(a['prediction'],policy['parameters'])[...,5]
            support=alarm_support(ep,policy['require_energy']);data.append((ep,p,support));alarms=starts(ep,p,support,policy['threshold'])
            times=ep.ends*.5;end=len(ep.links)*.5-8.5;truth=onsets(ep.links);nr,n=p.shape[1:];ns=ep.record['n_search']
            for r in range(nr):
                for j in range(n):
                    if j==ns+r:continue
                    es=np.flatnonzero(truth[:,ns+r,j])*.5;es=es[(es>=24)&(es<=end)]
                    aa=np.array([t for t,a in alarms if a[r,j]])
                    for when in es:
                        before=(times>when-7)&(times<=when-3)&(times>=21);known=before&support[:,r,j]&np.isfinite(p[:,r,j])
                        hit=bool(((aa>when-7)&(aa<=when-3)).any());high=bool((p[known,r,j]>=policy['threshold']).any())
                        cause='timely_first_alert' if hit else 'no_qualified_input' if not known.any() else 'risk_below_threshold' if not high else 'already_active_without_new_alert'
                        causes[cause]+=1;causes['events']+=1;causes['any_high_risk_events']+=int(high)
                        if known.any():maximum.append(float(p[known,r,j].max()))
                        values=[]
                        for offset in range(-12,1):
                            q=np.flatnonzero((times>when+offset-.51)&(times<=when+offset+.01)&support[:,r,j]&np.isfinite(p[:,r,j]))
                            values.append(float(p[q[-1],r,j]) if len(q) else np.nan)
                        lead.append(values)
        thresholds=np.unique(np.r_[np.geomspace(1e-5,.01,14),np.linspace(.01,.99,99),policy['threshold']]);frontier=[]
        for th in thresholds:
            a=combine_alerts([alert_counts(ep.links,ep.ends,p,s,ep.record['n_search'],float(th)) for ep,p,s in data]);a['threshold']=float(th);frontier.append(a)
        def best(test):
            a=[p for p in frontier if test(p)]
            return max(a,key=lambda p:(p['event_recall_3s'] or 0,p['precision'] or 0)) if a else None
        result[task]=dict(candidate=entry['candidate'],policy=entry['policy'],policy_sha256=sha(entry['policy']),threshold=policy['threshold'],
            causes=dict(causes),event_risk_max_quantiles=np.quantile(maximum,[0,.1,.25,.5,.75,.9,1]).tolist(),
            lead_offsets_s=list(range(-12,1)),median_onset_aligned_risk=np.nanmedian(lead,axis=0).tolist(),
            best_recall_any=best(lambda p:True),best_FA1=best(lambda p:p['first_alerts']>=50 and p['false_or_late_per_drone_hour']<=1),
            best_P90_FA1=best(lambda p:p['first_alerts']>=50 and p['precision'] is not None and p['precision']>=.9 and p['false_or_late_per_drone_hour']<=1),frontier=frontier)
        print(task,dict(causes),'FA1',result[task]['best_FA1'],'P90FA1',result[task]['best_P90_FA1'],flush=True)
    write(ARTIFACTS/'selection_diagnosis.json',dict(created_at=now(),scope='Selection45 only; previous confirmation not fitted or used',tasks=result,
        parent_freeze_sha256=sha(PARENT/'freeze.json'),new_confirmation_required=True))


if __name__=='__main__':main()
