"""Selection-only interpretation after freeze; never used to fit or choose models."""
from collections import Counter
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,now
from .episodes import cohort
from .evaluate import verify_freeze
from forecast_rebuild.dataset import Episode
from forecast_rebuild.targets import onsets
from forecast_rebuild.calibration import transform,alarm_support


def connected(links,ns):
    a=np.array(links,bool);np.fill_diagonal(a,True)
    for k in range(len(a)):a |= a[:,k,None]&a[None,k,:]
    return bool(a[:ns,:ns].all())


def main():
    frozen=verify_freeze();choice=frozen['decisions']['service']['candidate']
    item=next(i for i in frozen['candidates'] if i['candidate']==choice)
    summary=read(item['selection_summary']);policy=read(item['policy'])
    lookup={i['key']:i for i in read(ARTIFACTS/'data_manifest.json')['records'] if i['role']=='selection'}
    counts=Counter();patterns=Counter();durations=Counter();examples=[];maxima=[]
    for case in summary['cases']:
        entry=lookup[case['key']];r=entry['source_record'];ep=Episode(r);service=cohort(r,'service',ep)
        with np.load(case['predictions_path'],allow_pickle=False) as a:p=transform(a['prediction'],policy['parameters'])[:,0,0,5]
        times=ep.ends*.5;support=alarm_support(service,False)[:,0,0];active=False;alerts=[];end=len(ep.links)*.5-8.5
        for t in np.flatnonzero((times>=21)&(times<=end)):
            new=bool(support[t] and np.isfinite(p[t]) and float(p[t])>=(policy['threshold']*.5 if active else policy['threshold']))
            if new and not active:alerts.append(times[t])
            active=new
        alerts=np.asarray(alerts);events=np.flatnonzero(onsets(service.links)[:,1,0]);health=service.links[:,1,0]
        for q in events:
            when=q*.5
            if not 24<=when<=end:continue
            before=(times>when-7)&(times<=when-3)&(times>=21);known=before&support&np.isfinite(p)
            hit=bool(((alerts>when-7)&(alerts<=when-3)).any());high=bool((p[known]>=policy['threshold']).any())
            outcome='timely_first_alert' if hit else 'no_qualified_history' if not known.any() else 'risk_below_entry_threshold' if not high else 'high_risk_without_new_first_alert'
            counts[outcome]+=1;counts['eligible_events']+=1;counts['any_high_risk_events']+=int(high)
            maximum=float(p[known].max()) if known.any() else None
            if maximum is not None:maxima.append(maximum)
            current=connected(ep.links[q],r['n_search']);prior=connected(ep.links[max(0,q-6)],r['n_search'])
            patterns[f'connected_3s_before_{prior}__connected_at_onset_{current}']+=1
            recover=q+1
            while recover<len(health) and not health[recover]:recover+=1
            duration=(recover-q)*.5;durations['1_to_2s' if duration<2 else '2_to_5s' if duration<5 else 'at_least_5s']+=1
            if len(examples)<12:examples.append(dict(key=r['key'],region=r['region'],kind=r['spec']['kind'],onset_s=when,duration_s=duration,
                outcome=outcome,maximum_qualified_risk=maximum,physical_search_connected_3s_before=prior,physical_search_connected_at_onset=current))
    total=counts['eligible_events'];value=dict(created_at=now(),scope='Selection45 only; post-freeze interpretation, no optimization',candidate=choice['tag'],
        threshold=policy['threshold'],outcomes=dict(counts),first_alert_event_recall=counts['timely_first_alert']/max(total,1),
        any_high_risk_event_coverage=counts['any_high_risk_events']/max(total,1),physical_connectivity_associations=dict(patterns),
        service_duration_bands=dict(durations),qualified_pre_event_risk_quantiles=np.quantile(maxima,[0,.25,.5,.75,.9,1]).tolist(),examples=examples,
        note='Whole-network connectivity and future event timing here are diagnostic truth, never inputs. Associations do not prove causes or an information-theoretic prediction limit.')
    write(ARTIFACTS/'service_diagnostics.json',value);print({k:value[k] for k in ('candidate','outcomes','first_alert_event_recall','any_high_risk_event_coverage','physical_connectivity_associations')})


if __name__=='__main__':main()
