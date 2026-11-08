"""Explain event observability without excluding unobservable events from recall."""
import numpy as np
from .targets import onsets
from .calibration import alarm_support


def observability(episode):
    times=np.asarray(episode.ends)*.5;end=len(episode.links)*.5-8.5
    use=(times>=21)&(times<=end);events=onsets(episode.links)
    masks=dict(legacy=episode.legacy_support,extended=episode.available,energy_qualified=alarm_support(episode,True))
    counts=dict(events=0,legacy=0,extended=0,energy_qualified=0)
    ns=episode.record['n_search'];nr,n=episode.values.shape[1:3]
    for r in range(nr):
        for j in range(n):
            if j==r+ns:continue
            es=np.flatnonzero(events[:,r+ns,j])*.5
            for event in es[(es>=24)&(es<=end)]:
                counts['events']+=1
                window=use&(times>event-7)&(times<=event-3)
                for name,mask in masks.items():counts[name]+=int(mask[window,r,j].any())
    return counts


def combine(rows):
    sums={key:sum(r[key] for r in rows) for key in ('events','legacy','extended','energy_qualified')}
    return dict(**sums,possible_recall_ceiling={k:sums[k]/max(1,sums['events']) for k in ('legacy','extended','energy_qualified')},
        note='Input-eligibility upper bound only. It assumes a perfect predictor and allows an alert at every eligible pre-event instant; it is not achieved recall.')


def severity(episode,score,threshold=.5,require_energy=False):
    times=np.asarray(episode.ends)*.5;end=len(episode.links)*.5-8.5
    use=np.flatnonzero((times>=21)&(times<=end));support=alarm_support(episode,require_energy)
    nr,n=score.shape[1:];active=np.zeros((nr,n),bool);starts=[]
    for t in use:
        new=support[t]&np.isfinite(score[t])&(score[t]>=np.where(active,threshold*.5,threshold))
        starts.append(new&~active);active=new
    starts=np.asarray(starts,bool);events=onsets(episode.links);ns=episode.record['n_search']
    groups={name:dict(events=0,first_alert_hits_3s=0,any_high_risk_before_3s=0) for name in ('duration_1to2s','duration_2to5s','duration_at_least5s')}
    for r in range(nr):
        for j in range(n):
            if j==r+ns:continue
            es=np.flatnonzero(events[:,r+ns,j]);alerts=times[use][starts[:,r,j]]
            for q in es[(es*.5>=24)&(es*.5<=end)]:
                later=np.flatnonzero(episode.links[q:,r+ns,j])
                duration=(int(later[0]) if len(later) else len(episode.links)-q)*.5
                category='duration_1to2s' if duration<2 else 'duration_2to5s' if duration<5 else 'duration_at_least5s'
                row=groups[category];row['events']+=1;event=q*.5
                row['first_alert_hits_3s']+=int(((alerts>event-7)&(alerts<=event-3)).any())
                window=(times[use]>event-7)&(times[use]<=event-3)
                row['any_high_risk_before_3s']+=int((support[use,r,j]&window&(score[use,r,j]>=threshold)).any())
    return groups
