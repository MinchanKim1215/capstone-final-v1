"""Separate window prediction, timely first alerts, and mission metrics."""
import numpy as np
from solo.metrics import binary_metrics
from .targets import TARGET_NAMES,binary_targets


def onset_events(links,n_search):
    up=np.asarray(links,bool);out={}
    starts=np.zeros_like(up)
    starts[1:-1]=up[:-2]&~up[1:-1]&~up[2:]
    for i in range(n_search,up.shape[1]):
        for j in range(up.shape[1]):
            if i!=j:out[(i-n_search,j)]=np.flatnonzero(starts[:,i,j])*.5
    return out


def alert_counts(a,n_search,enter=.5,exit_threshold=None,cadence=1):
    if not 0<enter<=1 or cadence not in (1,3):raise ValueError('Invalid alarm protocol')
    leave=enter*.5 if exit_threshold is None else exit_threshold
    if not 0<=leave<=enter:raise ValueError('Invalid hysteresis')
    times=a['aux_endpoints']*.5
    end=len(a['links'])*.5-8.5
    use=np.flatnonzero((times>=21)&(times<=end)&(np.mod(times,cadence)==0))
    nr,n=a['links'].shape[1]-n_search,a['links'].shape[1]
    active=np.zeros((nr,n),bool);starts=[]
    for t in use:
        score=a['aux_probabilities'][t,...,2]
        new=a['aux_support'][t]&np.isfinite(score)&(score>=np.where(active,leave,enter))
        starts.append(new&~active);active=new
    starts=np.asarray(starts,bool).reshape(len(use),nr,n)
    first=correct=hits=total=0
    for (r,j),events in onset_events(a['links'],n_search).items():
        alerts=times[use][starts[:,r,j]]
        first+=len(alerts)
        # Strictly match the predicted [3,7) interval, unlike legacy any-8s precision.
        correct+=sum(bool(((events>=t+3)&(events<t+7)).any()) for t in alerts)
        eligible=events[(events>=24)&(events<=end)]
        total+=len(eligible)
        hits+=sum(bool(((alerts>e-7)&(alerts<=e-3)).any()) for e in eligible)
    return dict(first_alerts=int(first),timely_correct_first_alerts=int(correct),
                eligible_directed_link_events=int(total),event_hits_3s=int(hits),
                drone_hours=max(0.,end-21)*nr/3600)


def summarize_alerts(counts):
    keys=('first_alerts','timely_correct_first_alerts','eligible_directed_link_events','event_hits_3s','drone_hours')
    out={k:sum(r[k] for r in counts) for k in keys}
    out.update(timely_first_alert_precision=out['timely_correct_first_alerts']/out['first_alerts'] if out['first_alerts'] else None,
               event_recall_3s=out['event_hits_3s']/out['eligible_directed_link_events'] if out['eligible_directed_link_events'] else None,
               false_or_late_per_drone_hour=(out['first_alerts']-out['timely_correct_first_alerts'])/out['drone_hours'] if out['drone_hours'] else None)
    return out


def summarize(episodes,threshold=.5):
    rows=[];targets=[];probability=[];counts={1:[],3:[]};known=possible=0
    for a,r in episodes:
        rows.append(r)
        if len(a['aux_endpoints']):
            known+=int(a['aux_support'].sum())
            possible+=len(a['aux_endpoints'])*(r['total_drones']-r['search_drones'])*(r['total_drones']-1)
            good=a['aux_valid']&a['aux_support']
            targets.append(binary_targets(a['aux_labels'])[good])
            probability.append(a['aux_probabilities'][good])
        for cadence in counts:counts[cadence].append(alert_counts(a,r['search_drones'],threshold,cadence=cadence))
    if not rows:raise ValueError('Empty evaluation')
    scope='SMOKE ONLY: execution checks, not performance evidence' if any(r['smoke_only'] for r in rows) else 'Development validation; not independent final test'
    keys=('search_connected_mean','pdr','timely_pdr','outage_total_s','outage_max_s',
          'duration_s','delay_mean_s','delay_p95_s','relay_travel_m','observable_coverage',
          'aoi_mean_s','information_shared_fraction','outage_events','search_formation_rmse_m',
          'relocation_fraction','protection_fraction','actual_delivered_hops_mean')
    means={k:float(np.mean([r[k] for r in rows if r.get(k) is not None]))
           if any(r.get(k) is not None for r in rows) else None for k in keys}
    prediction={}
    if targets:
        y,p=np.concatenate(targets),np.concatenate(probability)
        for h,name in enumerate(TARGET_NAMES):
            m=binary_metrics(y[:,h],p[:,h])
            if not m.get('tp',0)+m.get('fp',0):m['precision']=None
            prediction[name]=m
    return dict(cases=len(rows),regions=len(set(r['region'] for r in rows)),scope=scope,
                completed=sum(r['mission_completed'] for r in rows),safety_passed=all(r['audit']['passed'] for r in rows),
                means=means,global_longest_outage_s=max(r['outage_max_s'] for r in rows),
                prediction=prediction,alarms_at_threshold={str(k):summarize_alerts(v) for k,v in counts.items()} if possible else None,
                compatible_prediction_stream_measured=bool(possible),
                causal_prediction_support_fraction=known/possible if possible else None,
                alarm_threshold=threshold,alarm_definition='Timely first alert: actual onset in [3,7)s; recall requires >=3s lead; same 1s/3s exposure',
                predictor_target='Next persistent onset; v5 any-event AP cannot be compared directly')


def fit_alarm(episodes,precision=.90,recall=.80,false_per_hour=1.,minimum_alerts=20):
    if any(r['smoke_only'] for _,r in episodes):raise ValueError('Never fit an alarm on smoke trajectories')
    samples=[a['aux_probabilities'][a['aux_support']][...,2] for a,_ in episodes if len(a['aux_endpoints']) and a['aux_support'].any()]
    thresholds=np.unique(np.r_[np.linspace(.01,.99,25),
                       np.quantile(np.concatenate(samples),np.linspace(.01,.99,25))]) if samples else np.array([.5])
    trials=[]
    for threshold in thresholds:
        if threshold<=0:continue
        m=summarize_alerts([alert_counts(a,r['search_drones'],float(threshold)) for a,r in episodes])
        m['threshold']=float(threshold)
        m['qualified']=bool(m['first_alerts']>=minimum_alerts and m['timely_first_alert_precision'] is not None
            and m['timely_first_alert_precision']>=precision and m['event_recall_3s'] is not None
            and m['event_recall_3s']>=recall and m['false_or_late_per_drone_hour']<=false_per_hour)
        trials.append(m)
    eligible=[m for m in trials if m['qualified']]
    best=max(eligible,key=lambda m:(m['event_recall_3s'],m['timely_first_alert_precision'])) if eligible else None
    return dict(enabled=best is not None,selected=best,trials=trials,
                development_goals=dict(precision=precision,event_recall=recall,false_per_hour=false_per_hour,minimum_alerts=minimum_alerts),
                scope='Fit on calibration geography only; confirmation and sealed test use this frozen threshold unchanged')


def paired_comparison(rows):
    """Within-case, map-cluster bootstrap. No outcomes enter online selection."""
    groups={}
    for r in rows:
        s=r['spec'];key=(s['region'],s['direction'],s['kind'],s['seed'])
        groups.setdefault(key,{})[r['deployment']]=r
    return groups


def paired_interval(pairs,metric,replicates=1000,seed=730):
    by_map={}
    for old,new in pairs:
        if old[metric] is not None and new[metric] is not None:
            by_map.setdefault(old['region'],[]).append(new[metric]-old[metric])
    means=np.array([np.mean(v) for v in by_map.values()])
    if not len(means):return dict(mean_difference=None,ci95=None,maps=0)
    rng=np.random.default_rng(seed)
    values=means[rng.integers(len(means),size=(replicates,len(means)))].mean(1)
    return dict(mean_difference=float(means.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),maps=len(means),
                warning='Exploratory map-cluster interval; selected development geography is not an independent test')
