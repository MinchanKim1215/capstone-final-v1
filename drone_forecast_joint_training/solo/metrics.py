"""Prediction classification and first-alert event metrics with explicit censoring."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss


def binary_metrics(target,probability,threshold=.5):
    y=np.asarray(target,bool).ravel();p=np.asarray(probability,float).ravel()
    if y.shape!=p.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities')
    n=len(y)
    if not n:return dict(samples=0,average_precision=None,auroc=None,brier=None)
    positive=int(y.sum());pred=p>=threshold
    tp=int(np.sum(pred&y));fp=int(np.sum(pred&~y));fn=int(np.sum(~pred&y));tn=n-tp-fp-fn
    return dict(samples=n,positives=positive,positive_fraction=positive/n,
        average_precision=float(average_precision_score(y,p)) if positive else None,
        auroc=float(roc_auc_score(y,p)) if 0<positive<n else None,
        brier=float(brier_score_loss(y,p)),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        f1=2*tp/max(2*tp+fp+fn,1),accuracy=(tp+tn)/n,threshold=threshold,tp=tp,fp=fp,fn=fn,tn=tn)


def events_from_links(links):
    links=np.asarray(links,bool);events={}
    for relay in range(5):
        i=relay+3
        for j in range(8):
            if j==i:continue
            bad=~links[:,i,j]
            start=np.flatnonzero(bad & np.r_[False,~bad[:-1]])
            events[(relay,j)]=np.asarray([q*.5 for q in start if q+1<len(bad) and bad[q+1]],float)
    return events


def alert_metrics(episodes,head,threshold,exit_threshold=None,horizon=8):
    """Alert generation uses predictions and past-observation support, never future labels."""
    if exit_threshold is None:exit_threshold=threshold*.5
    first=correct=event_total=event_hits=0;drone_hours=0.
    for episode in episodes:
        prediction=episode['prediction'];supported=episode['prediction_supported'];links=episode['links']
        duration=len(links)*.5
        times=np.arange(len(prediction))*3.
        decisions=(times>=21)&(times<=duration-8)
        drone_hours+=float(decisions.sum()*3*5/3600)
        events=events_from_links(links)
        for relay in range(5):
            for peer in range(8):
                if peer==relay+3:continue
                active=False;alerts=[]
                for k in np.flatnonzero(decisions):
                    p=float(prediction[k,relay,peer,head]);available=bool(supported[k,relay,peer])
                    new=available and np.isfinite(p) and (p>=(exit_threshold if active else threshold))
                    if new and not active:alerts.append(times[k])
                    active=new
                alerts=np.asarray(alerts,float);future=events[(relay,peer)]
                first+=len(alerts)
                for alarm in alerts:correct+=int(np.any((future>=alarm)&(future<=alarm+horizon)))
                eligible=future[(future>=24)&(future<=duration-8)]
                event_total+=len(eligible)
                for onset in eligible:event_hits+=int(np.any((alerts>=onset-horizon)&(alerts<=onset-3)))
    return dict(first_alerts=first,correct_first_alerts=correct,false_or_late_first_alerts=first-correct,
        first_alert_precision=correct/max(first,1),false_alerts_per_drone_hour=(first-correct)/max(drone_hours,1e-12),
        first_alert_event_recall_3s=event_hits/max(event_total,1),eligible_events=event_total,event_hits_3s=event_hits,
        drone_hours=drone_hours,head=head,threshold=threshold,exit_threshold=exit_threshold,horizon_s=horizon,
        warmup_s=21,tail_censor_s=8,minimum_lead_s=3,minimum_persistent_outage_s=1)


def alarm_trials(episodes,head,thresholds,exit_thresholds,horizon):
    """Vectorize threshold trials; event definitions exactly match alert_metrics."""
    enter=np.asarray(thresholds,float);leave=np.asarray(exit_thresholds,float)
    if enter.shape!=leave.shape or enter.ndim!=1:raise ValueError('Threshold arrays must match')
    count=len(enter);first=np.zeros(count,np.int64);correct=first.copy();hits=first.copy()
    event_total=0;hours=0.
    pairs=[(r,p) for r in range(5) for p in range(8) if p!=r+3]
    for episode in episodes:
        prediction=episode['prediction'];supported=episode['prediction_supported'];links=episode['links']
        duration=len(links)*.5;all_times=np.arange(len(prediction))*3.
        use=np.flatnonzero((all_times>=21)&(all_times<=duration-8));times=all_times[use]
        hours+=len(use)*3*5/3600
        if not len(use):continue
        scores=np.stack([prediction[use,r,p,head] for r,p in pairs],axis=1)
        available=np.stack([supported[use,r,p] for r,p in pairs],axis=1)&np.isfinite(scores)
        events=events_from_links(links)
        truth=np.zeros((len(use),len(pairs)),bool);windows=[]
        for column,pair in enumerate(pairs):
            future=events[pair]
            if len(future):truth[:,column]=np.any((future[None,:]>=times[:,None])&(future[None,:]<=times[:,None]+horizon),axis=1)
            eligible=future[(future>=24)&(future<=duration-8)];event_total+=len(eligible)
            for onset in eligible:
                indices=np.flatnonzero((times>=onset-horizon)&(times<=onset-3))
                if len(indices):windows.append((column,indices))
        active=np.zeros((count,len(pairs)),bool)
        starts=np.zeros((count,len(use),len(pairs)),bool)
        for k in range(len(use)):
            new=available[k][None,:]&(scores[k][None,:]>=np.where(active,leave[:,None],enter[:,None]))
            beginnings=new&~active;starts[:,k]=beginnings
            first+=beginnings.sum(1);correct+=(beginnings&truth[k][None,:]).sum(1);active=new
        for column,indices in windows:hits+=starts[:,indices,column].any(axis=1)
    return [dict(first_alerts=int(first[k]),correct_first_alerts=int(correct[k]),false_or_late_first_alerts=int(first[k]-correct[k]),
        first_alert_precision=float(correct[k]/max(first[k],1)),false_alerts_per_drone_hour=float((first[k]-correct[k])/max(hours,1e-12)),
        first_alert_event_recall_3s=float(hits[k]/max(event_total,1)),eligible_events=event_total,event_hits_3s=int(hits[k]),
        drone_hours=hours,head=head,threshold=float(enter[k]),exit_threshold=float(leave[k]),horizon_s=horizon,
        warmup_s=21,tail_censor_s=8,minimum_lead_s=3,minimum_persistent_outage_s=1) for k in range(count)]


def fit_alarm(episodes,head,horizon):
    # Quantile thresholds matter when a calibrated rare-event model emits probabilities below .01.
    samples=[]
    for e in episodes:
        times=np.arange(len(e['prediction']))*3.;valid_time=(times>=21)&(times<=len(e['links'])*.5-8)
        p=e['prediction'][...,head];valid=e['prediction_supported']&valid_time[:,None,None]&np.isfinite(p)
        samples.append(p[valid])
    scores=np.concatenate(samples) if samples else np.empty(0)
    quantiles=np.quantile(scores,np.unique(np.r_[np.linspace(0,1,49),np.linspace(.9,1,51)])) if len(scores) else []
    thresholds=np.unique(np.clip(np.r_[np.linspace(.01,.99,35),quantiles,.7674193548387096,.7451612903225806],0,1))
    enter=np.repeat(thresholds,3);leave=enter*np.tile([.25,.5,1.],len(thresholds))
    trials=alarm_trials(episodes,head,enter,leave,horizon)
    for trial in trials:trial['qualifies']=trial['first_alerts']>=10 and trial['first_alert_precision']>=.8 and trial['false_alerts_per_drone_hour']<=5
    feasible=[x for x in trials if x['qualifies']]
    best=max(feasible or trials,key=lambda x:(x['first_alert_event_recall_3s'],x['first_alert_precision'],-x['false_alerts_per_drone_hour']))
    return dict(best,enabled=bool(feasible),selection_source='validation only',search_candidates=len(trials),
                search='Empirical probability quantiles + fixed/legacy thresholds; exit/enter ratios .25/.5/1',
                limitation='Disabled if precision/false-alarm/count constraints are infeasible in the validation search; raw probability control is evaluated separately')
