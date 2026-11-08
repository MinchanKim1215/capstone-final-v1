"""Common truth and strict original event protocol, including missed events."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from .targets import onsets


def binary(y,p,threshold=.5):
    y=np.asarray(y,bool);p=np.asarray(p,float);good=np.isfinite(p);y=y[good];p=p[good]
    if not len(y):return dict(samples=0,ap=None)
    call=p>=threshold;tp=int((call&y).sum());fp=int((call&~y).sum());fn=int((~call&y).sum());tn=int((~call&~y).sum())
    return dict(samples=len(y),positives=int(y.sum()),ap=float(average_precision_score(y,p)) if y.any() else None,
                auroc=float(roc_auc_score(y,p)) if y.any() and not y.all() else None,
                brier=float(brier_score_loss(y,p)),precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,tp=tp,fp=fp,fn=fn,tn=tn)


def wilson(success,total):
    if not total:return [0.,1.]
    z=1.95996398454;p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,float(center-radius)),min(1.,float(center+radius))]


def alert_counts(links,ends,score,support,n_search,threshold=.5):
    times=np.asarray(ends)*.5;end=len(links)*.5-8.5
    use=np.flatnonzero((times>=21)&(times<=end))
    nr,n=score.shape[1:];active=np.zeros((nr,n),bool);starts=[]
    for t in use:
        new=support[t]&np.isfinite(score[t])&(score[t]>=np.where(active,threshold*.5,threshold))
        starts.append(new&~active);active=new
    starts=np.asarray(starts,bool).reshape(len(use),nr,n)
    events=onsets(links);first=correct=hits=total=0
    for r in range(nr):
        for j in range(n):
            if j==r+n_search:continue
            es=np.flatnonzero(events[:,r+n_search,j])*.5
            alerts=times[use][starts[:,r,j]]
            first+=len(alerts)
            # Equivalent to the old [3,7) first-alert protocol, not a generous
            # "any alert before any future outage" count.
            correct+=sum(bool(((es>=a+3)&(es<a+7)).any()) for a in alerts)
            eligible=es[(es>=24)&(es<=end)];total+=len(eligible)
            hits+=sum(bool(((alerts>e-7)&(alerts<=e-3)).any()) for e in eligible)
    return dict(first_alerts=int(first),timely_correct_first_alerts=int(correct),eligible_directed_link_events=int(total),
                event_hits_3s=int(hits),drone_hours=max(0.,end-21)*nr/3600)


def combine_alerts(rows):
    keys=('first_alerts','timely_correct_first_alerts','eligible_directed_link_events','event_hits_3s','drone_hours')
    result={k:sum(r[k] for r in rows) for k in keys};first=result['first_alerts'];total=result['eligible_directed_link_events']
    result.update(precision=result['timely_correct_first_alerts']/first if first else None,
                  event_recall_3s=result['event_hits_3s']/total if total else None,
                  false_or_late_per_drone_hour=(first-result['timely_correct_first_alerts'])/result['drone_hours'] if result['drone_hours'] else None,
                  precision_wilson95=wilson(result['timely_correct_first_alerts'],first))
    return result
