"""Optimize exact event recall under false-alarm budget; retain original goals."""
import numpy as np
from forecast_extension.metrics import alert_counts,combine_alerts,binary
from forecast_rebuild.calibration import meets
THRESHOLDS=np.unique(np.r_[np.geomspace(.0001,.05,15),np.linspace(.05,.95,46),.5,.975,.99,.995])


def trials(data,full=False):
    if full:
        from forecast_service.calibration import trials_for
        return trials_for(data,False)
    result=[]
    for threshold in THRESHOLDS:
        r=combine_alerts([alert_counts(ep.links,ep.ends,p[...,5],ep.legacy_support,1,float(threshold)) for ep,p in data])
        r['threshold']=float(threshold);result.append(r)
    return result


def choose(rows):
    full=[r for r in rows if meets(r)]
    budget=[r for r in rows if r['first_alerts']>=50 and r['false_or_late_per_drone_hour']<=1]
    exploratory=[r for r in rows if r['first_alerts']>=10 and r['false_or_late_per_drone_hour']<=1]
    def best(a):return max(a,key=lambda r:(r['event_recall_3s'] or 0,r['precision'] or 0,-r['false_or_late_per_drone_hour']))
    if full:return best(full),'original_full_goals',3
    if budget:return best(budget),'FA1_min50_recall_fallback',2
    if exploratory:return best(exploratory),'FA1_min10_exploratory_only',1
    r=min(rows,key=lambda r:abs(r['threshold']-.5));return r,'no_FA1_operating_point_half_fallback',0


def evaluate_rows(data,full=False):
    pp=[];yy=[]
    for ep,p in data:
        valid=ep.valid&ep.legacy_support
        pp.append(p[...,5][valid]);yy.append(ep.targets[...,5][valid])
    ap=binary(np.concatenate(yy),np.concatenate(pp));rows=trials(data,full);point,rule,grade=choose(rows)
    score=(grade,point['event_recall_3s'] or 0,point['precision'] or 0,ap['ap'] or 0)
    return dict(score=list(score),point=point,rule=rule,ap37=ap,full_goal_met=meets(point))
