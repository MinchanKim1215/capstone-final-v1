"""Recall-first, approximately4false/late alerts per drone-hour."""
import numpy as np
from forecast_recall.metrics import THRESHOLDS
from forecast_extension.metrics import alert_counts,combine_alerts,binary

def trials(data,full=False):
    if full:
        from forecast_service.calibration import trials_for
        return trials_for(data,False)
    out=[]
    for threshold in THRESHOLDS:
        r=combine_alerts([alert_counts(ep.links,ep.ends,p[...,5],ep.legacy_support,1,float(threshold)) for ep,p in data]);r['threshold']=float(threshold);out.append(r)
    return out

def choose(rows):
    for minimum,grade in [(50,2),(10,1)]:
        a=[r for r in rows if r['first_alerts']>=minimum and r['false_or_late_per_drone_hour']<=4]
        if a:return max(a,key=lambda r:(r['event_recall_3s'] or 0,r['precision'] or 0,-r['false_or_late_per_drone_hour'])),f'FA4_min{minimum}',grade
    return min(rows,key=lambda r:abs(r['threshold']-.5)),'NO_FA4_POINT_half_fallback',0

def evaluate_rows(data,full=False):
    y=[];p=[]
    for ep,a in data:
        valid=ep.valid&ep.legacy_support;y.append(ep.targets[...,5][valid]);p.append(a[...,5][valid])
    ap=binary(np.concatenate(y),np.concatenate(p));point,rule,grade=choose(trials(data,full))
    return dict(score=[grade,point['event_recall_3s'] or 0,point['precision'] or 0,ap['ap'] or 0],point=point,rule=rule,ap37=ap)
