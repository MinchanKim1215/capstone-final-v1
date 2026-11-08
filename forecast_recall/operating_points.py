"""Secondary FA-budget diagnostics, locked from calibration before new results.

These do not replace the primary frozen model/policy or relax original goals.
"""
from . import ARTIFACTS
from .contracts import read,write,sha,now


def prepare():
    f=read(ARTIFACTS/'freeze.json');path=ARTIFACTS/'secondary_operating_points_plan.json'
    if path.exists():return read(path)
    if (ARTIFACTS/'completed_comparison.json').exists():raise ValueError('Do not add a new pre-result plan after results')
    items={}
    for item in f['candidates']:
        policy=read(item['policy']);points={}
        for budget in (1,3,5):
            eligible=[r for r in policy['trials'] if r['first_alerts']>=50 and r['false_or_late_per_drone_hour']<=budget]
            points[str(budget)]=max(eligible,key=lambda r:(r['event_recall_3s'] or 0,r['precision'] or 0)) if eligible else None
        items[item['candidate']['tag']]=dict(candidate=item['candidate'],policy=item['policy'],policy_sha256=item['policy_sha256'],points=points)
    value=dict(created_at=now(),primary_freeze_sha256=sha(ARTIFACTS/'freeze.json'),
        timing='After primary freeze, during new raw collection, before any new prediction outcomes. Supplementary diagnostic only.',
        rule='For every frozen candidate, maximize calibration event recall atFA budgets1/3/5 and>=50alerts; same fixed evaluation metric.',
        primary_model_and_policy_unchanged=True,original_goals_unchanged=True,items=items)
    write(path,value);return value


def analyze():
    from .evaluate import records
    from forecast_rebuild.calibration import transform,meets
    from forecast_extension.metrics import alert_counts,combine_alerts
    plan=read(ARTIFACTS/'secondary_operating_points_plan.json');done=read(ARTIFACTS/'completed_comparison.json');result={}
    for outcome in done['outcomes']:
        tag=outcome['candidate']['tag'];i=plan['items'][tag];policy=read(i['policy'])
        if sha(i['policy'])!=i['policy_sha256']:raise ValueError('Secondary frozen policy changed')
        data=[(ep,transform(p,policy['parameters'])) for ep,p in records(read(outcome['raw_summary']))];points={}
        for budget,point in i['points'].items():
            if point is None:points[budget]=None;continue
            r=combine_alerts([alert_counts(ep.links,ep.ends,p[...,5],ep.legacy_support,1,point['threshold']) for ep,p in data])
            points[budget]=dict(calibration_threshold=point['threshold'],calibration_point=point,confirmation=r,original_full_goals_met=meets(r))
        result[tag]=points
    value=dict(created_at=now(),plan_sha256=sha(ARTIFACTS/'secondary_operating_points_plan.json'),models=result,
        note='Supplementary trade-offs, not primary model reselection. FA3/5 do not meet originalFA1goal merely by meeting their own budget.')
    write(ARTIFACTS/'secondary_operating_points.json',value);return value


if __name__=='__main__':prepare()
