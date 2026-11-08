"""Calibration-only threshold selection with fixed original alarm metric."""
from pathlib import Path
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from .contracts import read,write,sha,now,SEED
from .evaluate import records
from .metrics import trials,choose
from .inference import parent_entry
from forecast_rebuild.calibration import transform,meets,reliability
from forecast_extension.metrics import alert_counts,combine_alerts,binary


def frozen_policy():
    from . import PARENT,ARTIFACTS
    source=read(parent_entry()['policy']);plan=read(PARENT/'secondary_operating_points_plan.json')
    entry=plan['items'][parent_entry()['candidate']['tag']]
    point=entry['points']['5']
    source.update(threshold=point['threshold'],selected=point,rule='Existing user-selected V13FA5 point preserved',enabled=False)
    target=ARTIFACTS/'frozen_v13_policy.json'
    if target.exists() and read(target)!=source:raise ValueError('Frozen V13 policy changed')
    if not target.exists():write(target,source)
    return target


def fit(path):
    s=read(path)
    if s['scope']!='calibration':raise ValueError('Calibration only')
    if s['candidate']['tag']=='frozen_v13':return frozen_policy()
    out=Path(path).parent/'policy.json'
    if out.exists():
        v=read(out)
        if v['summary_sha256']!=sha(path) or v['implementation_sha256']!=sha(__file__):raise ValueError('Policy changed')
        return out
    data=records(s);parameters={};rng=np.random.default_rng(SEED+71);degenerate=False
    if s['candidate']['tag']=='retuned_v13':
        old=read(parent_entry()['policy']);parameters=old['parameters'];degenerate=old['degenerate_calibration']
    else:
        for head in (2,5):
            pp=[];yy=[]
            for ep,p in data:
                valid=ep.valid&ep.legacy_support&np.isfinite(p[...,head])
                if head==2:valid &= ep.currently_up
                pp.append(p[...,head][valid]);yy.append(ep.targets[...,head][valid])
            p=np.concatenate(pp);y=np.concatenate(yy);take=rng.choice(len(y),min(250000,len(y)),replace=False);reason=None
            if np.unique(y[take]).size<2:slope=0.;intercept=float(logit(np.clip(y[take].mean(),1e-6,1-1e-6)));reason='single_class'
            else:
                model=LogisticRegression(C=1.,solver='lbfgs',max_iter=100,tol=1e-8).fit(logit(np.clip(p[take],1e-6,1-1e-6))[:,None],y[take]);slope=float(model.coef_[0,0]);intercept=float(model.intercept_[0])
                if slope<=0:slope=0.;intercept=float(logit(np.clip(y[take].mean(),1e-6,1-1e-6)));reason='nonpositive_slope_intercept_only'
            parameters[str(head)]=dict(slope=slope,intercept=intercept,samples=len(take),degenerate_reason=reason);degenerate |= slope==0
    calibrated=[(ep,transform(p,parameters)) for ep,p in data];rows=trials(calibrated,full=True);point,rule,grade=choose(rows)
    budget=[r for r in rows if r['first_alerts']>=50 and r['false_or_late_per_drone_hour']<=1]
    partial=[r for r in budget if r['precision'] is not None and r['precision']>=.9]
    def best(a):return max(a,key=lambda r:(r['event_recall_3s'] or 0,r['precision'] or 0)) if a else None
    write(out,dict(task='service',model_tag=s['model_tag'],parameters=parameters,threshold=point['threshold'],selected=point,rule=rule,operating_point_grade=grade,
        common_FA_budget1=best(budget),precision90_FA1_partial=best(partial),require_energy=False,enabled=meets(point) and not degenerate,
        degenerate_calibration=degenerate,trials=rows,summary_sha256=sha(path),implementation_sha256=sha(__file__),created_at=now(),
        production_ready=False,hardware_validated=False,threshold_only_fixed_parent_probability=s['candidate']['tag']=='retuned_v13'))
    return out


def apply(path,policy_path):
    s=read(path);policy=read(policy_path)
    if s['scope']=='confirmation':
        from .evaluate import verify_freeze
        item=next(i for i in verify_freeze()['candidates'] if i['candidate']==s['candidate'])
        if sha(policy_path)!=item['policy_sha256']:raise ValueError('Unfrozen policy')
    out=Path(path).parent/'calibrated.json'
    if out.exists():
        old=read(out)
        if old['summary_sha256']!=sha(path) or old['policy_sha256']!=sha(policy_path):raise ValueError('Calibrated result changed')
        return out
    data=records(s);counts=[];budget=[];partial=[];pp={2:[],5:[]};yy={2:[],5:[]}
    for ep,p in data:
        c=transform(p,policy['parameters']);support=ep.legacy_support
        counts.append(alert_counts(ep.links,ep.ends,c[...,5],support,1,policy['threshold']))
        for name,dest in [('common_FA_budget1',budget),('precision90_FA1_partial',partial)]:
            if policy.get(name):dest.append(alert_counts(ep.links,ep.ends,c[...,5],support,1,policy[name]['threshold']))
        for head in (2,5):
            valid=ep.valid&support
            if head==2:valid &= ep.currently_up
            pp[head].append(c[...,head][valid]);yy[head].append(ep.targets[...,head][valid])
    alarms=combine_alerts(counts);p5=np.concatenate(pp[2]);y5=np.concatenate(yy[2]);p37=np.concatenate(pp[5]);y37=np.concatenate(yy[5])
    write(out,dict(task='service',scope=s['scope'],model_tag=s['model_tag'],summary_sha256=sha(path),policy_sha256=sha(policy_path),policy_path=str(policy_path),
        alarms=alarms,case_alarms=[dict(key=case['key'],region=case['region'],**counts[i]) for i,case in enumerate(s['cases'])],
        common_FA_budget1=combine_alerts(budget) if budget else None,precision90_FA1_partial=combine_alerts(partial) if partial else None,
        calibrated_next5=binary(y5,p5),calibrated_any3to7=binary(y37,p37),reliability_next5=reliability(y5,p5),reliability_any3to7=reliability(y37,p37),
        calibration_goal_met=policy['enabled'],current_fold_goal_met=meets(alarms),trusted_simulation_alarm_enabled=bool(policy['enabled'] and meets(alarms)),
        production_ready=False,hardware_validated=False,created_at=now()))
    return out
