"""Same precision/first-alert/FA goals, fitted on calibration geography only."""
from pathlib import Path
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from .contracts import read,write,sha,now,SEED,folds
from .evaluate import records
from forecast_service.calibration import trials_for,best
from forecast_rebuild.calibration import alarm_support,transform,meets,reliability
from forecast_extension.metrics import alert_counts,combine_alerts,binary


def sweep(path):
    s=read(path)
    if s['scope']!='selection':raise ValueError('Selection only')
    out=Path(path).parent/'alarm_sweep.json'
    if out.exists():
        value=read(out)
        if value['summary_sha256']!=sha(path):raise ValueError('Selection changed')
        return value
    trials=trials_for(records(s),s['task']=='link')
    partial=[r for r in trials if r['first_alerts']>=50 and r['precision'] is not None and r['precision']>=.9 and r['false_or_late_per_drone_hour']<=1]
    value=dict(summary_sha256=sha(path),full=best([r for r in trials if meets(r)]),partial=best(partial),trials=trials)
    write(out,value);return value


def fit(path):
    s=read(path)
    if s['scope']!='calibration':raise ValueError('Calibration geography only')
    groups={region:sorted(c['kind'] for c in s['cases'] if c['region']==region) for region in {c['region'] for c in s['cases']}}
    if set(groups)!=set(folds()['calibration']) or any(k!=['clear','continuous','intermittent'] for k in groups.values()):raise ValueError('Incomplete balanced calibration cohort')
    out=Path(path).parent/'policy.json'
    if out.exists():
        value=read(out)
        if value['summary_sha256']!=sha(path) or value['implementation_sha256']!=sha(__file__):raise ValueError('Calibration changed')
        return out
    data=records(s);energy=s['task']=='link';parameters={};rng=np.random.default_rng(SEED+71);degenerate=False
    for head in (2,5):
        pp=[];yy=[]
        for ep,p in data:
            valid=ep.valid&alarm_support(ep,energy)&np.isfinite(p[...,head])
            if head==2:valid &= ep.currently_up
            pp.append(p[...,head][valid]);yy.append(ep.targets[...,head][valid])
        p=np.concatenate(pp);y=np.concatenate(yy);take=rng.choice(len(y),min(250000,len(y)),replace=False)
        reason=None
        if np.unique(y[take]).size<2:
            slope=0.;intercept=float(logit(np.clip(y[take].mean(),1e-6,1-1e-6)));reason='single_class'
        else:
            model=LogisticRegression(C=1.,solver='lbfgs',max_iter=100,tol=1e-8).fit(logit(np.clip(p[take],1e-6,1-1e-6))[:,None],y[take])
            slope=float(model.coef_[0,0]);intercept=float(model.intercept_[0])
            if slope<=0:slope=0.;intercept=float(logit(np.clip(y[take].mean(),1e-6,1-1e-6)));reason='nonpositive_slope_intercept_only'
        degenerate|=slope==0;parameters[str(head)]=dict(slope=slope,intercept=intercept,samples=len(take),degenerate_reason=reason)
    trials=trials_for([(ep,transform(p,parameters)) for ep,p in data],energy)
    full=[r for r in trials if meets(r)];advisory=[r for r in trials if r['first_alerts']>=10 and r['precision'] is not None and r['precision']>=.8 and r['false_or_late_per_drone_hour']<=5]
    budget=[r for r in trials if r['first_alerts']>=50 and r['false_or_late_per_drone_hour']<=1]
    partial=[r for r in budget if r['precision'] is not None and r['precision']>=.9]
    selected=best(full) or best(advisory) or min(trials,key=lambda r:abs(r['threshold']-.5))
    write(out,dict(task=s['task'],model_tag=s['model_tag'],parameters=parameters,threshold=selected['threshold'],selected=selected,
        common_FA_budget1=best(budget),precision90_FA1_partial=best(partial),trials=trials,require_energy=energy,
        enabled=bool(full) and not degenerate,degenerate_calibration=degenerate,summary_sha256=sha(path),implementation_sha256=sha(__file__),
        created_at=now(),production_ready=False,hardware_validated=False))
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
        if old['summary_sha256']!=sha(path) or old['policy_sha256']!=sha(policy_path):raise ValueError('Calibrated evaluation changed')
        return out
    data=records(s);counts=[];budget=[];partial=[];p5=[];p37=[];y5=[];y37=[]
    for ep,p in data:
        c=transform(p,policy['parameters']);support=alarm_support(ep,policy['require_energy'])
        counts.append(alert_counts(ep.links,ep.ends,c[...,5],support,ep.record['n_search'],policy['threshold']))
        for name,acc in [('common_FA_budget1',budget),('precision90_FA1_partial',partial)]:
            if policy[name]:acc.append(alert_counts(ep.links,ep.ends,c[...,5],support,ep.record['n_search'],policy[name]['threshold']))
        common=ep.valid&ep.legacy_support;live=common&ep.currently_up
        p5.append(c[...,2][live]);y5.append(ep.targets[...,2][live]);p37.append(c[...,5][common]);y37.append(ep.targets[...,5][common])
    alarms=combine_alerts(counts)
    write(out,dict(task=s['task'],scope=s['scope'],model_tag=s['model_tag'],summary_sha256=sha(path),policy_sha256=sha(policy_path),policy_path=str(policy_path),
        alarms=alarms,case_alarms=[dict(key=c['key'],region=c['region'],**counts[i]) for i,c in enumerate(s['cases'])],
        common_FA_budget1=combine_alerts(budget) if budget else None,precision90_FA1_partial=combine_alerts(partial) if partial else None,
        calibrated_next5=binary(np.concatenate(y5),np.concatenate(p5)),calibrated_any3to7=binary(np.concatenate(y37),np.concatenate(p37)),
        reliability_next5=reliability(np.concatenate(y5),np.concatenate(p5)),reliability_any3to7=reliability(np.concatenate(y37),np.concatenate(p37)),
        calibration_goal_met=policy['enabled'],current_fold_goal_met=meets(alarms),
        trusted_simulation_alarm_enabled=bool(policy['enabled'] and meets(alarms)),production_ready=False,hardware_validated=False,created_at=now()))
    return out

