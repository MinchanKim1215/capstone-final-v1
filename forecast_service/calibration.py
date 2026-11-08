"""Task-specific calibration and fixed alarm operating points, no test fitting."""
from pathlib import Path
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from .contracts import read,write,sha,now,folds,SEED
from .data import ServiceEpisode
from forecast_rebuild.dataset import Episode
from forecast_rebuild.calibration import alarm_support,transform,meets,reliability
from forecast_extension.metrics import alert_counts,combine_alerts,binary


def records(summary):
    if sha(summary['dataset_path'])!=summary['dataset_sha256']:raise ValueError('Evaluated inputs changed')
    rows={r['key']:r for r in read(summary['dataset_path'])['rows']};result=[]
    for case in summary['cases']:
        r=rows[case['key']]
        if r['role']!=summary['scope']:raise ValueError('Cross-fold fitting is forbidden')
        if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Raw prediction changed')
        with np.load(case['predictions_path'],allow_pickle=False) as a:p=a['prediction']
        result.append((Episode(r) if summary['task']=='link' else ServiceEpisode(r),p))
    return result


def require_energy(summary):return summary['task']=='link' and bool(read(summary['model_manifest']).get('use_energy',False))


def trials_for(data,energy):
    pooled=np.concatenate([p[...,5][ep.valid&alarm_support(ep,energy)&np.isfinite(p[...,5])] for ep,p in data])
    thresholds=np.unique(np.r_[np.linspace(.01,.99,50),np.quantile(pooled,np.linspace(.5,.999,35))]);trials=[]
    for threshold in thresholds:
        counts=combine_alerts([alert_counts(ep.links,ep.ends,p[...,5],alarm_support(ep,energy),ep.record['n_search'],float(threshold)) for ep,p in data])
        trials.append(dict(threshold=float(threshold),**counts))
    return trials


def best(values):return max(values,key=lambda r:(r['event_recall_3s'],r['precision'],-r['false_or_late_per_drone_hour'])) if values else None


def sweep(summary_path):
    summary=read(summary_path)
    if summary['scope']!='selection':raise ValueError('Model selection uses selection only')
    path=Path(summary_path).parent/'alarm_sweep.json'
    if path.exists():
        prior=read(path)
        if prior['summary_sha256']!=sha(summary_path):raise ValueError('Selection changed')
        return prior
    trials=trials_for(records(summary),require_energy(summary))
    partial=[r for r in trials if r['first_alerts']>=50 and r['precision'] is not None and r['precision']>=.9 and r['false_or_late_per_drone_hour']<=1]
    result=dict(model_tag=summary['model_tag'],summary_sha256=sha(summary_path),full=best([r for r in trials if meets(r)]),partial=best(partial),trials=trials)
    write(path,result);return result


def fit(summary_path):
    summary=read(summary_path)
    if summary['scope']!='calibration':raise ValueError('Calibration-fold fitting only')
    cases=summary['cases'];groups={region:sorted(c['kind'] for c in cases if c['region']==region) for region in {c['region'] for c in cases}}
    if set(groups)!=set(folds()['calibration']) or any(k!=['clear','continuous','intermittent'] for k in groups.values()):raise ValueError('Incomplete balanced calibration')
    out=Path(summary_path).parent/'policy.json'
    if out.exists():
        p=read(out)
        if p['summary_sha256']!=sha(summary_path) or p['implementation_sha256']!=sha(__file__):raise ValueError('Existing calibration differs')
        return out
    data=records(summary);energy=require_energy(summary);rng=np.random.default_rng(SEED+71);parameters={};degenerate=False
    for head in (2,5):
        scores=[];truth=[]
        for ep,p in data:
            good=ep.valid&alarm_support(ep,energy)&np.isfinite(p[...,head])
            if head==2:good &= ep.currently_up
            scores.append(p[...,head][good]);truth.append(ep.targets[...,head][good])
        p=np.concatenate(scores);y=np.concatenate(truth);take=rng.choice(len(y),min(250000,len(y)),replace=False)
        if np.unique(y[take]).size<2:slope=0.;intercept=float(logit(np.clip(y[take].mean(),1e-6,1-1e-6)));reason='single_class_calibration'
        else:
            model=LogisticRegression(C=1.,solver='lbfgs',max_iter=100,tol=1e-8).fit(logit(np.clip(p[take],1e-6,1-1e-6))[:,None],y[take])
            slope=float(model.coef_[0,0]);intercept=float(model.intercept_[0]);reason=None
            if slope<=0:slope=0.;intercept=float(logit(np.clip(y[take].mean(),1e-6,1-1e-6)));reason='nonpositive_slope_intercept_only_no_rank_reversal'
        degenerate |= slope==0
        parameters[str(head)]=dict(slope=slope,intercept=intercept,samples=len(take),positive_fraction=float(y[take].mean()),degenerate_reason=reason)
    calibrated=[(ep,transform(p,parameters)) for ep,p in data];trials=trials_for(calibrated,energy)
    full=[r for r in trials if meets(r)];advisory=[r for r in trials if r['first_alerts']>=10 and r['precision'] is not None and r['precision']>=.8 and r['false_or_late_per_drone_hour']<=5]
    budget=[r for r in trials if r['first_alerts']>=50 and r['false_or_late_per_drone_hour']<=1]
    partial=[r for r in budget if r['precision'] is not None and r['precision']>=.9]
    selected=best(full) or best(advisory) or min(trials,key=lambda r:abs(r['threshold']-.5))
    write(out,dict(model_tag=summary['model_tag'],task=summary['task'],parameters=parameters,threshold=selected['threshold'],selected=selected,
        common_FA_budget1=best(budget),precision90_FA1_partial=best(partial),trials=trials,require_energy=energy,
        enabled=bool(full) and not degenerate,degenerate_calibration=degenerate,summary_sha256=sha(summary_path),implementation_sha256=sha(__file__),created_at=now(),
        production_ready=False,hardware_validated=False,scope_note='Service task criterion is a study assumption. Nonpositive Platt slopes become intercept-only, never reversed risk rankings.'))
    return out


def apply(summary_path,policy_path):
    summary=read(summary_path);policy=read(policy_path)
    if summary['scope']=='confirmation':
        from .evaluate import verify_freeze
        frozen=verify_freeze();item=next(i for i in frozen['models'] if i['task']==summary['task'] and i['pin']['path']==summary['model_manifest'])
        if sha(policy_path)!=item['policy_sha256']:raise ValueError('Unfrozen alarm policy')
    data=records(summary);counts=[];budget=[];partial=[];p5=[];p37=[];y5=[];y37=[]
    for ep,p in data:
        c=transform(p,policy['parameters']);support=alarm_support(ep,policy['require_energy'])
        counts.append(alert_counts(ep.links,ep.ends,c[...,5],support,ep.record['n_search'],policy['threshold']))
        for name,acc in [('common_FA_budget1',budget),('precision90_FA1_partial',partial)]:
            if policy[name]:acc.append(alert_counts(ep.links,ep.ends,c[...,5],support,ep.record['n_search'],policy[name]['threshold']))
        common=ep.valid&ep.legacy_support;live=common&ep.currently_up
        p5.append(c[...,2][live]);y5.append(ep.targets[...,2][live]);p37.append(c[...,5][common]);y37.append(ep.targets[...,5][common])
    a=np.concatenate(p5);ya=np.concatenate(y5);b=np.concatenate(p37);yb=np.concatenate(y37);alarms=combine_alerts(counts)
    out=Path(summary_path).parent/'calibrated.json'
    write(out,dict(task=summary['task'],scope=summary['scope'],model_tag=summary['model_tag'],policy_sha256=sha(policy_path),policy_path=str(policy_path),
        alarms=alarms,case_alarms=[dict(key=c['key'],region=c['region'],**counts[i]) for i,c in enumerate(summary['cases'])],
        common_FA_budget1=combine_alerts(budget) if budget else None,precision90_FA1_partial=combine_alerts(partial) if partial else None,
        calibrated_next5=binary(ya,a),calibrated_any3to7=binary(yb,b),reliability_next5=reliability(ya,a),reliability_any3to7=reliability(yb,b),
        calibration_goal_met=policy['enabled'],current_fold_goal_met=meets(alarms),trusted_simulation_alarm_enabled=bool(policy['enabled'] and meets(alarms)),
        production_ready=False,hardware_validated=False,created_at=now()))
    return out
