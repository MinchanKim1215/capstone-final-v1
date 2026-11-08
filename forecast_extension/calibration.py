"""Fresh calibration only, with a predeclared common false-alarm budget."""
from pathlib import Path
import argparse
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from .contracts import read,write,sha,now,folds,GOALS,SEED
from .evaluate import verify_freeze
from forecast_rebuild.calibration import records,alarm_support,transform,meets,reliability
from .metrics import binary,alert_counts,combine_alerts


def audit_fresh(summary):
    rows={r['key']:r for r in read(summary['dataset_path'])['rows']};groups={}
    for case in summary['cases']:
        record=rows[case['key']]
        if record['source_kind']!='fresh_v9' or record['role']!=summary['scope']:raise ValueError('Calibration/confirmation must use fresh matching-fold flights')
        groups.setdefault(record['region'],[]).append(record['spec']['kind'])
    if set(groups)!=set(folds()[summary['scope']]) or any(sorted(k)!=['clear','continuous','intermittent'] for k in groups.values()):
        raise ValueError('Fresh per-map RF-condition coverage differs')


def best(rows):return max(rows,key=lambda r:(r['event_recall_3s'],r['precision'],-r['false_or_late_per_drone_hour'])) if rows else None


def fit(summary_path):
    summary=read(summary_path)
    if summary['scope']!='calibration':raise ValueError('Only fresh spatial calibration can set thresholds')
    audit_fresh(summary);out=Path(summary_path).parent/'alarm_policy.json'
    if out.exists():
        prior=read(out)
        if prior['source_sha256']==sha(summary_path) and prior['implementation_sha256']==sha(__file__):return out
        raise ValueError('An existing policy cannot silently refit')
    data=records(summary);require_energy=summary['model_manifest']!='baseline' and read(summary['model_manifest']).get('use_energy',False)
    rng=np.random.default_rng(SEED+7);parameters={}
    for head in (2,5):
        scores=[];truth=[]
        for episode,p in data:
            good=episode.valid&alarm_support(episode,require_energy)&np.isfinite(p[...,head])
            if head==2:good &= episode.currently_up
            scores.append(p[...,head][good]);truth.append(episode.targets[...,head][good])
        score=np.concatenate(scores);y=np.concatenate(truth);take=rng.choice(len(y),min(250000,len(y)),replace=False)
        estimator=LogisticRegression(C=1.,solver='lbfgs',max_iter=100,tol=1e-8)
        estimator.fit(logit(np.clip(score[take],1e-6,1-1e-6))[:,None],y[take]);slope=float(estimator.coef_[0,0])
        if slope<=0:raise ValueError('Reversed ranking is not deployable calibration')
        parameters[str(head)]=dict(slope=slope,intercept=float(estimator.intercept_[0]),samples=len(take),positive_fraction=float(y[take].mean()))
    calibrated=[(episode,transform(p,parameters)) for episode,p in data]
    pooled=np.concatenate([p[...,5][episode.valid&alarm_support(episode,require_energy)] for episode,p in calibrated])
    thresholds=np.unique(np.r_[np.linspace(.01,.99,50),np.quantile(pooled,np.linspace(.5,.999,35))]);trials=[]
    for threshold in thresholds:
        result=combine_alerts([alert_counts(ep.links,ep.ends,p[...,5],alarm_support(ep,require_energy),ep.record['n_search'],float(threshold)) for ep,p in calibrated])
        trials.append(dict(threshold=float(threshold),**result))
    full=[row for row in trials if meets(row)]
    advisory=[row for row in trials if row['first_alerts']>=10 and row['precision'] is not None and row['precision']>=.8 and row['false_or_late_per_drone_hour']<=5]
    budget=[row for row in trials if row['first_alerts']>=50 and row['false_or_late_per_drone_hour']<=1]
    partial=[row for row in budget if row['precision'] is not None and row['precision']>=.9]
    selected=best(full) or best(advisory) or min(trials,key=lambda r:abs(r['threshold']-.5))
    value=dict(model_tag=summary['model_tag'],parameters=parameters,threshold=selected['threshold'],selected=selected,
        require_energy=bool(require_energy),enabled=bool(full),mode='simulation_goal' if full else 'research_only',
        common_FA_budget1=best(budget),precision90_FA1_partial=best(partial),trials=trials,goals=GOALS,
        source_summary=str(Path(summary_path).resolve()),source_sha256=sha(summary_path),implementation_sha256=sha(__file__),
        created_at=now(),hardware_validated=False,production_ready=False,
        note='Head5 calibration uses all states; head2 calibration is conditioned on currently-up links to match the reported 5s target. Budget1 does not assert the precision or recall goals.')
    write(out,value);return out


def apply(summary_path,policy_path):
    summary=read(summary_path);policy=read(policy_path)
    if summary['scope']=='confirmation':
        frozen=verify_freeze();item=next((item for item in frozen['models'] if item['pin']['path']==summary['model_manifest']),None)
        if item is None or sha(policy_path)!=item['policy_sha256']:raise ValueError('Confirmation policy was not frozen')
    audit_fresh(summary)
    if summary['model_tag']!=policy['model_tag']:raise ValueError('Policy/model mismatch')
    data=records(summary);counts=[];budget_counts=[];partial_counts=[];p5=[];p37=[];y5=[];y37=[]
    for ep,p in data:
        c=transform(p,policy['parameters']);support=alarm_support(ep,policy['require_energy'])
        counts.append(alert_counts(ep.links,ep.ends,c[...,5],support,ep.record['n_search'],policy['threshold']))
        if policy['common_FA_budget1']:budget_counts.append(alert_counts(ep.links,ep.ends,c[...,5],support,ep.record['n_search'],policy['common_FA_budget1']['threshold']))
        if policy['precision90_FA1_partial']:partial_counts.append(alert_counts(ep.links,ep.ends,c[...,5],support,ep.record['n_search'],policy['precision90_FA1_partial']['threshold']))
        good=ep.valid&ep.legacy_support;live=good&ep.currently_up
        p5.append(c[...,2][live]);y5.append(ep.targets[...,2][live]);p37.append(c[...,5][good]);y37.append(ep.targets[...,5][good])
    alarms=combine_alerts(counts);p5=np.concatenate(p5);p37=np.concatenate(p37);y5=np.concatenate(y5);y37=np.concatenate(y37)
    value=dict(model_tag=summary['model_tag'],scope=summary['scope'],policy_path=str(policy_path),policy_sha256=sha(policy_path),
        alarms=alarms,case_alarms=[dict(key=case['key'],region=case['region'],**counts[i]) for i,case in enumerate(summary['cases'])],
        common_FA_budget1=combine_alerts(budget_counts) if budget_counts else None,
        precision90_FA1_partial=combine_alerts(partial_counts) if partial_counts else None,
        calibrated_next5=binary(y5,p5),calibrated_any3to7=binary(y37,p37),reliability_next5=reliability(y5,p5),reliability_any3to7=reliability(y37,p37),
        calibration_goal_met=policy['enabled'],current_fold_goal_met=meets(alarms),
        trusted_simulation_alarm_enabled=bool(policy['enabled'] and meets(alarms)),production_ready=False,hardware_validated=False,created_at=now())
    out=Path(summary_path).parent/'calibrated_result.json';write(out,value);return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fit','apply']);p.add_argument('--summary',required=True);p.add_argument('--policy');a=p.parse_args()
    print(fit(a.summary) if a.command=='fit' else apply(a.summary,a.policy))
