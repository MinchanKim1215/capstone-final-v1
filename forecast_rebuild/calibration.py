"""Probability calibration and alarm fitting use calibration geography only."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from . import ARTIFACTS
from .contracts import read,write,sha,digest,GOALS,SEED,now,BASELINE,folds
from .dataset import Episode
from .metrics import alert_counts,combine_alerts,binary


def alarm_support(episode,require_energy=False):
    support=np.asarray(episode.legacy_support).copy()
    if require_energy:
        e=np.asarray(episode.energy[np.asarray(episode.ends)-1],np.float32)
        support &= (e[...,3]>.5)&(e[...,4]>.5)&np.isfinite(e[...,0])&np.isfinite(e[...,1])&np.isfinite(e[...,5])
        support &= (e[...,2]>=1)&(e[...,2]<=2)&(e[...,5]>=.5)&(e[...,5]<=1)
    return support


def audit_cohort(summary):
    if summary['scope'] not in ('selection','calibration','confirmation'):raise ValueError('Unknown development scope')
    dataset=read(summary['dataset_path']);lookup={r['key']:r for r in dataset['rows']}
    groups={}
    for case in summary['cases']:
        record=lookup[case['key']]
        if record['source_kind']!='fresh_v7':raise ValueError('Final evaluation uses new balanced flights only')
        groups.setdefault(record['region'],[]).append(record['spec']['kind'])
    if set(groups)!=set(folds()[summary['scope']]):raise ValueError('Development geography coverage is incomplete')
    if any(sorted(kinds)!=['clear','continuous','intermittent'] for kinds in groups.values()):
        raise ValueError('Each development map must include all three RF conditions exactly once')
    return dict(regions=len(groups),flights=len(summary['cases']),interference_conditions_per_map=3)


def transform(probability,parameters):
    output=np.asarray(probability,np.float32).copy()
    for head,item in parameters.items():
        h=int(head);p=output[...,h]
        output[...,h]=expit(item['slope']*logit(np.clip(p,1e-6,1-1e-6))+item['intercept'])
    return output


def records(summary):
    dataset=read(summary['dataset_path'])
    if sha(summary['dataset_path'])!=summary['dataset_sha256']:raise ValueError('Evaluated dataset changed')
    registry={r['key']:r for r in dataset['rows']}
    result=[]
    for case in summary['cases']:
        record=registry[case['key']]
        if record['role']!=summary['scope']:raise ValueError('Cross-fold probability fitting is forbidden')
        if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Predictions changed')
        with np.load(case['predictions_path'],allow_pickle=False) as z:p=z['prediction']
        result.append((Episode(record),p))
    return result


def meets(result,goals=GOALS):
    return bool(result['first_alerts']>=goals['minimum_alerts'] and result['precision'] is not None
        and result['precision']>=goals['alert_precision'] and result['event_recall_3s'] is not None
        and result['event_recall_3s']>=goals['event_recall_3s'] and result['false_or_late_per_drone_hour']<=goals['false_or_late_per_drone_hour'])


def reliability(truth,probability,bins=10):
    y=np.asarray(truth,bool);p=np.asarray(probability,float);good=np.isfinite(p);y=y[good];p=p[good]
    groups=[];error=0.
    for low,high in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        take=(p>=low)&((p<high) if high<1 else (p<=high))
        if not take.any():continue
        predicted=float(p[take].mean());observed=float(y[take].mean());size=int(take.sum())
        error+=size*abs(predicted-observed)
        groups.append(dict(lower=float(low),upper=float(high),samples=size,mean_prediction=predicted,event_fraction=observed))
    return dict(ece=error/max(1,len(y)),bins=groups,note='Equal-width reliability bins. ECE depends on the chosen binning.')


def fit(summary_path):
    summary=read(summary_path)
    if summary['scope']!='calibration':raise ValueError('Only the spatial calibration fold can set alarm rules')
    audit_cohort(summary)
    data=records(summary);parameters={};rng=np.random.default_rng(SEED+7)
    require_energy=bool(summary['model_manifest']!='baseline' and read(summary['model_manifest']).get('use_energy',False))
    for head in (2,5):
        x=[];y=[]
        for episode,p in data:
            good=episode.valid&alarm_support(episode,require_energy)&np.isfinite(p[...,head])
            x.append(p[...,head][good]);y.append(episode.targets[...,head][good])
        score=np.concatenate(x);truth=np.concatenate(y)
        selected=rng.choice(len(score),size=min(250000,len(score)),replace=False)
        model=LogisticRegression(C=1.,solver='lbfgs',max_iter=100,tol=1e-8)
        model.fit(logit(np.clip(score[selected],1e-6,1-1e-6))[:,None],truth[selected])
        slope=float(model.coef_[0,0])
        if slope<=0:raise ValueError('A reversed probability ranking cannot be deployed as calibration')
        parameters[str(head)]=dict(slope=slope,intercept=float(model.intercept_[0]),samples=len(selected),
                                   positive_fraction=float(truth[selected].mean()))
    calibrated=[(ep,transform(p,parameters)) for ep,p in data]
    sampled=np.concatenate([p[...,5][ep.valid&alarm_support(ep,require_energy)] for ep,p in calibrated])
    thresholds=np.unique(np.r_[np.linspace(.01,.99,50),np.quantile(sampled,np.linspace(.5,.999,35))])
    trials=[]
    for threshold in thresholds:
        if not 0<threshold<1:continue
        result=combine_alerts([alert_counts(ep.links,ep.ends,p[...,5],alarm_support(ep,require_energy),ep.record['n_search'],float(threshold)) for ep,p in calibrated])
        trials.append(dict(threshold=float(threshold),**result))
    feasible=[r for r in trials if meets(r)]
    partial=[r for r in trials if r['first_alerts']>=GOALS['minimum_alerts'] and r['precision'] is not None
             and r['precision']>=GOALS['alert_precision'] and r['false_or_late_per_drone_hour']<=GOALS['false_or_late_per_drone_hour']]
    # Advisory fallback is explicitly disabled for trusted automatic alarms.
    advisory=[r for r in trials if r['first_alerts']>=10 and r['precision'] is not None and r['precision']>=.8 and r['false_or_late_per_drone_hour']<=5]
    candidates=feasible or advisory
    if candidates:selected=max(candidates,key=lambda r:(r['event_recall_3s'],r['precision'],-r['false_or_late_per_drone_hour']))
    else:selected=min(trials,key=lambda r:abs(r['threshold']-.5))
    result=dict(model_tag=summary['model_tag'],parameters=parameters,threshold=selected['threshold'],require_energy=require_energy,
                enabled=bool(feasible),mode='validated_calibration_goal' if feasible else 'research_advisory_only',
                selected=selected,goals=GOALS,trials=trials,source_summary=str(Path(summary_path).resolve()),
                high_precision_partial_point=max(partial,key=lambda r:r['event_recall_3s']) if partial else None,
                partial_point_note='Secondary calibration-only operating point: precision/false-alarm goals without the recall goal. Not full acceptance and never automatically enabled.',
                source_sha256=sha(summary_path),created_at=now(),confirmation_used=False,hardware_validated=False,
                note='A fallback threshold does not satisfy the trusted-alarm requirements and must not be enabled automatically.')
    path=Path(summary_path).parent/'alarm_policy.json';write(path,result)
    return path


def apply(summary_path,policy_path):
    summary=read(summary_path);policy=read(policy_path)
    if summary['scope']=='confirmation':verify_freeze()
    audit_cohort(summary)
    if summary['model_tag']!=policy['model_tag']:raise ValueError('Alarm policy belongs to another model')
    data=records(summary);counts=[];partial_counts=[];p5=[];p37=[];y5=[];y37=[]
    for ep,p in data:
        calibrated=transform(p,policy['parameters'])
        counts.append(alert_counts(ep.links,ep.ends,calibrated[...,5],alarm_support(ep,policy.get('require_energy',False)),ep.record['n_search'],policy['threshold']))
        if policy.get('high_precision_partial_point'):
            partial_counts.append(alert_counts(ep.links,ep.ends,calibrated[...,5],alarm_support(ep,policy.get('require_energy',False)),
                                               ep.record['n_search'],policy['high_precision_partial_point']['threshold']))
        support=ep.valid&ep.legacy_support;live=support&ep.currently_up
        p5.append(calibrated[...,2][live]);y5.append(ep.targets[...,2][live])
        p37.append(calibrated[...,5][support]);y37.append(ep.targets[...,5][support])
    outcome=combine_alerts(counts)
    p5=np.concatenate(p5);p37=np.concatenate(p37);y5=np.concatenate(y5);y37=np.concatenate(y37)
    result=dict(model_tag=summary['model_tag'],scope=summary['scope'],policy_path=str(policy_path),
                policy_sha256=sha(policy_path),alarms=outcome,calibration_goal_met=policy['enabled'],
                case_alarms=[dict(key=case['key'],region=case['region'],**counts[i]) for i,case in enumerate(summary['cases'])],
                high_precision_partial_operating_point=combine_alerts(partial_counts) if partial_counts else None,
                current_fold_goal_met=meets(outcome),trusted_simulation_alarm_enabled=bool(policy['enabled'] and meets(outcome)),
                calibrated_next5_on_common_currently_up_links=binary(y5,p5),
                calibrated_any3to7_on_common_support=binary(y37,p37),
                reliability_next5=reliability(y5,p5),reliability_any3to7=reliability(y37,p37),
                production_ready=False,hardware_validated=False,created_at=now())
    path=Path(summary_path).parent/'calibrated_result.json';write(path,result)
    return path


def freeze(selected_summary,selected_policy,baseline_policy):
    selected=read(selected_summary);policy=read(selected_policy);baseline=read(baseline_policy)
    if selected['scope']!='selection':raise ValueError('Choose a model using selection only')
    audit_cohort(selected)
    if policy['model_tag']!=selected['model_tag']:raise ValueError('Selected model/alarm identity mismatch')
    from . import ROOT
    sources={name:sha(ROOT/name) for name in ('__init__.py','features.py','targets.py','models.py','models_dense.py','inference.py','dataset.py','metrics.py','calibration.py','runtime.py','planning.py','online_planning.py','evaluate.py','prepare_data.py','prepare_plans.py','sensors.py','contracts.py','beacon_codec.py')}
    value=dict(created_at=now(),selected_model=selected['model_manifest'],selected_tag=selected['model_tag'],reference_baseline_sha256=sha(BASELINE),
               cohort_policy_sha256=sha(ROOT/'DATA_COHORT_CORRECTION.md'),
               selected_model_sha256=sha(selected['model_manifest']) if selected['model_manifest']!='baseline' else None,
               source_sha256=sources,
               selection_summary=str(Path(selected_summary).resolve()),selection_sha256=sha(selected_summary),
               selected_alarm=str(Path(selected_policy).resolve()),selected_alarm_sha256=sha(selected_policy),
               baseline_alarm=str(Path(baseline_policy).resolve()),baseline_alarm_sha256=sha(baseline_policy),
               confirmation_opened=False,final_test_opened=False,production_ready=False,
               calibration_enabled=policy['enabled'],baseline_calibration_enabled=baseline['enabled'])
    path=ARTIFACTS/'model_alarm_freeze.json'
    if path.exists():
        old=read(path)
        for key in ('selected_model','selection_sha256','selected_alarm_sha256','baseline_alarm_sha256'):
            if old[key]!=value[key]:raise ValueError('A frozen confirmation model cannot be changed after the fact')
        return path
    write(path,value);return path


def verify_freeze():
    from . import ROOT
    value=read(ARTIFACTS/'model_alarm_freeze.json')
    if sha(ROOT/'DATA_COHORT_CORRECTION.md')!=value['cohort_policy_sha256']:raise ValueError('Frozen cohort policy changed')
    if sha(BASELINE)!=value['reference_baseline_sha256']:raise ValueError('Reference baseline weights changed')
    for name,expected in value['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen confirmation implementation changed: '+name)
    for path_key,hash_key in [('selection_summary','selection_sha256'),('selected_alarm','selected_alarm_sha256'),('baseline_alarm','baseline_alarm_sha256')]:
        if sha(value[path_key])!=value[hash_key]:raise ValueError('Frozen model/alarm evidence changed')
    for key in ('selected_alarm','baseline_alarm'):
        policy=read(value[key])
        if sha(policy['source_summary'])!=policy['source_sha256']:raise ValueError('Calibration evidence changed')
    if value['selected_model']!='baseline' and sha(value['selected_model'])!=value['selected_model_sha256']:
        raise ValueError('Frozen model manifest changed')
    marker=ARTIFACTS/'confirmation_access.json'
    if marker.exists() and read(marker)['freeze_sha256']!=sha(ARTIFACTS/'model_alarm_freeze.json'):
        raise ValueError('Confirmation belongs to another frozen comparison')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fit','apply','freeze']);p.add_argument('--summary')
    p.add_argument('--policy');p.add_argument('--baseline-policy');a=p.parse_args()
    if a.command=='fit':out=fit(a.summary)
    elif a.command=='apply':out=apply(a.summary,a.policy)
    else:out=freeze(a.summary,a.policy,a.baseline_policy)
    print(out)
