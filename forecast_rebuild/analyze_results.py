"""Frozen comparison diagnostics and paired geography-cluster uncertainty."""
from collections import defaultdict
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,maps,now,sha
from .dataset import Episode
from .metrics import combine_alerts
from .diagnose import observability,combine as combine_observability,severity
from .calibration import transform


def macro_group(cases):
    ap5=[r['next5']['ap'] for r in cases if r['next5']['ap'] is not None]
    ap37=[r['any3to7']['ap'] for r in cases if r['any3to7']['ap'] is not None]
    return dict(flights=len(cases),regions=len({r['region'] for r in cases}),
        macro_flight_ap5=float(np.mean(ap5)) if ap5 else None,
        macro_flight_ap37=float(np.mean(ap37)) if ap37 else None,
        ap5_defined_flights=len(ap5),ap37_defined_flights=len(ap37),
        alarms=combine_alerts([r['alerts'] for r in cases]),
        note='Group AP is a mean over flights with at least one positive, not pooled AP. Zero-positive flights have undefined AP and are counted separately, not assigned zero.')


def paired_uncertainty(old,new,old_cal,new_cal,draws=10000):
    regions=sorted({r['region'] for r in old['cases']});registry=maps()
    groups=sorted({registry[region]['joint_spatial_group'] for region in regions});n=len(groups)
    rng=np.random.default_rng(2026110207)
    counts=rng.multinomial(n,np.full(n,1/n),size=draws)
    samples={}
    for field,label in [('next5','macro_spatial_group_ap5_difference'),('any3to7','macro_spatial_group_ap37_difference')]:
        a=[];b=[]
        for group in groups:
            va=[r[field]['ap'] for r in old['cases'] if registry[r['region']]['joint_spatial_group']==group and r[field]['ap'] is not None]
            vb=[r[field]['ap'] for r in new['cases'] if registry[r['region']]['joint_spatial_group']==group and r[field]['ap'] is not None]
            a.append(np.mean(va) if va else np.nan);b.append(np.mean(vb) if vb else np.nan)
        a=np.asarray(a);b=np.asarray(b);good=np.isfinite(a)&np.isfinite(b);m=int(good.sum())
        if not m:
            samples[label]=dict(point=None,percentile95=None,defined_spatial_groups=0);continue
        difference=(b-a)[good];ap_counts=rng.multinomial(m,np.full(m,1/m),size=draws);values=(ap_counts@difference)/m
        samples[label]=dict(point=float(difference.mean()),percentile95=[float(x) for x in np.quantile(values,[.025,.975])],defined_spatial_groups=m)
    keys=('first_alerts','timely_correct_first_alerts','eligible_directed_link_events','event_hits_3s','drone_hours')
    def alarm_bootstrap(source):
        matrix=np.array([[sum(r[k] for r in source['case_alarms'] if registry[r['region']]['joint_spatial_group']==group) for k in keys] for group in groups])
        sums=counts@matrix;first,correct,events,hits,hours=sums.T
        return dict(precision=np.divide(correct,first,out=np.full_like(first,np.nan),where=first>0),
            recall=hits/events,false_or_late_per_drone_hour=(first-correct)/hours)
    a=alarm_bootstrap(old_cal);b=alarm_bootstrap(new_cal)
    for metric in a:
        values=b[metric]-a[metric]
        samples['calibrated_'+metric+'_difference']=dict(percentile95=[float(x) for x in np.nanquantile(values,[.025,.975])])
    return dict(geography_clusters=n,regions=len(regions),bootstrap_draws=draws,paired=True,metrics=samples,
        note='Percentile resampling over shared spatial groups with all member maps and RF conditions together. This remains a limited generalization sample; not independent directed-link confidence.')


def main():
    completed=read(ARTIFACTS/'completed_comparison.json');registry=maps()
    old=read(completed['baseline_confirmation']);new=read(completed['selected_confirmation'])
    old_cal=read(completed['baseline_calibrated_confirmation']);new_cal=read(completed['selected_calibrated_confirmation'])
    records={r['key']:r for r in read(new['dataset_path'])['rows']}
    groups={}
    for label,selector in [('legacy_geographies',lambda r:registry[r['region']]['source_dataset']=='legacy_v6'),
                           ('added_geographies',lambda r:registry[r['region']]['source_dataset']!='legacy_v6')]:
        groups[label]=dict(baseline=macro_group([r for r in old['cases'] if selector(r)]),selected=macro_group([r for r in new['cases'] if selector(r)]))
    for condition in ('clear','intermittent','continuous'):
        groups[condition]=dict(baseline=macro_group([r for r in old['cases'] if records[r['key']]['spec']['kind']==condition]),
                               selected=macro_group([r for r in new['cases'] if records[r['key']]['spec']['kind']==condition]))
    observed=[];cases=[];severity_rows={'baseline':[],'selected':[]}
    policies={label:read(completed[label+'_alarm']) for label in ('baseline','selected')}
    old_lookup={r['key']:r for r in old['cases']}
    for case in new['cases']:
        record=records[case['key']];episode=Episode(record);outcome=observability(episode);observed.append(outcome)
        cases.append(dict(key=record['key'],region=record['region'],condition=record['spec']['kind'],**outcome))
        for label,prediction_case in [('baseline',old_lookup[case['key']]),('selected',case)]:
            policy=policies[label]
            with np.load(prediction_case['predictions_path'],allow_pickle=False) as z:prediction=z['prediction']
            score=transform(prediction,policy['parameters'])[...,5]
            severity_rows[label].append(severity(episode,score,policy['threshold'],policy.get('require_energy',False)))
    severity_summary={}
    for label,rows in severity_rows.items():
        severity_summary[label]={}
        for group in rows[0]:
            counts={key:sum(row[group][key] for row in rows) for key in rows[0][group]}
            severity_summary[label][group]=dict(**counts,first_alert_recall_3s=counts['first_alert_hits_3s']/max(1,counts['events']),
                persistent_risk_coverage_3s=counts['any_high_risk_before_3s']/max(1,counts['events']))
    uncertainty=paired_uncertainty(old,new,old_cal,new_cal)
    result=dict(created_at=now(),selected_model_tag=completed['selected_tag'],selected_model=completed['selected_model'],
        comparison_sha256=sha(ARTIFACTS/'completed_comparison.json'),
        baseline=dict(raw=old,calibrated=old_cal),selected=dict(raw=new,calibrated=new_cal),
        groups=groups,paired_uncertainty=uncertainty,event_observability=combine_observability(observed),observability_cases=cases,
        calibrated_event_severity=severity_summary,severity_note='Same frozen thresholds. Persistent high risk is diagnostic coverage, not a new first alert or a replacement for strict event recall.',
        production_ready=False,hardware_validated=False,original_final_test_opened=False)
    write(ARTIFACTS/'final_results.json',result)
    print(ARTIFACTS/'final_results.json')


if __name__=='__main__':main()
