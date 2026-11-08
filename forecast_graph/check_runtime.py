"""Actual exported live APIs versus batch probabilities and strict alarm starts."""
import time
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .runtime import load_link,load_service
from .episodes import cohort
from .evaluate import physical,case_prediction
from .features import LocalHistory,build
from forecast_rebuild.calibration import transform,alarm_support
from forecast_rebuild.planning import public_context


def expected_starts(target,probability,policy):
    score=transform(probability,policy['parameters'])[...,5];support=alarm_support(target,policy['require_energy'])
    active=np.zeros(score.shape[1:],bool);starts={}
    for t,end in enumerate(target.ends):
        if end*.5<21 or end*.5>len(target.links)*.5-8.5:continue
        new=support[t]&np.isfinite(score[t])&(np.asarray(score[t],np.float64)>=np.where(active,policy['threshold']*.5,policy['threshold']))
        starts[int(end)]=new&~active;active=new
    return starts


def main():
    bundle=read(ARTIFACTS/'research_candidates.json')
    item=next(i for i in read(ARTIFACTS/'data_manifest.json')['records'] if i['role']=='selection')
    record=item['source_record'];ep=physical(record);city,nominal=public_context(record);n=ep.values.shape[2];ns=record['n_search']
    city.radio.enable_cache();streams={task:loader(city,n,ns,nominal=nominal) for task,loader in [('link',load_link),('service',load_service)]}
    raw={};expected={};starts={};index={int(e):t for t,e in enumerate(ep.ends)}
    for task in streams:
        p,_=case_prediction(bundle['tasks'][task]['candidate'],item,ep,'selection');raw[task]=p
        policy=read(bundle['tasks'][task]['policy']);target=cohort(record,task,ep)
        expected[task]=transform(p,policy['parameters']);starts[task]=expected_starts(target,p,policy)
    with np.load(ARTIFACTS/'traces'/(record['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n]
    with np.load(item['path'],allow_pickle=False) as a:
        feature_reference={k:a[k] for k in ('node_local','node_shared','edge_local','edge_shared','physics','global')}
        queries=a['query']
    query_lookup={int(ep.ends[t]):i for i,(t,r) in enumerate(queries)}
    feature_error={k:0. for k in feature_reference};feature_checks=0;worst_frames={task:None for task in streams}
    maximum={task:0. for task in streams};tested={task:0 for task in streams};latency={task:[] for task in streams};mismatches=0
    for q in range(len(ep.values)):
        outgoing=tx[q//2,ns] if q%2==0 else None
        received=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else []
        for task,stream in streams.items():
            tick=time.perf_counter();out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],received,outgoing)
            if out['status']!='ready':continue
            latency[task].append((time.perf_counter()-tick)*1000);end=q+1
            if end not in index:continue
            t=index[end]
            if task=='link':
                for row in out['links']:
                    j=row['peer']
                    for field,h in [('risk_5s',2),('risk_3to7s',5)]:
                        if row[field] is not None and np.isfinite(expected[task][t,0,j,h]):
                            error=abs(row[field]-float(expected[task][t,0,j,h]))
                            if error>maximum[task]:maximum[task]=error;worst_frames[task]=end
                            tested[task]+=1
                    wanted=starts[task].get(end,np.zeros((1,n),bool))[0,j]
                    if end in starts[task]:mismatches+=int(row['qualified_first_alert']!=wanted)
                    if end*.5<21:assert row['qualified_first_alert'] is False
                    assert row['trusted_alarm'] is False
            else:
                for field,h in [('risk_5s',2),('risk_3to7s',5)]:
                    if out[field] is not None and np.isfinite(expected[task][t,0,0,h]):
                        error=abs(out[field]-float(expected[task][t,0,0,h]))
                        if error>maximum[task]:maximum[task]=error;worst_frames[task]=end
                        tested[task]+=1
                wanted=starts[task].get(end,np.zeros((1,1),bool))[0,0]
                if end in starts[task]:mismatches+=int(out['qualified_first_alert']!=wanted)
                if end*.5<21:assert out['qualified_first_alert'] is False
                assert out['trusted_alarm'] is False
            #The deployed float32CPU and cached float32GPU execute different
            #GRU/linear kernels. Verify exact inputs at every larger discrepancy;
            #never change a risk, threshold, candidate or cached evaluation.
            if end in query_lookup and (maximum[task]>2e-5 or end%100==0):
                actual=build(LocalHistory(stream.local.values,stream.local.masks,stream.local.intents,n,ns),stream.received,[[end,0]],city)
                qi=query_lookup[end]
                for key,value in actual.items():feature_error[key]=max(feature_error[key],float(np.max(np.abs(value.astype(float)-feature_reference[key][qi:qi+1].astype(float)))))
                feature_checks+=1
    diagnostic=dict(created_at=now(),maximum_probability_error=maximum,worst_frames=worst_frames,
        graph_feature_error=feature_error,feature_checks=feature_checks,first_alert_mismatches=mismatches)
    write(ARTIFACTS/'runtime_numerical_diagnostic.json',diagnostic)
    if any(feature_error.values()):raise ValueError('Actual graph inputs diverged: '+str(feature_error))
    if any(v>1e-4 for v in maximum.values()):raise ValueError('Float32CPU/GPU probability discrepancy exceeds1e-4: '+str(maximum))
    if mismatches:raise ValueError('Live first-alert mismatch count: '+str(mismatches))
    if not all(tested.values()):raise ValueError('Actual exported probabilities were not exercised')
    missing={}
    for task,stream in streams.items():
        q=stream.local.frames;out=None
        for k in range(2):
            out=stream.observe((q+k+1)*.5,np.zeros((n,48)),np.zeros((n,48),bool),np.zeros((n,9)))
        missing[task]=all(r['risk_5s'] is None and r['risk_3to7s'] is None for r in out['links']) if task=='link' else out['risk_5s'] is None and out['risk_3to7s'] is None
    assert all(missing.values())
    value=dict(passed=True,created_at=now(),source=record['key'],frames=len(ep.values),tested_predictions=tested,
        max_abs_probability_error=maximum,first_alert_mismatches=mismatches,missing_RF_returns_None=missing,
        numerical_tolerance=1e-4,graph_feature_error=feature_error,graph_feature_checks=feature_checks,
        numerical_note='Float32CPU live versus frozen float32GPU batch. Verified exact graph inputs; probabilities and policies were not altered.',
        trusted_alarm_disabled=True,alarm_start_s=21,export_sha256=sha(ARTIFACTS/'research_candidates.json'),
        runtime_sha256=sha(ROOT/'runtime.py'),checker_sha256=sha(ROOT/'check_runtime.py'),freeze_sha256=sha(ARTIFACTS/'freeze.json'),
        cpu_ms={task:dict(cold=v[0],warm_median=float(np.median(v[1:])),warm_max=max(v[1:])) for task,v in latency.items()},
        embedded_latency_certified=False)
    write(ARTIFACTS/'runtime_checks.json',value);print(value)


if __name__=='__main__':main()
