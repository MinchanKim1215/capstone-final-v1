"""Full actual exported APIs, exact two-view inputs, strict first alarms."""
import time
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,PARENT
from .runtime import load_link,load_service
from .episodes import cohort
from .evaluate import case_prediction
from .data import arrays,INPUT_KEYS
from .wire import extend_trace
from forecast_graph.episodes import physical
from forecast_graph.check_runtime import expected_starts
from forecast_rebuild.calibration import transform
from forecast_rebuild.planning import public_context


def main():
    bundle=read(ARTIFACTS/'research_candidates.json')
    item=next(i for i in read(ARTIFACTS/'data_manifest.json')['records'] if i['role']=='selection')
    record=item['source_record'];ep=physical(record);city,nominal=public_context(record)
    n=ep.values.shape[2];ns=record['n_search'];city.radio.enable_cache()
    streams={task:loader(city,n,ns,nominal=nominal) for task,loader in [('link',load_link),('service',load_service)]}
    expected={};starts={};lookup={int(e):t for t,e in enumerate(cohort(item,'link').ends)}
    references={profile:arrays(item,profile) for profile in ('original','mission')}
    for task in streams:
        p,_=case_prediction(bundle['tasks'][task]['candidate'],item,'selection');policy=read(bundle['tasks'][task]['policy'])
        expected[task]=transform(p,policy['parameters']);starts[task]=expected_starts(cohort(item,task),p,policy)
    with np.load(PARENT/'traces'/(record['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n]
    with np.load(ARTIFACTS/'deltas'/(record['key']+'.npz'),allow_pickle=False) as a:tx=extend_trace(tx,a['delta'])
    maximum={task:0. for task in streams};tested={task:0 for task in streams};latency={task:[] for task in streams}
    feature_error={profile:{key:0. for key in INPUT_KEYS} for profile in references};checks=0;mismatches=0
    for q in range(len(ep.values)):
        outgoing=tx[q//2,ns] if q%2==0 else None
        received=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else []
        for task,stream in streams.items():
            tick=time.perf_counter();out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],received,outgoing)
            if out['status']!='ready':continue
            latency[task].append((time.perf_counter()-tick)*1000);end=q+1
            assert out['trusted_alarm_enabled'] is False
            if end not in lookup:continue
            t=lookup[end]
            for profile in references:
                for key in INPUT_KEYS:
                    a=stream.last_inputs[profile][key].astype(float);b=references[profile][key][t:t+1].astype(float)
                    feature_error[profile][key]=max(feature_error[profile][key],float(np.max(np.abs(a-b))))
            checks+=1
            if task=='link':
                for row in out['links']:
                    j=row['peer']
                    for field,h in [('risk_5s',2),('risk_3to7s',5)]:
                        if row[field] is not None and np.isfinite(expected[task][t,0,j,h]):
                            maximum[task]=max(maximum[task],abs(row[field]-float(expected[task][t,0,j,h])));tested[task]+=1
                    if end in starts[task]:mismatches+=int(row['qualified_first_alert']!=starts[task][end][0,j])
                    assert row['trusted_alarm'] is False
            else:
                for field,h in [('risk_5s',2),('risk_3to7s',5)]:
                    if out[field] is not None and np.isfinite(expected[task][t,0,0,h]):
                        maximum[task]=max(maximum[task],abs(out[field]-float(expected[task][t,0,0,h])));tested[task]+=1
                if end in starts[task]:mismatches+=int(out['qualified_first_alert']!=starts[task][end][0,0])
                assert out['trusted_alarm'] is False
    diagnostic=dict(created_at=now(),maximum_probability_error=maximum,feature_error=feature_error,first_alert_mismatches=mismatches)
    write(ARTIFACTS/'runtime_diagnostic.json',diagnostic)
    if any(v for d in feature_error.values() for v in d.values()):raise ValueError('Runtime input differs: '+str(feature_error))
    if any(v>1e-4 for v in maximum.values()):raise ValueError('CPU/GPU float32 difference exceeds1e-4: '+str(maximum))
    if mismatches or not all(tested.values()):raise ValueError('First alarms differ or probabilities untested')
    missing={}
    for task,stream in streams.items():
        q=stream.local.frames
        for k in range(2):out=stream.observe((q+k+1)*.5,np.zeros((n,48)),np.zeros((n,48),bool),np.zeros((n,9)))
        missing[task]=all(r['risk_5s'] is None and r['risk_3to7s'] is None for r in out['links']) if task=='link' else out['risk_5s'] is None and out['risk_3to7s'] is None
    assert all(missing.values())
    value=dict(passed=True,created_at=now(),source=record['key'],frames=len(ep.values),tested_predictions=tested,
        max_abs_probability_error=maximum,first_alert_mismatches=mismatches,feature_error=feature_error,feature_checks=checks,
        missing_RF_returns_None=missing,numerical_tolerance=1e-4,trusted_alarm_disabled=True,embedded_latency_certified=False,
        export_sha256=sha(ARTIFACTS/'research_candidates.json'),runtime_sha256=sha(ROOT/'runtime.py'),checker_sha256=sha(ROOT/'check_runtime.py'),freeze_sha256=sha(ARTIFACTS/'freeze.json'),
        cpu_ms={task:dict(cold=v[0],warm_median=float(np.median(v[1:])),warm_max=max(v[1:])) for task,v in latency.items()})
    write(ARTIFACTS/'runtime_checks.json',value);print(value,flush=True)


if __name__=='__main__':main()
