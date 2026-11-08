"""Full-flight actual exported API, all20snapshots and first-alert equivalence."""
import time
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .runtime import load_service
from .evaluate import case_prediction
from forecast_recall.episodes import cohort
from forecast_graph.episodes import physical
from forecast_navigation.wire import extend_trace
from forecast_navigation.check_runtime import expected_starts
from forecast_rebuild.planning import public_context
from forecast_rebuild.calibration import transform

def main():
    b=read(ARTIFACTS/'research_candidate.json');item=next(i for i in read(ARTIFACTS/'data_manifest.json')['records'] if i['role']=='selection')
    r=item['source_record'];ep=physical(r);ns=r['n_search'];n=ep.values.shape[2];city,nominal=public_context(r);city.radio.enable_cache();stream=load_service(city,n,ns,nominal)
    p,_=case_prediction(b['candidate'],item,'selection');policy=read(b['policy']);target=cohort(item);wanted=transform(p,policy['parameters']);alerts=expected_starts(target,p,policy);lookup={int(e):i for i,e in enumerate(target.ends)}
    with np.load(item['path']) as a:reference={k:a[k] for k in ('node','edge','index','physics','global_features')}
    with np.load(item['trace_path']) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n]
    with np.load(item['delta_path']) as a:tx=extend_trace(tx,a['delta'])
    errors={k:0. for k in ('node','edge','physics','global_features','probability')};tested=0;mismatch=0;latencies=[]
    for q in range(len(ep.values)):
        packets=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else [];own=tx[q//2,ns] if q%2==0 else None
        tick=time.perf_counter();out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],packets,own)
        if out['status']!='ready':continue
        latencies.append((time.perf_counter()-tick)*1000);end=q+1
        if end not in lookup:continue
        i=lookup[end]
        for key in ('node','edge'):
            value=reference[key][reference['index'][i]];errors[key]=max(errors[key],float(np.abs(stream.last_inputs[key].astype(float)-value.astype(float)).max()))
        for key in ('physics','global_features'):errors[key]=max(errors[key],float(np.abs(stream.last_inputs[key].astype(float)-reference[key][i:i+1].astype(float)).max()))
        for field,h in [('risk_5s',2),('risk_3to7s',5)]:
            if out[field] is not None:errors['probability']=max(errors['probability'],abs(out[field]-float(wanted[i,0,0,h])));tested+=1
        if end in alerts:mismatch+=int(out['qualified_first_alert']!=alerts[end][0,0])
        assert out['trusted_alarm_enabled'] is False
        if end%500==0:print('actual runtime',end,'/',len(ep.values),errors,flush=True)
    write(ARTIFACTS/'runtime_diagnostic.json',dict(created_at=now(),errors=errors,first_alert_mismatches=mismatch))
    if any(v for k,v in errors.items() if k!='probability') or errors['probability']>1e-4 or mismatch or tested==0:raise ValueError(str(errors)+' first-alert '+str(mismatch))
    end=stream.local.frames
    for k in range(2):out=stream.observe((end+k+1)*.5,np.zeros((n,48)),np.zeros((n,48),bool),np.zeros((n,9)))
    missing=out['risk_5s'] is None and out['risk_3to7s'] is None
    if not missing:raise ValueError('Missing RF must not fabricate risk')
    value=dict(passed=True,created_at=now(),source=r['key'],frames=len(ep.values),tested_predictions=tested,errors=errors,first_alert_mismatches=mismatch,missing_RF_returns_None=missing,
        trusted_alarm_disabled=True,numerical_tolerance=1e-4,embedded_latency_certified=False,cpu_ms=dict(cold=latencies[0],warm_median=float(np.median(latencies[1:])),warm_max=max(latencies[1:])),
        export_sha256=sha(ARTIFACTS/'research_candidate.json'),runtime_sha256=sha(ROOT/'runtime.py'),checker_sha256=sha(ROOT/'check_runtime.py'),freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'runtime_checks.json',value);print(value,flush=True)

if __name__=='__main__':main()
