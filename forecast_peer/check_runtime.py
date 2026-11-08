"""Verify the actual frozen default API on a complete selection flight."""
import time
import numpy as np
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now
from .runtime import load_service
from .evaluate import case_prediction
from forecast_recall.episodes import cohort
from .inference import baseline
from .features import sequence
from .data import inputs as data_inputs
from forecast_graph.episodes import physical
from forecast_graph import ARTIFACTS as GRAPH
from forecast_navigation.wire import extend_trace
from forecast_navigation.check_runtime import expected_starts
from forecast_rebuild.planning import public_context
from forecast_rebuild.calibration import transform


def main():
    b=read(ARTIFACTS/'research_candidate.json');item=next(i for i in read(ARTIFACTS/'data_manifest.json')['records'] if i['role']=='selection')
    r=item['source_record'];ep=physical(r);n=ep.values.shape[2];ns=r['n_search'];city,nominal=public_context(r);city.radio.enable_cache()
    stream=load_service(city,n,ns,nominal);p,_=case_prediction(b['candidate'],item,'selection');policy=read(b['policy']);target=cohort(item)
    wanted=transform(p,policy['parameters']);alerts=expected_starts(target,p,policy);lookup={int(e):t for t,e in enumerate(target.ends)};old=baseline(item,'selection')
    reference=data_inputs(item)
    with np.load(GRAPH/'traces'/(r['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n]
    with np.load(ROOT.parent/'forecast_navigation/artifacts/deltas'/(r['key']+'.npz'),allow_pickle=False) as a:tx=extend_trace(tx,a['delta'])
    errors=dict(peer_history=0.,history=0.,embedding=0.,logits=0.,baseline_raw=0.,probability=0.);tested=0;mismatches=0;latencies=[]
    for q in range(len(ep.values)):
        received=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else [];own=tx[q//2,ns] if q%2==0 else None
        tick=time.perf_counter();out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],received,own)
        errors['history']=max(errors['history'],float(np.abs(stream.parent.history[-1].astype(float)-reference['history'][q].astype(float)).max()))
        if (q+1)%2==0:errors['peer_history']=max(errors['peer_history'],float(np.abs(stream.peer_history[-1].astype(float)-reference['peer_history'][q//2].astype(float)).max()))
        if out['status']!='ready':continue
        latencies.append((time.perf_counter()-tick)*1000);end=q+1;assert out['trusted_alarm_enabled'] is False
        if end not in lookup:continue
        t=lookup[end]
        for key in ('embedding','logits'):errors[key]=max(errors[key],float(np.abs(stream.last_inputs[key]-reference[key][t:t+1]).max()))
        errors['baseline_raw']=max(errors['baseline_raw'],float(np.abs(stream.baseline_raw[0,[2,5]]-old[t,0,0,[2,5]]).max()))
        for field,head in [('risk_5s',2),('risk_3to7s',5)]:
            if out[field] is not None:
                errors['probability']=max(errors['probability'],abs(out[field]-float(wanted[t,0,0,head])));tested+=1
        if end in alerts:mismatches+=int(out['qualified_first_alert']!=alerts[end][0,0])
        if end%500==0:print('runtime frames',end,'/',len(ep.values),'error',errors['probability'],flush=True)
    write(ARTIFACTS/'runtime_diagnostic.json',dict(created_at=now(),errors=errors,first_alert_mismatches=mismatches))
    if errors['history']!=0 or errors['peer_history']!=0 or any(v>1e-4 for v in errors.values()):raise ValueError('Runtime numerical/input discrepancy: '+str(errors))
    if mismatches or tested==0:raise ValueError('First-alert mismatch or no supported predictions')
    q=stream.local.frames
    for k in range(2):out=stream.observe((q+k+1)*.5,np.zeros((n,48)),np.zeros((n,48),bool),np.zeros((n,9)))
    missing=out['risk_5s'] is None and out['risk_3to7s'] is None;assert missing
    value=dict(passed=True,created_at=now(),source=r['key'],frames=len(ep.values),tested_predictions=tested,errors=errors,first_alert_mismatches=mismatches,
        missing_RF_returns_None=missing,trusted_alarm_disabled=True,numerical_tolerance=1e-4,embedded_latency_certified=False,
        cpu_ms=dict(cold=latencies[0],warm_median=float(np.median(latencies[1:])),warm_max=max(latencies[1:])),
        export_sha256=sha(ARTIFACTS/'research_candidate.json'),runtime_sha256=sha(ROOT/'runtime.py'),checker_sha256=sha(ROOT/'check_runtime.py'),freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'runtime_checks.json',value);print(value,flush=True)


if __name__=='__main__':main()
