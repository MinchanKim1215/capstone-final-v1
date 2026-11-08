"""Full-flight real API: observed network state, restored RF and exact first alerts."""
import time
import numpy as np
import torch
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .runtime import load_service
from .evaluate import case_prediction
from .inputs import load
from forecast_recall.episodes import cohort
from forecast_graph.episodes import physical
from forecast_navigation.check_runtime import expected_starts
from forecast_rebuild.planning import public_context
from forecast_rebuild.calibration import transform

def main():
    torch.set_num_threads(2);b=read(ARTIFACTS/'research_candidate.json');item=next(i for i in read(ARTIFACTS/'data_manifest.json')['records'] if i['role']=='selection')
    r=item['source_record'];ep=physical(r);ns=r['n_search'];n=ep.values.shape[2];city,nominal=public_context(r);city.radio.enable_cache();stream=load_service(city,n,ns,nominal)
    p,_=case_prediction(b['candidate'],item,'selection');policy=read(b['policy']);target=cohort(item);wanted=transform(p,policy['parameters']);alerts=expected_starts(target,p,policy);lookup={int(e):i for i,e in enumerate(target.ends)};reference=load(item)
    with np.load(item['replay_path']) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n];observed=a['observed']
    errors={k:0. for k in ('network_history','network','physics','base_RF','probability')};tested=0;mismatch=0;latencies=[]
    for q in range(len(ep.values)):
        packets=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else [];own=tx[q//2,ns] if q%2==0 else None
        tick=time.perf_counter();out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],packets,own,observed[q,ns])
        history=np.asarray(stream.summaries[-1],np.float16);ref=reference['network_history'][q];np.testing.assert_equal(np.isnan(history),np.isnan(ref));errors['network_history']=max(errors['network_history'],float(np.nanmax(np.abs(history.astype(float)-ref.astype(float)),initial=0)))
        if out['status']!='ready':continue
        latencies.append((time.perf_counter()-tick)*1000);end=q+1
        if end not in lookup:continue
        i=lookup[end]
        for key in ('network','physics'):
            a=stream.last_inputs[key];z=reference[key][i:i+1];np.testing.assert_equal(np.isnan(a),np.isnan(z));errors[key]=max(errors[key],float(np.nanmax(np.abs(a-z),initial=0)))
        a=stream.last_inputs['base'][:,:432];z=reference['base'][i:i+1,:432];np.testing.assert_equal(np.isnan(a),np.isnan(z));errors['base_RF']=max(errors['base_RF'],float(np.nanmax(np.abs(a-z),initial=0)))
        for field,h in [('risk_5s',2),('risk_3to7s',5)]:
            if out[field] is not None:errors['probability']=max(errors['probability'],abs(out[field]-float(wanted[i,0,0,h])));tested+=1
        if end in alerts:mismatch+=int(out['qualified_first_alert']!=alerts[end][0,0])
        assert out['trusted_alarm_enabled'] is False
        if end%500==0:print('actual runtime',end,'/',len(ep.values),errors,flush=True)
    write(ARTIFACTS/'runtime_diagnostic.json',dict(created_at=now(),errors=errors,first_alert_mismatches=mismatch))
    if any(v for k,v in errors.items() if k!='probability') or errors['probability']>1e-4 or mismatch or tested==0:raise ValueError(str(errors)+' first-alert '+str(mismatch))
    end=stream.local.frames
    for k in range(2):out=stream.observe((end+k+1)*.5,np.zeros((n,48)),np.zeros((n,48),bool),np.zeros((n,9)),local_network=np.zeros(27))
    missing=out.get('risk_5s') is None and out.get('risk_3to7s') is None
    if not missing:raise ValueError('Missing RF must not fabricate risk')
    network_missing=None
    if stream.requires_network:
        end=stream.local.frames
        for k in range(2):out=stream.observe((end+k+1)*.5,ep.values[-1,0],ep.masks[-1,0],ep.intents[-1,0],ep.energy[-1,0])
        network_missing=out['status']=='missing_local_packet_state' and out['risk_3to7s'] is None and not out['qualified_first_alert']
        if not network_missing:raise ValueError('Missing local measurement fabricated risk')
    value=dict(passed=True,created_at=now(),source=r['key'],frames=len(ep.values),tested_predictions=tested,errors=errors,first_alert_mismatches=mismatch,missing_RF_returns_None=missing,missing_required_network_returns_unknown=network_missing,
        diagnostic_truth_not_API_input=True,trusted_alarm_disabled=True,numerical_tolerance=1e-4,embedded_latency_certified=False,cpu_ms=dict(cold=latencies[0],warm_median=float(np.median(latencies[1:])),warm_max=max(latencies[1:])),
        export_sha256=sha(ARTIFACTS/'research_candidate.json'),runtime_sha256=sha(ROOT/'runtime.py'),checker_sha256=sha(ROOT/'check_runtime.py'),freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'runtime_checks.json',value);print(value,flush=True)

if __name__=='__main__':main()
