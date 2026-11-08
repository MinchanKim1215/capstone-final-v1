"""Actual optionalFA5 API against the fixed calibrated threshold and batch."""
import numpy as np
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now
from .runtime_options import export,load_service
from .evaluate import case_prediction
from .episodes import cohort
from forecast_graph import ARTIFACTS as GRAPH
from forecast_graph.episodes import physical
from forecast_navigation.wire import extend_trace
from forecast_navigation.check_runtime import expected_starts
from forecast_rebuild.planning import public_context
from forecast_rebuild.calibration import transform


def main():
    option=export();item=next(i for i in read(ARTIFACTS/'data_manifest.json')['records'] if i['role']=='selection');r=item['source_record'];ep=physical(r)
    n=ep.values.shape[2];ns=r['n_search'];city,nominal=public_context(r);city.radio.enable_cache();stream=load_service(city,n,ns,nominal,budget_per_hour=5)
    primary=read(ARTIFACTS/'research_candidate.json');plan=read(ARTIFACTS/'secondary_operating_points_plan.json');policy=read(primary['policy'])
    policy=dict(policy,threshold=plan['items'][primary['candidate']['tag']]['points']['5']['threshold'])
    p,_=case_prediction(primary['candidate'],item,'selection');wanted=transform(p,policy['parameters']);target=cohort(item);starts=expected_starts(target,p,policy);lookup={int(e):t for t,e in enumerate(target.ends)}
    with np.load(GRAPH/'traces'/(r['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n]
    with np.load(PARENT/'deltas'/(r['key']+'.npz'),allow_pickle=False) as a:tx=extend_trace(tx,a['delta'])
    error=0.;mismatches=0;tested=0
    for q in range(len(ep.values)):
        packets=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else [];own=tx[q//2,ns] if q%2==0 else None
        out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],packets,own)
        if out['status']!='ready' or q+1 not in lookup:continue
        t=lookup[q+1]
        for field,h in [('risk_5s',2),('risk_3to7s',5)]:
            if out[field] is not None:error=max(error,abs(out[field]-float(wanted[t,0,0,h])));tested+=1
        if q+1 in starts:mismatches+=int(out['qualified_first_alert']!=starts[q+1][0,0])
        assert out['trusted_alarm_enabled'] is False and out['calibration_FA_budget_per_hour']==5
    if error>1e-4 or mismatches or tested==0:raise ValueError('Optional operating API mismatch')
    value=dict(passed=True,created_at=now(),frames=len(ep.values),budget=5,threshold=policy['threshold'],tested_predictions=tested,max_probability_error=error,
        first_alert_mismatches=mismatches,trusted_alarm_disabled=True,options_export_sha256=sha(ARTIFACTS/'research_operating_options.json'),
        adapter_sha256=sha(ROOT/'runtime_options.py'),checker_sha256=sha(__file__),primary_default_unchanged=True)
    write(ARTIFACTS/'runtime_options_checks.json',value);print(value,flush=True)


if __name__=='__main__':main()
