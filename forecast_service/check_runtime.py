"""Exported APIs versus direct batch predictions across one complete flight."""
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now,V9
from .runtime_aligned import load_link,load_service
from .evaluate import verify_freeze
from .inference import tree_predict
from .features import build
from .data import ServiceEpisode
from forecast_rebuild.dataset import Episode
from forecast_rebuild.inference import Predictor
from forecast_rebuild.planning import public_context
from forecast_rebuild.calibration import transform,alarm_support


def main():
    frozen=verify_freeze();export=read(ARTIFACTS/'research_candidates.json')
    record=next(r for r in read(V9/'dataset.json')['rows'] if r['role']=='selection' and r['source_kind']=='fresh_v9')
    ep=Episode(record);service_ep=ServiceEpisode(record);city,nominal=public_context(record);ns=record['n_search'];n=ep.values.shape[2]
    streams=dict(link=load_link(city,ns,n,ns,nominal),service=load_service(city,ns,n,ns,nominal))
    policies={task:read(item['policy']) for task,item in export['tasks'].items()}
    linkmodel=Predictor(export['tasks']['link']['model'],'cpu');service_model=read(export['tasks']['service']['model'])
    proxy=Predictor(service_model['link_model'],'cpu') if service_model['family']=='service_proxy' else None
    support=alarm_support(ep,policies['link']['require_energy']);previous=np.zeros(n,bool);service_previous=False
    maximum={'link':0.,'service':0.};tested={'link':0,'service':0};mismatches=0;latency={task:[] for task in streams}
    frames=len(ep.values)-4;frames-=frames%2
    for q in range(frames):
        args=((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0])
        values={task:stream.observe(*args) for task,stream in streams.items()}
        if (q+1)<40 or (q+1)%2:continue
        t=int(np.flatnonzero(ep.ends==q+1)[0]);peers=[j for j in range(n) if j!=ns];queries=np.array([[t,0,j] for j in peers])
        raw=linkmodel.predict(ep,queries)[0];p=transform(raw,policies['link']['parameters'])
        for k,row in enumerate(values['link']['links']):
            j=row['peer']
            if row['risk_5s'] is not None:
                maximum['link']=max(maximum['link'],abs(row['risk_5s']-p[k,2]),abs(row['risk_3to7s']-p[k,5]));tested['link']+=1
            active=bool((q+1)*.5>=21 and support[t,0,j] and float(p[k,5])>=(policies['link']['threshold']*.5 if previous[j] else policies['link']['threshold']))
            mismatches+=int(row['qualified_first_alert']!=(active and not previous[j]));previous[j]=active;assert not row['trusted_alarm']
        if proxy:
            raw=proxy.predict(ep,queries)[0];known=np.asarray(ep.legacy_support[t,0,peers]);raw=np.where(known[:,None],raw,np.nan)
            expected=np.full((1,6),np.nan,np.float32)
            for head in (2,5):
                finite=np.isfinite(raw[:,head])
                if finite.any():expected[0,head]=raw[finite,head].max()
        else:expected=tree_predict(export['tasks']['service']['model'],build(ep,np.array([[t,0]]),ns,city))
        expected=transform(expected,policies['service']['parameters']);value=values['service'];eligible=service_ep.legacy_support[t,0,0]
        if value['service_risk_5s'] is not None:
            maximum['service']=max(maximum['service'],abs(value['service_risk_5s']-expected[0,2]),abs(value['service_risk_3to7s']-expected[0,5]));tested['service']+=1
        active=bool((q+1)*.5>=21 and eligible and np.isfinite(expected[0,5]) and float(expected[0,5])>=(policies['service']['threshold']*.5 if service_previous else policies['service']['threshold']))
        mismatches+=int(value['first_alert']!=(active and not service_previous));service_previous=active;assert not value['trusted_alarm']
        for task in streams:
            if 'inference_ms' in values[task]:latency[task].append(values[task]['inference_ms'])
    assert mismatches==0 and all(tested[task]>0 and maximum[task]<2e-5 for task in streams),(mismatches,tested,maximum)
    for q in (frames,frames+1):
        mask=np.array(ep.masks[q,0]);mask[:,41]=False
        results={task:stream.observe((q+1)*.5,ep.values[q,0],mask,ep.intents[q,0]) for task,stream in streams.items()}
    assert all(row['risk_5s'] is None and not row['trusted_alarm'] for row in results['link']['links'])
    assert results['service']['service_risk_5s'] is None and not results['service']['trusted_alarm']
    write(ARTIFACTS/'runtime_checks.json',dict(passed=True,created_at=now(),frames=frames,record_key=record['key'],
        tested_predictions=tested,max_abs_probability_error={k:float(v) for k,v in maximum.items()},alarm_mismatches=mismatches,
        missing_RF_returns_None=True,trusted_alarm_disabled=True,export_sha256=sha(ARTIFACTS/'research_candidates.json'),
        freeze_sha256=sha(ARTIFACTS/'freeze.json'),checker_sha256=sha(ROOT/'check_runtime.py'),
        alarm_start_s=21.,alarm_start_adapter_sha256=sha(ROOT/'runtime_aligned.py'),
        cpu_ms={task:dict(cold=times[0],warm_median=float(np.median(times[1:])),warm_max=float(np.max(times[1:]))) for task,times in latency.items()},hardware_latency_certified=False))
    print(ARTIFACTS/'runtime_checks.json')


if __name__=='__main__':main()
