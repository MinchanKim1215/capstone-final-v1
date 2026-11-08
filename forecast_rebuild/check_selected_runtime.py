"""Selected-model batch/stream and unavailable-sensor integration checks."""
import json
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,now
from .dataset import Episode
from .planning import public_context
from .runtime_stable import StableLocalForecast as LocalForecast
from .inference import Predictor
from .calibration import transform,alarm_support,verify_freeze


def main():
    value=verify_freeze();summary=read(value['selection_summary']);model=value['selected_model'];policy=read(value['selected_alarm'])
    records=read(summary['dataset_path'])['rows']
    record=next(r for r in records if r['role']=='selection');episode=Episode(record)
    city,nominal=public_context(record);observer=record['n_search'];n=episode.values.shape[2]
    mission_frames=len(episode.values)-4;mission_frames-=mission_frames%2
    stream=LocalForecast(model,city,observer,n,record['n_search'],policy=value['selected_alarm'],nominal=nominal,max_frames=len(episode.values)+2)
    batch=Predictor(model,'cpu');maximum=0.;latency=[];tested=0;feature_delta=0.;feature_details=[]
    offline_active=np.zeros(n,bool);alarm_differences=0
    for q in range(mission_frames):
        result=stream.observe((q+1)*.5,episode.values[q,0],episode.masks[q,0],episode.intents[q,0],episode.energy[q,0])
        if result['status']!='ready':continue
        t=int(np.flatnonzero(episode.ends==q+1)[0]);peers=[j for j in range(n) if j!=observer]
        prediction=transform(batch.predict(episode,np.array([[t,0,j] for j in peers]))[0],policy['parameters'])
        offline_features=episode.features(np.array([[t,0,j] for j in peers]),batch.meta['use_energy'],batch.meta.get('use_plan',False))
        online_features=stream.features(np.array([[0,0,j] for j in peers]),batch.meta['use_energy'],batch.meta.get('use_plan',False))
        np.testing.assert_equal(np.isnan(offline_features),np.isnan(online_features))
        delta=np.abs(np.nan_to_num(offline_features)-np.nan_to_num(online_features))
        feature_delta=max(feature_delta,float(np.max(delta)))
        if np.max(delta)>1e-7:
            ii,jj=np.unravel_index(np.argmax(delta),delta.shape)
            feature_details.append(dict(time=q*.5+.5,column=int(jj),offline=float(offline_features[ii,jj]),online=float(online_features[ii,jj]),delta=float(delta[ii,jj])))
        eligible=alarm_support(episode,policy.get('require_energy',False))[t,0]
        for k,row in enumerate(result['links']):
            if row['risk_5s'] is not None:
                maximum=max(maximum,abs(row['risk_5s']-prediction[k,2]),abs(row['risk_3to7s']-prediction[k,5]));tested+=1
            assert row['alarm_data_eligible']==bool(eligible[row['peer']])
            j=row['peer'];call=bool(eligible[j] and prediction[k,5]>=(policy['threshold']*.5 if offline_active[j] else policy['threshold']))
            alarm_differences+=int(row['qualified_first_alert']!=(call and not offline_active[j]));offline_active[j]=call
            assert not row['trusted_alarm']
        assert not result['trusted_alarm_enabled'];latency.append(result['inference_ms'])
        if q%400==399:print(json.dumps(dict(frames=q+1,total=mission_frames,feature_delta=feature_delta,probability_delta=float(maximum))),flush=True)
    print(json.dumps(dict(feature_max_abs_delta=feature_delta,probability_max_abs_delta=float(maximum),alarm_differences=alarm_differences,details=feature_details[:5])),flush=True)
    assert tested>0 and maximum<2e-5,(tested,maximum)
    assert feature_delta<2e-6 and alarm_differences==0,(feature_delta,alarm_differences)
    for q in (mission_frames,mission_frames+1):
        masks=np.array(episode.masks[q,0]);masks[:,41]=False
        result=stream.observe((q+1)*.5,episode.values[q,0],masks,episode.intents[q,0])
    assert all(r['risk_5s'] is None and not r['trusted_alarm'] for r in result['links'])
    # Fresh RF with explicitly absent optional sensors must also lose the
    # energy-qualified alarm entitlement, never become a "zero risk" reading.
    for q in (mission_frames+2,mission_frames+3):
        result=stream.observe((q+1)*.5,episode.values[q,0],episode.masks[q,0],episode.intents[q,0])
    if policy.get('require_energy',False):assert not any(r['alarm_data_eligible'] for r in result['links'])
    out=dict(model_tag=summary['model_tag'],created_at=now(),tested_link_predictions=tested,
        batch_stream_max_abs_probability_error=maximum,missing_data_returns_null_risk=True,
        public_plan_feature_max_abs_delta=feature_delta,alarm_decision_mismatches=alarm_differences,frames=mission_frames,
        runtime_adapter='runtime_stable.StableLocalForecast',
        missing_energy_blocks_sensor_qualified_alarm=True,unverified_alarm_disabled=True,
        cpu_ms=dict(cold=latency[0],warm_median=float(np.median(latency[1:])),warm_max=float(np.max(latency[1:]))),
        hardware_latency_certified=False)
    write(ARTIFACTS/'selected_runtime_checks.json',out);print(json.dumps(out))


if __name__=='__main__':main()
