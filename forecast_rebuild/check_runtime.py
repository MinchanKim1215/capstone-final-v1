"""Real-record batch/stream equivalence and safe missing-observation behavior."""
import json
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,now
from .dataset import Episode
from .planning import public_context,profile
from .online_planning import PlanContext
from .runtime import LocalForecast
from .inference import Predictor


def main():
    records=read(ARTIFACTS/'samples/engineering/manifest.json')['contract']['records']
    record=next(r for r in records if r['role']=='selection' and r['region']=='gyeonggi_20261030_bh3_01')
    episode=Episode(record);city,nominal=public_context(record)
    observer=record['n_search'];n=episode.values.shape[2]
    model=ARTIFACTS/'models/engineering_lgbm_base/model.json'
    stream=LocalForecast(str(model),city,observer,n,record['n_search'],max_frames=60,device='cpu')
    batch=Predictor(str(model),'cpu');maximum=0.;latency=[]
    for q in range(44):
        result=stream.observe((q+1)*.5,episode.values[q,0],episode.masks[q,0],episode.intents[q,0])
        if result['status']=='ready':
            t=int(np.flatnonzero(episode.ends==q+1)[0])
            peers=[j for j in range(n) if j!=observer]
            prediction=batch.predict(episode,np.array([[t,0,j] for j in peers]))[0]
            for k,row in enumerate(result['links']):
                if row['risk_5s'] is not None:maximum=max(maximum,abs(row['risk_5s']-prediction[k,2]),abs(row['risk_3to7s']-prediction[k,5]))
                assert not row['trusted_alarm']
            assert not result['trusted_alarm_enabled'];latency.append(result['inference_ms'])
    assert maximum<1e-5,maximum
    ends=np.array([40,42,44]);offline=profile(city,nominal,episode.values[:44],episode.masks[:44],episode.intents[:44],ends,record['n_search'])
    online=PlanContext(city,nominal,observer,record['n_search']);plan_delta=0.
    for t,end in enumerate(ends):
        local=online.compute(episode.values[end-1,0].astype(np.float32),episode.masks[end-1,0],episode.intents[end-1,0].astype(np.float32),end*.5)
        np.testing.assert_allclose(offline[t,0],local,atol=3e-5,rtol=1e-5,equal_nan=True)
        good=np.isfinite(local)&np.isfinite(offline[t,0]);plan_delta=max(plan_delta,float(np.max(np.abs(local[good]-offline[t,0][good]))))
    for q in (44,45):
        masks=np.array(episode.masks[q,0]);masks[:,41]=False
        result=stream.observe((q+1)*.5,episode.values[q,0],masks,episode.intents[q,0])
    assert all(r['risk_5s'] is None and not r['trusted_alarm'] for r in result['links'])
    try:stream.observe(25.,episode.values[46,0],episode.masks[46,0],episode.intents[46,0])
    except ValueError:pass
    else:raise AssertionError('An out-of-order frame was silently accepted')
    result=dict(created_at=now(),batch_stream_max_abs_probability_error=maximum,
        public_plan_batch_stream_max_abs_error=plan_delta,missing_data_returns_null_risk=True,
        unverified_alarm_disabled=True,time_order_enforced=True,cpu_prediction_ms=latency,
        scope='Desktop CPU integration check, not selected-device latency certification')
    write(ARTIFACTS/'runtime_checks.json',result);print(json.dumps(result))


if __name__=='__main__':main()
