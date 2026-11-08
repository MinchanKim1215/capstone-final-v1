"""Prototype integration before freeze; final exported bundle is tested separately."""
import numpy as np
from . import ARTIFACTS
from .contracts import source_records,read,write,now
from . import runtime
from .features import LocalHistory,ReceivedTrace,build
from .inference import predict
from forecast_rebuild.dataset import Episode
from forecast_rebuild.planning import public_context
from forecast_rebuild.calibration import transform


def main():
    r=source_records()[0];ep=Episode(r);n=ep.values.shape[2];ns=r['n_search'];city,nominal=public_context(r);city.radio.enable_cache()
    model=ARTIFACTS/'models/v11_hybrid_shared/model.json'
    policy=ARTIFACTS/'preflight_identity_policy.json'
    write(policy,dict(parameters={'2':dict(slope=1.,intercept=0.),'5':dict(slope=1.,intercept=0.)},threshold=.5,require_energy=True))
    candidate=dict(task='link',family='new',tag='prototype_hybrid',model=str(model))
    #Only this isolated preflight process bypasses export existence. The final
    #check_runtime module always loads and verifies the actual frozen export.
    runtime.bundle=lambda:dict(tasks={'link':dict(candidate=candidate,policy=str(policy))})
    stream=runtime.Stream(city,n,ns,'link',nominal)
    with np.load(ARTIFACTS/'traces'/(r['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=a['rx'];received=np.unpackbits(rx,axis=-1)[...,:n]
    trace=ReceivedTrace(tx,rx,n,ns);local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns)
    ends=np.arange(40,121,2);arrays=build(local,trace,np.c_[ends,np.zeros(len(ends),int)],city)
    arrays.update(n=np.full(len(ends),n),ns=np.full(len(ends),ns),observer=np.full(len(ends),ns),query=np.c_[np.arange(len(ends)),np.zeros(len(ends),int)])
    batch=transform(predict(str(model),arrays,device='cuda')['link'],read(policy)['parameters'])
    maximum=0.;tested=0;input_difference={key:0. for key in ('node_local','node_shared','edge_local','edge_shared','physics','global')}
    for q in range(120):
        packets=[tx[q//2,j] for j in range(n) if j!=ns and received[q//2,ns,j]] if q%2==0 else []
        own=tx[q//2,ns] if q%2==0 else None
        out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],packets,own)
        if out['status']!='ready':continue
        t=int((q+1-40)//2)
        check=build(LocalHistory(stream.local.values,stream.local.masks,stream.local.intents,n,ns),stream.received,[[q+1,0]],city)
        for key,value in check.items():input_difference[key]=max(input_difference[key],float(np.max(np.abs(value.astype(float)-arrays[key][t:t+1].astype(float)))))
        for row in out['links']:
            for field,h in [('risk_5s',2),('risk_3to7s',5)]:
                if row[field] is not None:maximum=max(maximum,abs(row[field]-float(batch[t,row['peer'],h])));tested+=1
    assert maximum<2e-5 and tested>0
    assert all(v==0 for v in input_difference.values()),input_difference
    value=dict(passed=True,prototype_only=True,final_export_verification_still_required=True,frames=120,tested_predictions=tested,
        max_abs_probability_error=maximum,feature_difference=input_difference,created_at=now())
    write(ARTIFACTS/'stream_preflight.json',value);print(value)


if __name__=='__main__':main()
