"""Nonzero peer-head actual incremental API against independent training cache."""
import numpy as np
import torch
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .data import inputs as data_inputs
from .model import PeerAdapter
from .inference import predict
from .features import sequence
from . import runtime
from forecast_graph.episodes import physical
from forecast_rebuild.planning import public_context
from forecast_navigation.wire import extend_trace

def main():
    torch.set_num_threads(2);item=read(ARTIFACTS/'data_manifest.json')['records'][0];r=item['source_record'];ep=physical(r);ns=r['n_search'];n=ep.values.shape[2];city,nominal=public_context(r);city.radio.enable_cache()
    folder=ARTIFACTS/'fixture';folder.mkdir(exist_ok=True);checkpoint=folder/'peer_nonzero.pt';meta=folder/'peer_nonzero.json';policy=folder/'identity_policy.json'
    torch.manual_seed(2026110315);net=PeerAdapter('peer_event');torch.nn.init.normal_(net.head[-1].weight,std=.01);torch.save(dict(model=net.state_dict()),checkpoint)
    write(meta,dict(family='v15_peer_adapter',variant='peer_event',member_index=0,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
    write(policy,dict(parameters={str(h):dict(slope=1.,intercept=0.) for h in (2,5)},threshold=.5));candidate=dict(task='service',tag='nonzero_peer_fixture',family='new',model=str(meta))
    runtime.bundle=lambda:dict(candidate=candidate,policy=str(policy));stream=runtime.load_service(city,n,ns,nominal)
    a=data_inputs(item);expected=predict(str(meta),a,device='cpu');lookup={int(e):i for i,e in enumerate(a['ends'])};maximum=0.;feature_error=0.;tested=0
    with np.load(ROOT.parent/'forecast_graph/artifacts/traces'/(r['key']+'.npz')) as z:tx=z['tx'];rx=np.unpackbits(z['rx'],axis=-1)[...,:n]
    with np.load(ROOT.parent/'forecast_navigation/artifacts/deltas'/(r['key']+'.npz')) as z:tx=extend_trace(tx,z['delta'])
    for q in range(min(120,len(ep.values))):
        received=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else [];own=tx[q//2,ns] if q%2==0 else None
        out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],received,own)
        if out['status']!='ready' or q+1 not in lookup:continue
        i=lookup[q+1];feature_error=max(feature_error,float(np.abs(sequence(np.asarray(stream.peer_history),[q+1])-sequence(a['peer_history'],[q+1])).max()))
        for field,h in [('risk_5s',2),('risk_3to7s',5)]:
            if out[field] is not None:maximum=max(maximum,abs(out[field]-float(expected[i,h])));tested+=1
        assert out['trusted_alarm_enabled'] is False
    if feature_error or maximum>1e-4 or tested==0:raise ValueError(str((feature_error,maximum,tested)))
    value=dict(passed=True,created_at=now(),training_only=True,frames=120,tested_risks=tested,raw_peer_history_error=feature_error,nonzero_head_risk_error=maximum,
        runtime_sha256=sha(ROOT/'runtime.py'),inference_sha256=sha(ROOT/'inference.py'))
    write(ARTIFACTS/'stream_preflight.json',value);print(value,flush=True)

if __name__=='__main__':main()
