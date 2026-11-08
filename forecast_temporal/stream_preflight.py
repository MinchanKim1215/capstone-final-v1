"""Actual incremental20-node/edge history against independently cached input."""
import numpy as np
import torch
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now
from .train import initialize
from .inference import predict
from forecast_graph.episodes import physical
from forecast_rebuild.planning import public_context
from forecast_navigation.wire import extend_trace
from . import runtime

def main():
    torch.set_num_threads(2);parent=read(PARENT/'data_manifest.json')['records'][0];item=read(ARTIFACTS/'data'/(parent['key']+'.json'));r=item['source_record']
    ep=physical(r);ns=r['n_search'];n=ep.values.shape[2];city,nominal=public_context(r);city.radio.enable_cache()
    folder=ARTIFACTS/'fixture';folder.mkdir(exist_ok=True);checkpoint=folder/'long_stream.pt';meta=folder/'long_stream.json';policy=folder/'long_policy.json'
    net=initialize('long_event',0,'cpu');torch.save(dict(model=net.state_dict()),checkpoint)
    write(meta,dict(family='v14_temporal_graph',variant='long_event',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
    parameters={str(h):dict(slope=1.,intercept=0.) for h in (2,5)};write(policy,dict(parameters=parameters,threshold=.5))
    candidate=dict(task='service',family='new',tag='fixture_long_event',model=str(meta));runtime.bundle=lambda:dict(candidate=candidate,policy=str(policy))
    stream=runtime.load_service(city,n,ns,nominal)
    with np.load(item['path']) as a:reference={k:a[k] for k in ('ends','node','edge','index','physics','global_features','n','ns','observer')}
    lookup={int(e):i for i,e in enumerate(reference['ends'])};expected=predict(str(meta),reference,device='cpu');errors={k:0. for k in ('node','edge','physics','global_features','probability')};tested=0
    with np.load(item['trace_path']) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n]
    with np.load(item['delta_path']) as a:tx=extend_trace(tx,a['delta'])
    frames=min(120,len(ep.values))
    for q in range(frames):
        packets=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else [];own=tx[q//2,ns] if q%2==0 else None
        out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],packets,own)
        if out['status']!='ready' or q+1 not in lookup:continue
        i=lookup[q+1]
        for key in ('node','edge'):
            wanted=reference[key][reference['index'][i]];errors[key]=max(errors[key],float(np.abs(stream.last_inputs[key].astype(float)-wanted.astype(float)).max()))
        for key in ('physics','global_features'):errors[key]=max(errors[key],float(np.abs(stream.last_inputs[key].astype(float)-reference[key][i:i+1].astype(float)).max()))
        for field,h in [('risk_5s',2),('risk_3to7s',5)]:
            if out[field] is not None:errors['probability']=max(errors['probability'],abs(out[field]-float(expected[i,h])));tested+=1
        assert not out['trusted_alarm_enabled']
    if any(v for k,v in errors.items() if k!='probability') or errors['probability']>1e-4 or tested==0:raise ValueError(str(errors))
    value=dict(passed=True,created_at=now(),training_only=True,frames=frames,tested_risks=tested,errors=errors,actual_20_snapshot_runtime=True,source_pins={k:sha(ROOT/k) for k in ('runtime.py','features.py','model.py','inference.py')})
    write(ARTIFACTS/'stream_preflight.json',value);print(value,flush=True)

if __name__=='__main__':main()
