"""Real incremental packet-state/physics pipeline, nonzero TRAINfixture head."""
import numpy as np
import torch
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .model import CausalAdapter
from .inputs import load
from .inference import predict
from . import runtime
from forecast_graph.episodes import physical
from forecast_rebuild.planning import public_context

def main():
    torch.set_num_threads(2);item=read(ARTIFACTS/'data/969533adae14a5e6b101.json');r=item['source_record'];ep=physical(r);ns=r['n_search'];n=ep.values.shape[2];city,nominal=public_context(r);city.radio.enable_cache()
    folder=ARTIFACTS/'fixture_runtime';folder.mkdir(exist_ok=True);ckpt=folder/'nonzero.pt';meta=folder/'model.json';policy=folder/'identity_policy.json'
    torch.manual_seed(1601);net=CausalAdapter('gru_network');torch.nn.init.normal_(net.head[-1].weight,std=.01);torch.save(dict(model=net.state_dict()),ckpt)
    write(meta,dict(family='v16_gru',variant='gru_network',member_index=0,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt)));write(policy,dict(parameters={str(h):dict(slope=1.,intercept=0.) for h in (2,5)},threshold=.5))
    candidate=dict(task='service',tag='nonzero_measured_network_fixture',family='new',model=str(meta));runtime.bundle=lambda:dict(candidate=candidate,policy=str(policy));stream=runtime.load_service(city,n,ns,nominal)
    inputs=load(item);expected=predict(str(meta),inputs,device='cpu');lookup={int(e):q for q,e in enumerate(inputs['ends'])};errors={k:0. for k in ('network_history','network','physics','base_RF','risk')};tested=0
    with np.load(item['replay_path']) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n];observed=a['observed']
    for q in range(120):
        received=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else [];own=tx[q//2,ns] if q%2==0 else None
        out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],received,own,observed[q,ns])
        errors['network_history']=max(errors['network_history'],float(np.abs(np.asarray(stream.summaries[-1],np.float16).astype(float)-inputs['network_history'][q].astype(float))[np.isfinite(inputs['network_history'][q])].max(initial=0)))
        if out['status']!='ready' or q+1 not in lookup:continue
        i=lookup[q+1]
        for key in ('network','physics'):
            a=stream.last_inputs[key];b=inputs[key][i:i+1];np.testing.assert_equal(np.isnan(a),np.isnan(b));errors[key]=max(errors[key],float(np.nanmax(np.abs(a-b))))
        errors['base_RF']=max(errors['base_RF'],float(np.nanmax(np.abs(stream.last_inputs['base'][:,:432]-inputs['base'][i:i+1,:432]))))
        for field,h in [('risk_5s',2),('risk_3to7s',5)]:
            if out[field] is not None:errors['risk']=max(errors['risk'],abs(out[field]-float(expected[i,h])));tested+=1
    if any(v for k,v in errors.items() if k!='risk') or errors['risk']>1e-4 or tested==0:raise ValueError(str(errors))
    value=dict(passed=True,created_at=now(),training_fixture_only=True,frames=120,tested_risks=tested,errors=errors,raw_diagnostic_truth_not_runtime_argument=True,
        source_pins={k:sha(ROOT/k) for k in ('runtime.py','inference.py','features.py','physics.py','wire.py')})
    write(ARTIFACTS/'stream_preflight.json',value);print(value,flush=True)

if __name__=='__main__':main()
