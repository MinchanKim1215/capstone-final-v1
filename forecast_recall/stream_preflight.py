"""Prefreeze actual live path with a nonzero, untrained memory fixture."""
import numpy as np
import torch
from . import ARTIFACTS,PARENT
from .contracts import read,write,sha,now
from .model import EventAdapter
from .inference import predict,baseline
from .features import sequence
from . import runtime
from forecast_graph.episodes import physical
from forecast_graph.contracts import ARTIFACTS as GRAPH
from forecast_navigation.wire import extend_trace
from forecast_rebuild.planning import public_context
from forecast_rebuild.calibration import transform


def main():
    item=next(i for i in read(ARTIFACTS/'data_manifest.json')['records'] if i['role']=='selection');r=item['source_record'];ep=physical(r)
    n=ep.values.shape[2];ns=r['n_search'];city,nominal=public_context(r);city.radio.enable_cache();folder=ARTIFACTS/'fixture';folder.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(130131);net=EventAdapter('memory_event').eval()
    torch.nn.init.normal_(net.head[-1].weight,0,.01);checkpoint=folder/'stream_prototype.pt';torch.save(dict(model=net.state_dict()),checkpoint)
    meta=folder/'stream_prototype.json';write(meta,dict(family='v13_event_adapter',variant='memory_event',member_index=0,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
    policy=folder/'identity_policy.json';write(policy,dict(parameters={'2':dict(slope=1.,intercept=0.),'5':dict(slope=1.,intercept=0.)},threshold=.5))
    runtime.bundle=lambda:dict(candidate=dict(family='new',tag='prototype',model=str(meta)),policy=str(policy))
    stream=runtime.Stream(city,n,ns,nominal)
    with np.load(item['path'],allow_pickle=False) as a:reference={k:a[k] for k in ('embedding','logits','history','ends')};support=a['support']
    count=int((reference['ends']<=120).sum());lookup={int(e):i for i,e in enumerate(reference['ends'][:count])}
    small=dict(reference,embedding=reference['embedding'][:count],logits=reference['logits'][:count],ends=reference['ends'][:count])
    expected=transform(predict(meta,small,device='cuda'),read(policy)['parameters']);old=baseline(item,'selection')[:count,0,0]
    with np.load(GRAPH/'traces'/(r['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=np.unpackbits(a['rx'],axis=-1)[...,:n]
    with np.load(PARENT/'deltas'/(r['key']+'.npz'),allow_pickle=False) as a:tx=extend_trace(tx,a['delta'])
    errors=dict(history=0.,embedding=0.,logits=0.,risk=0.,baseline_risk=0.);mismatches=0;active=False;tested=0
    for q in range(120):
        received=[tx[q//2,j] for j in range(n) if j!=ns and rx[q//2,ns,j]] if q%2==0 else [];own=tx[q//2,ns] if q%2==0 else None
        out=stream.observe((q+1)*.5,ep.values[q,0],ep.masks[q,0],ep.intents[q,0],ep.energy[q,0],received,own)
        if out['status']!='ready':continue
        t=lookup[q+1]
        for key in ('embedding','logits'):errors[key]=max(errors[key],float(np.abs(stream.last_inputs[key]-reference[key][t:t+1]).max()))
        errors['history']=max(errors['history'],float(np.abs(sequence(stream.last_inputs['history'],[q+1])-sequence(reference['history'],[q+1])).max()))
        errors['baseline_risk']=max(errors['baseline_risk'],float(np.abs(stream.baseline_raw[0,[2,5]]-old[t,[2,5]]).max()))
        if out['risk_5s'] is not None:
            errors['risk']=max(errors['risk'],abs(out['risk_5s']-float(expected[t,2])),abs(out['risk_3to7s']-float(expected[t,5])));tested+=2
        current=bool(support[t] and expected[t,5]>=(.25 if active else .5))
        if (q+1)*.5<21:current=False
        mismatches+=int(out['qualified_first_alert']!=(current and not active));active=current
    write(ARTIFACTS/'stream_preflight_diagnostic.json',dict(errors=errors,first_alert_mismatches=mismatches,created_at=now()))
    assert errors['history']==0 and all(v<=1e-4 for v in errors.values()) and mismatches==0 and tested>0
    value=dict(passed=True,prototype_only=True,frames=120,nonzero_memory_fixture=True,errors=errors,first_alert_mismatches=mismatches,tested_predictions=tested,created_at=now())
    write(ARTIFACTS/'stream_preflight.json',value);print(value,flush=True)


if __name__=='__main__':main()
