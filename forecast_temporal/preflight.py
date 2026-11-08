"""Causality, preserved short-history baseline and actual GPU gradient checks."""
import copy
import time
import numpy as np
import torch
from . import ARTIFACTS,PARENT,ROOT,NAV
from .contracts import read,write,now,sha
from .data import prepare_one
from .train import load,DeviceData,initialize,loss
from .features import build
from forecast_rebuild.dataset import Episode
from forecast_graph.features import LocalHistory,ReceivedTrace,build as old_build
from forecast_navigation.wire import extend_trace
from forecast_rebuild.planning import public_context
from forecast_graph.model import tensors

def main():
    torch.set_num_threads(2);parent=next(i for i in read(PARENT/'data_manifest.json')['records'] if i['role']=='train');item=prepare_one(parent)
    a=load([item]);data=DeviceData(a);ix=torch.arange(3,device='cuda');n=initialize('short_event',0);n.eval()
    r=item['source_record'];ep=Episode(r);ns=r['n_search'];fleet=ep.values.shape[2];city,_=public_context(r);city.radio.enable_cache()
    with np.load(item['trace_path']) as z:tx=z['tx'];rx=z['rx']
    with np.load(item['delta_path']) as z:tx=extend_trace(tx,z['delta'])
    local=LocalHistory(ep.values,ep.masks,ep.intents,fleet,ns);trace=ReceivedTrace(tx,rx,fleet,ns);ends=np.asarray(a['cases'][0][0].ends[:3])
    old=old_build(local,trace,np.c_[ends,np.zeros(3,int)],city);old.update(n=np.full(3,fleet),ns=np.full(3,ns),observer=np.full(3,ns))
    with torch.inference_mode():
        expected=n(**tensors(old,np.arange(3),'hybrid_shared'))['service'].cpu().numpy();actual=n(**data.inputs(ix))['service'].cpu().numpy()
    delta=float(np.max(np.abs(expected-actual)))
    if delta>1e-4:raise ValueError('Cropped short-history warm model differs: '+str(delta))
    # Future local frames and not-received beacon reports cannot change input.
    end=int(ends[0]);base=build(local,trace,[end],city);v=np.array(ep.values,copy=True);v[end:]+=91;v[:end][~np.asarray(ep.masks[:end],bool)]+=57
    changed=build(LocalHistory(v,ep.masks,ep.intents,fleet,ns),trace,[end],city)
    for key in base:np.testing.assert_equal(base[key],changed[key])
    timings={};gradients={}
    u=np.resize(data.uniform,96);bags=np.resize(data.bags,(16,4))
    for variant in ('short_bce','short_event','long_event'):
        model=initialize(variant,0);opt=torch.optim.AdamW(model.parameters(),lr=.00015);began=time.monotonic()
        for k in range(5):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):value,terms=loss(model,data,u,bags,0)
            value.backward()
            if not torch.isfinite(value) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Invalid GPU gradients')
            opt.step()
        torch.cuda.synchronize();timings[variant]=(time.monotonic()-began)/5
        gradients[variant]={key:float(next(getattr(model,key).parameters()).grad.abs().sum()) for key in ('node','edge','message','temporal','edge_time','service')}
        if any(v<=0 for v in gradients[variant].values()):raise ValueError('Backbone was not trainable')
        del model,opt;torch.cuda.empty_cache()
    write(ARTIFACTS/'preflight.json',dict(state='passed',created_at=now(),train_fixture=item['key'],short_history_max_logit_delta=delta,past_only_and_unobserved_perturbation_exact=True,
        actual_gpu_step_seconds=timings,actual_backbone_gradient_sums=gradients,peak_gpu_bytes=torch.cuda.max_memory_allocated(),source_pins={k:sha(ROOT/k) for k in ('train.py','model.py','features.py','data.py')}))
    print('preflight passed',delta,timings,flush=True)

if __name__=='__main__':main()
