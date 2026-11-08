"""Real teacher zero-equivalence, observed-only input and finite GPU training."""
import numpy as np
import torch
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now
from .data import prepare_one
from .features import steps,sequence
from .train import load,DeviceData,loss
from .model import PeerAdapter
from forecast_graph.episodes import physical

def main():
    torch.set_num_threads(2);parent=next(i for i in read(PARENT/'data_manifest.json')['records'] if i['role']=='train');item=prepare_one(parent);a=load([item]);data=DeviceData(a,0)
    ep=physical(item['source_record']);n=ep.values.shape[2];ns=item['source_record']['n_search'];end=int(ep.ends[5]);raw=np.asarray(ep.values[1::2,0]);m=np.asarray(ep.masks[1::2,0]);c=np.asarray(ep.intents[1::2,0]);e=np.asarray(ep.energy[1::2,0])
    original=steps(raw,m,c,e,n,ns);changed=raw.copy();changed[~m]+=999;changed[end//2:]+=88
    altered=steps(changed,m,c,e,n,ns);np.testing.assert_equal(sequence(original,[end]),sequence(altered,[end]))
    ix=torch.arange(3,device='cuda');maximum=0.;losses={};gradients={}
    for variant in ('global_event','peer_bce','peer_event'):
        torch.manual_seed(2026110315);net=PeerAdapter(variant).cuda();net.eval()
        with torch.no_grad():z,_=net(*data.inputs(ix));maximum=max(maximum,float((z-data.a['base_logits'][ix]).abs().max()))
        net.train();opt=torch.optim.AdamW(net.parameters(),lr=.0003);u=np.resize(data.uniform,256);bags=np.resize(data.bags,(32,4))
        for k in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):value,terms=loss(net,data,u,bags,variant)
            value.backward()
            if not torch.isfinite(value) or any(not torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):raise ValueError('Invalid peer GPU gradient')
            opt.step()
        losses[variant]=float(value.detach());gradients[variant]=float(next(net.memory.parameters()).grad.abs().sum())
        if gradients[variant]<=0:raise ValueError('Temporal input did not learn')
        del net,opt;torch.cuda.empty_cache()
    if maximum!=0:raise ValueError('Zero-residual teacher equivalence failed')
    value=dict(state='passed',passed=True,created_at=now(),zero_residual_logit_error=maximum,unobserved_and_future_RF_perturbations_exact=True,finite_actual_GPU_loss=losses,memory_gradients=gradients,
        source_pins={k:sha(ROOT/k) for k in ('features.py','data.py','model.py','train.py','inference.py')})
    write(ARTIFACTS/'preflight.json',value);print(value,flush=True)

if __name__=='__main__':main()
