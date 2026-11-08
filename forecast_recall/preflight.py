"""Meaningful data-causality, complete-event target and GPU-gradient checks."""
import numpy as np
import torch
from . import ARTIFACTS,PARENT
from .contracts import read,write,now
from .data import prepare_one
from .features import steps,sequence
from .train import load,DeviceData,loss
from .model import EventAdapter
from forecast_rebuild.dataset import Episode


def main():
    item=read(PARENT/'data_manifest.json')['records'][0];prepared=prepare_one(item);ep=Episode(item['source_record']);r=item['source_record'];n=ep.values.shape[2];ns=r['n_search']
    end=60;v=np.array(ep.values[:100,0]);m=np.array(ep.masks[:100,0]);c=np.array(ep.intents[:100,0]);e=np.array(ep.energy[:100,0])
    original=steps(v,m,c,e,n,ns)
    future=v.copy();future[end:]=5000;future_context=steps(future,m,c,e,n,ns)
    np.testing.assert_equal(sequence(original,[end]),sequence(future_context,[end]))
    hidden=v.copy();hidden[~m]=5000;np.testing.assert_equal(steps(hidden,m,c,e,n,ns),original)
    a=load([prepared]);assert len(a['bags']) and all(a['y'][row[row>=0],5].all() for row in a['bags'])
    tr=DeviceData(a,0);uniform=tr.uniform[:256];bags=tr.bags[np.arange(32)%len(tr.bags)];losses={}
    for variant in ('bce','event','memory_event'):
        torch.manual_seed(1301);net=EventAdapter(variant).cuda();net.eval()
        with torch.inference_mode():
            ix=torch.as_tensor(uniform[:8],device='cuda');out,_=net(*tr.inputs(ix,variant=='memory_event'))
            torch.testing.assert_close(out,tr.a['logits'][ix],rtol=0,atol=0)
        net.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):value,terms=loss(net,tr,uniform,bags,variant)
        value.backward();assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        losses[variant]=dict(total=float(value.detach()),**terms)
    value=dict(passed=True,created_at=now(),training_region_only=True,future_RF_perturbation_exact=True,
        unobserved_RF_perturbation_exact=True,initial_residual_predictions_exact=True,event_bags_have_only_true_pre_event_targets=True,
        finite_GPU_mixed_precision_forward_backward=True,losses=losses)
    write(ARTIFACTS/'preflight.json',value);print(value,flush=True)


if __name__=='__main__':main()
