"""One bounded dense-target candidate; no future channel enters model inputs."""
from pathlib import Path
import ctypes
import json
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,SEED,now
from .dataset import Samples
from .models_dense import DenseResidual
from .train_neural import validate,atomic_checkpoint


def main():
    tag='expanded_dense_residual_energy';folder=ARTIFACTS/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.json').exists():return
    sample_folder=ARTIFACTS/'samples/expanded_planned'
    if not (sample_folder/'margin_manifest.json').exists():raise ValueError('Dense future targets not verified')
    data=Samples(sample_folder,'train');selection=Samples(sample_folder,'selection')
    truth=np.load(sample_folder/'train_margin.npy',mmap_mode='r',allow_pickle=False)
    torch.set_num_threads(1);torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    model=DenseResidual(data.x.shape[1]).cuda();device='cuda';steps=5000
    old=set(id(p) for p in list(model.backbone.parameters())+list(model.timing.parameters()))
    optimizer=torch.optim.AdamW([
        dict(params=[p for p in model.backbone.parameters() if p.requires_grad],lr=3e-6,initial_lr=3e-6),
        dict(params=list(model.timing.parameters()),lr=1e-5,initial_lr=1e-5),
        dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in old],lr=3e-4,initial_lr=3e-4)],weight_decay=1e-4)
    group={int(e):np.flatnonzero(data.index[:,0]==e) for e in np.unique(data.index[:,0])};groups=np.array(list(group))
    validation_positions=np.random.default_rng(SEED+1).choice(len(selection.y),size=min(20000,len(selection.y)),replace=False)
    pins={name:sha(ROOT/name) for name in ('features.py','targets.py','dataset.py','models.py','models_dense.py','train_dense.py')}
    best,best_metrics=validate(model,selection,validation_positions,True,device);history=[];stale=0
    atomic_checkpoint(folder/'best.pt',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        static_dim=data.x.shape[1],width=64,step=0,selection_score=best,source_sha256=pins))
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));started=time.monotonic()
    try:
        for step in range(1,steps+1):
            episodes=rng.choice(groups,size=4,replace=False)
            positions=np.concatenate([rng.choice(group[int(e)],64,replace=len(group[int(e)])<64) for e in episodes])
            seq,static,y,up,next_bin=data.neural_batch(positions,True)
            margin=np.asarray(truth[positions]);dense=np.isfinite(margin).all(-1)
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,next_logits,quantiles=model(torch.as_tensor(seq,device=device),torch.as_tensor(static,device=device),True)
                loss=F.binary_cross_entropy_with_logits(logits,torch.as_tensor(y,device=device))
                if up.any():loss=loss+.2*F.cross_entropy(next_logits[torch.as_tensor(up,device=device)],torch.as_tensor(next_bin[up],device=device))
                if dense.any():
                    target=torch.as_tensor(np.clip(margin[dense],-60,60)/20,device=device)[...,None]
                    error=target-quantiles[torch.as_tensor(dense,device=device)]
                    tau=torch.tensor([.1,.5,.9],device=device)
                    loss=loss+.25*torch.maximum(tau*error,(tau-1)*error).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite dense-target objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite dense-target gradient')
            optimizer.step()
            for item in optimizer.param_groups:item['lr']=item['initial_lr']*(.1+.9*.5*(1+np.cos(np.pi*step/steps)))
            if step%100==0:write(folder/'status.json',dict(state='training',step=step,target=steps,wall_seconds=time.monotonic()-started,updated_at=now()))
            if step%250==0 or step==steps:
                score,metrics=validate(model,selection,validation_positions,True,device)
                improved=score>best+1e-5
                if improved:best=score;best_metrics=metrics;stale=0
                else:stale+=1
                state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},static_dim=data.x.shape[1],width=64,
                    step=step,optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),source_sha256=pins,selection_score=score)
                atomic_checkpoint(folder/'latest.pt',state)
                if improved or not (folder/'best.pt').exists():atomic_checkpoint(folder/'best.pt',state)
                history.append(dict(step=step,score=score,best=best,metrics=metrics));write(folder/'history.json',history)
                print(json.dumps(history[-1]),flush=True)
                if stale>=6 and step>=1500:break
        for name,expected in pins.items():
            if sha(ROOT/name)!=expected:raise ValueError('Source changed during dense training')
        value=dict(family='dense_residual',tag=tag,use_energy=True,use_plan=True,checkpoint=str(folder/'best.pt'),sha256=sha(folder/'best.pt'),
            selected_step=int(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['step']),
            selection_score=best,selection=best_metrics,source_sha256=pins,samples=str(sample_folder),
            sample_manifest_sha256=sha(sample_folder/'manifest.json'),dense_targets_sha256=sha(sample_folder/'train_margin.npy'),
            completed_steps=step,wall_seconds=time.monotonic()-started,calibration_used=False,confirmation_used=False,final_test_used=False,created_at=now())
        write(folder/'model.json',value);write(folder/'status.json',dict(state='completed',step=step,selection_score=best))
    except BaseException as error:
        write(folder/'status.json',dict(state='failed',error=repr(error),updated_at=now()));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
