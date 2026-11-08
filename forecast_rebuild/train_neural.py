"""Bounded, resumable supervised training. All model choice uses selection only."""
from pathlib import Path
import argparse
import ctypes
import json
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,SEED
from .dataset import Samples
from .models import TemporalPredictor,ResidualTransformer
from .metrics import binary


def atomic_checkpoint(path,value):
    temporary=Path(str(path)+'.working');torch.save(value,temporary);os.replace(temporary,path)


def validate(model,data,positions,use_energy,device):
    probabilities=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(positions),512):
            seq,static,_,_,_=data.neural_batch(positions[start:start+512],use_energy)
            a,_=model(torch.as_tensor(seq,device=device),torch.as_tensor(static,device=device))
            probabilities.append(torch.sigmoid(a).cpu().numpy())
    p=np.concatenate(probabilities);y=np.asarray(data.y[positions]);old=data.up[positions]&data.legacy[positions]
    result=dict(next5=binary(y[old,2],p[old,2]),any3to7=binary(y[:,5],p[:,5]))
    return float(np.mean([result['next5']['ap'],result['any3to7']['ap']])),result


def train(samples,tag,use_energy=True,steps=3000,resume=False,family='tcn'):
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED)
    if not torch.cuda.is_available():raise RuntimeError('A GPU is required for this bounded study')
    device='cuda';folder=ARTIFACTS/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.json').exists():return read(folder/'model.json')
    data=Samples(samples,'train');selection=Samples(samples,'selection')
    pins={name:sha(ROOT/name) for name in ('features.py','targets.py','dataset.py','models.py','train_neural.py')}
    model=(ResidualTransformer(data.x.shape[1]) if family=='residual_transformer' else TemporalPredictor(data.x.shape[1])).to(device)
    prior=np.asarray(data.y).mean(0).clip(1e-5,1-1e-5)
    if family=='tcn':
        with torch.no_grad():model.any_head.bias.copy_(torch.as_tensor(np.log(prior/(1-prior)),device=device,dtype=torch.float32))
        groups=[dict(params=list(model.parameters()),lr=3e-4,initial_lr=3e-4)]
    else:
        old=set(id(p) for p in list(model.backbone.parameters())+list(model.timing.parameters()))
        groups=[dict(params=[p for p in model.backbone.parameters() if p.requires_grad],lr=3e-6,initial_lr=3e-6),
                dict(params=list(model.timing.parameters()),lr=1e-5,initial_lr=1e-5),
                dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in old],lr=3e-4,initial_lr=3e-4)]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    rng=np.random.default_rng(SEED);start=0;best=-1.;best_metrics=None;stale=0;history=[]
    config=dict(steps=steps,batch_size=256,use_energy=use_energy,family=family,samples_manifest_sha256=sha(Path(samples)/'manifest.json'))
    if resume:
        state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if state['config']!=config or state['source_sha256']!=pins:raise ValueError('Resume settings/source changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for opt in optimizer.state.values():
            for key,value in opt.items():
                if torch.is_tensor(value):opt[key]=value if key=='step' else value.to(device)
        start=state['step'];best=state['best'];best_metrics=state['best_metrics'];stale=state['stale'];history=state['history']
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    group={int(e):np.flatnonzero(data.index[:,0]==e) for e in np.unique(data.index[:,0])}
    groups=np.array(list(group))
    validation_positions=np.random.default_rng(SEED+1).choice(len(selection.y),size=min(20000,len(selection.y)),replace=False)
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));started=time.monotonic()
    try:
        if start==0 and family=='residual_transformer':
            best,best_metrics=validate(model,selection,validation_positions,use_energy,device)
            history=[dict(step=0,score=best,metrics=best_metrics)]
            initial=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                step=0,config=config,source_sha256=pins,best=best,best_metrics=best_metrics,stale=0,history=history,
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                static_dim=data.x.shape[1],width=64)
            atomic_checkpoint(folder/'best.pt',initial)
            print(json.dumps(dict(stage='preserved_initial_predictor',selection_score=best)),flush=True)
        for step in range(start+1,steps+1):
            episodes=rng.choice(groups,size=4,replace=False)
            positions=np.concatenate([rng.choice(group[int(e)],size=64,replace=len(group[int(e)])<64) for e in episodes])
            seq,static,y,up,next_bin=data.neural_batch(positions,use_energy)
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,next_logits=model(torch.as_tensor(seq,device=device),torch.as_tensor(static,device=device))
                loss=F.binary_cross_entropy_with_logits(logits,torch.as_tensor(y,device=device))
                if up.any():loss=loss+.2*F.cross_entropy(next_logits[torch.as_tensor(up,device=device)],torch.as_tensor(next_bin[up],device=device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite prediction loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            rate=.1+.9*.5*(1+np.cos(np.pi*step/steps))
            for item in optimizer.param_groups:item['lr']=item['initial_lr']*rate
            if step%100==0:
                write(folder/'status.json',dict(state='training',family=family,step=step,target=steps,
                    loss=float(loss.detach()),best_selection_score=best,wall_seconds=time.monotonic()-started,
                    updated_at=now(),system_sleep_inhibited=held))
            if step%250==0 or step==steps:
                score,metrics=validate(model,selection,validation_positions,use_energy,device)
                history.append(dict(step=step,score=score,metrics=metrics,loss=float(loss.detach())))
                improved=score>best+1e-5
                if improved:best=score;best_metrics=metrics;stale=0
                else:stale+=1
                state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                    step=step,config=config,source_sha256=pins,best=best,best_metrics=best_metrics,stale=stale,history=history,
                    numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    static_dim=data.x.shape[1],width=64)
                atomic_checkpoint(folder/'latest.pt',state)
                if improved:atomic_checkpoint(folder/'best.pt',state)
                write(folder/'history.json',history)
                print(json.dumps(dict(step=step,selection_score=score,best=best,metrics=metrics)),flush=True)
                if stale>=6 and step>=1500:break
        for name,expected in pins.items():
            if sha(ROOT/name)!=expected:raise ValueError('Source changed during training')
        selected_step=int(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['step'])
        result=dict(family=family,tag=tag,use_energy=use_energy,use_plan=bool(data.manifest['contract'].get('planning',False)),selected_step=selected_step,checkpoint=str(folder/'best.pt'),sha256=sha(folder/'best.pt'),
                    samples=str(Path(samples).resolve()),sample_manifest_sha256=config['samples_manifest_sha256'],
                    source_sha256=pins,selection_score=best,selection=best_metrics,completed_steps=step,
                    wall_seconds=time.monotonic()-started,calibration_used=False,confirmation_used=False,final_test_used=False,created_at=now())
        write(folder/'model.json',result);write(folder/'status.json',dict(state='completed',step=step,best_selection_score=best))
        return result
    except BaseException as error:
        write(folder/'status.json',dict(state='failed',error=repr(error),updated_at=now()));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--no-energy',action='store_true');p.add_argument('--steps',type=int,default=3000);p.add_argument('--resume',action='store_true')
    p.add_argument('--family',choices=['tcn','residual_transformer'],default='tcn');a=p.parse_args()
    result=train(a.samples,a.tag,not a.no_energy,a.steps,a.resume,a.family);print(json.dumps(result['selection'],ensure_ascii=False))
