"""Controlled 5000->20000 continuation; separate artifacts and LR plan."""
from pathlib import Path
import argparse
import ctypes
import json
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import ARTIFACTS
from .contracts import prepare,verify_protocol,read,write,sha,now,pin,INITIAL,OLD_SAMPLES,SEED
from .samples import CombinedSamples,make_probe
from forecast_rebuild.dataset import Samples
from forecast_rebuild.models import ResidualTransformer
from forecast_rebuild.inference import coherent
from forecast_rebuild.metrics import binary


def rates(additional,total=15000):
    if additional<=250:factor=.1+.9*additional/250
    else:factor=.1+.9*.5*(1+np.cos(np.pi*(additional-250)/(total-250)))
    return [base*factor for base in (3e-6,1e-5,1e-4)]


def save(path,value):
    temporary=Path(str(path)+'.working');torch.save(value,temporary);os.replace(temporary,path)


def validate(model,data,positions):
    model.eval();probabilities=[];truth=[];live=[];next_truth=[];losses=[]
    with torch.inference_mode():
        for start in range(0,len(positions),512):
            seq,static,y,up,nxt=data.neural_batch(positions[start:start+512])
            a,b=model(torch.as_tensor(seq,device='cuda'),torch.as_tensor(static,device='cuda'))
            probabilities.append(coherent(torch.sigmoid(a).cpu().numpy()));truth.append(y);live.append(up);next_truth.append(nxt)
            losses.append((float(F.binary_cross_entropy_with_logits(a,torch.as_tensor(y,device='cuda'))),len(y)))
    p=np.concatenate(probabilities);y=np.concatenate(truth);up=np.concatenate(live)
    # Probe rows have common legacy support. The training probe reports its
    # wider support separately and never selects a checkpoint by training AP.
    result=dict(next5=binary(y[up,2],p[up,2]),any3to7=binary(y[:,5],p[:,5]),
                bce=sum(a*n for a,n in losses)/sum(n for _,n in losses))
    result['score']=(result['next5']['ap']+result['any3to7']['ap'])/2
    return result


def train(tag='same_data_20000',new_samples=None,resume=False,smoke_steps=None):
    protocol=prepare();torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    folder=ARTIFACTS/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.json').exists():return read(folder/'model.json')
    data=CombinedSamples(new_folder=new_samples);probe=Samples(make_probe(),'selection')
    pins=pin(['train.py','samples.py','contracts.py'])
    model=ResidualTransformer(data.dimension,64,False).cuda()
    ids={id(p) for p in list(model.backbone.parameters())+list(model.timing.parameters())}
    groups=[dict(params=[p for p in model.backbone.parameters() if p.requires_grad]),dict(params=list(model.timing.parameters())),
            dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in ids])]
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    state=torch.load(folder/'latest.pt' if resume else INITIAL,map_location='cpu',weights_only=False)
    if resume and (state['v9_source_sha256']!=pins or state['v9_new_samples']!=str(new_samples)):raise ValueError('Resume contract changed')
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    for item in opt.state.values():
        for key,value in item.items():
            if torch.is_tensor(value):item[key]=value if key=='step' else value.cuda()
    rng=np.random.default_rng(SEED)
    if new_samples is None or resume:rng.bit_generator.state=state['numpy_rng']
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    start=int(state['step']);target=20000 if smoke_steps is None else start+smoke_steps
    selection_positions=np.arange(len(probe.y))
    # Preserve exactly the same fixed probe membership while grouping disk
    # reads by episode. Random order thrashes Windows mmap handles/cache.
    training_positions=np.sort(np.random.default_rng(SEED+31).choice(data.n,min(12000,data.n),replace=False))
    history=state['v9_history'] if resume else [];best=state['v9_best'] if resume else -1.;best_metrics=None
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));clock=time.monotonic()
    def checkpoint(step,metrics):
        return dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),step=step,
            static_dim=data.dimension,width=64,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            v9_history=history,v9_best=best,v9_source_sha256=pins,v9_new_samples=str(new_samples),v9_protocol_sha256=sha(ARTIFACTS/'protocol.json'),
            selection=metrics,initial_checkpoint_sha256=sha(INITIAL))
    try:
        if not resume:
            metrics=validate(model,probe,selection_positions);best=metrics['score'];best_metrics=metrics
            history.append(dict(step=start,selection=metrics,training=validate(model,data,training_positions)))
            save(folder/'best.pt',checkpoint(start,metrics));write(folder/'history.json',history)
        for step in range(start+1,target+1):
            positions=data.draw(rng);seq,static,y,up,nxt=data.neural_batch(positions)
            model.train();opt.zero_grad(set_to_none=True)
            for item,lr in zip(opt.param_groups,rates(step-5000)):item['lr']=lr
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a,b=model(torch.as_tensor(seq,device='cuda'),torch.as_tensor(static,device='cuda'))
                loss=F.binary_cross_entropy_with_logits(a,torch.as_tensor(y,device='cuda'))
                if up.any():loss=loss+.2*F.cross_entropy(b[torch.as_tensor(up,device='cuda')],torch.as_tensor(nxt[up],device='cuda'))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            opt.step()
            if step%100==0:write(folder/'status.json',dict(state='training',step=step,target=target,additional_updates=step-5000,
                train_samples=data.n,best_selection_score=best,loss=float(loss.detach()),lrs=rates(step-5000),updated_at=now(),wall_seconds=time.monotonic()-clock))
            if step%1000==0 or step==target:
                metrics=validate(model,probe,selection_positions);training=validate(model,data,training_positions)
                improved=metrics['score']>best+1e-5
                if improved:best=metrics['score'];best_metrics=metrics
                history.append(dict(step=step,selection=metrics,training=training));write(folder/'history.json',history)
                value=checkpoint(step,metrics);save(folder/'latest.pt',value)
                if improved:save(folder/'best.pt',value)
                if step in (10000,15000,20000):save(folder/f's{step:07d}.pt',value)
                print(json.dumps(dict(step=step,selection=metrics,training=training,best=best)),flush=True)
        verify_protocol()
        if pin(list(pins))!=pins:raise ValueError('Trainer source changed during optimization')
        chosen=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        result=dict(family='residual_transformer',tag=tag,use_energy=True,use_plan=True,checkpoint=str(folder/'best.pt'),sha256=sha(folder/'best.pt'),
            selected_step=int(chosen['step']),completed_steps=step,initial_step=5000,initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
            selection_score=best,selection=chosen['selection'],probe_manifest_sha256=sha(ARTIFACTS/'samples/balanced_probe/manifest.json'),
            old_samples=str(OLD_SAMPLES),new_samples=str(new_samples),source_sha256=pins,wall_seconds=time.monotonic()-clock,
            calibration_used=False,confirmation_used=False,final_test_used=False,smoke_only=smoke_steps is not None,created_at=now())
        write(folder/'model.json',result);write(folder/'status.json',dict(state='completed',step=step,selected_step=chosen['step'],best_selection_score=best))
        return result
    except BaseException as error:
        write(folder/'status.json',dict(state='failed',error=repr(error),updated_at=now()));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',default='same_data_20000');p.add_argument('--new-samples');p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int)
    a=p.parse_args();print(train(a.tag,a.new_samples,a.resume,a.smoke_steps))
