"""Fixed-budget two-seed temporal graph fits with selection-only checkpoints."""
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,digest,now,verify,status,SEED
from .data import FEATURE_KEYS,TARGET_KEYS
from .model import ForecastGraph,tensors
from forecast_rebuild.metrics import binary


def load_ram(records,probe=False):
    keys=FEATURE_KEYS+TARGET_KEYS;parts={key:[] for key in keys};indices=[]
    for record in records:
        with np.load(record['path'],allow_pickle=False) as a:
            count=len(a['n']);take=np.arange(count)
            if probe:
                rng=np.random.default_rng(int(digest(['v11-probe',SEED,record['key']])[:16],16))
                take=np.sort(rng.choice(count,min(256,count),replace=False))
            for key in keys:parts[key].append(a[key][take])
            indices.append(dict(key=record['key'],positions=take.tolist()))
    value={key:np.concatenate(part) for key,part in parts.items()}
    return value,indices


def infer(model,arrays,variant,device='cuda',batch=128):
    model.eval();link=[];service=[];aux_error=[]
    with torch.no_grad():
        for start in range(0,len(arrays['n']),batch):
            p=np.arange(start,min(start+batch,len(arrays['n'])))
            out=model(**tensors(arrays,p,variant,device))
            link.append(torch.sigmoid(out['link']).cpu().numpy());service.append(torch.sigmoid(out['service']).cpu().numpy())
    return np.concatenate(link),np.concatenate(service)


def scores(arrays,link,service):
    support=arrays['link_support'];up=arrays['link_up'];ss=arrays['service_support'];su=arrays['service_up']
    result={}
    for name,y,p,m in [('link5',arrays['link_y'][...,2],link[...,2],support&up),
            ('link37',arrays['link_y'][...,5],link[...,5],support),
            ('service5',arrays['service_y'][:,2],service[:,2],ss&su),
            ('service37',arrays['service_y'][:,5],service[:,5],ss)]:result[name]=binary(y[m],p[m])
    result['score']=float(np.mean([r['ap'] for r in result.values() if isinstance(r,dict) and r['ap'] is not None]))
    return result


def loss_for(output,arrays,positions,hybrid,device):
    y=torch.as_tensor(arrays['link_y'][positions],device=device,dtype=torch.float32)
    mask=torch.as_tensor(arrays['link_support'][positions],device=device).unsqueeze(-1).expand_as(y).clone()
    up=torch.as_tensor(arrays['link_up'][positions],device=device);mask[...,2]&=up
    link=F.binary_cross_entropy_with_logits(output['link'],y,reduction='none')
    link=(link*mask).sum()/mask.sum().clamp_min(1)
    sy=torch.as_tensor(arrays['service_y'][positions],device=device,dtype=torch.float32)
    sm=torch.as_tensor(arrays['service_support'][positions],device=device).unsqueeze(-1).expand_as(sy).clone()
    sm[:,2]&=torch.as_tensor(arrays['service_up'][positions],device=device)
    service=F.binary_cross_entropy_with_logits(output['service'],sy,reduction='none')
    service=(service*sm).sum()/sm.sum().clamp_min(1)
    aux=link*0
    if hybrid:
        truth=torch.as_tensor(arrays['aux'][positions],device=device,dtype=torch.float32)
        n=torch.as_tensor(arrays['n'][positions],device=device);k=truth.shape[-1]
        valid=torch.arange(k,device=device)[None,:]<n[:,None]
        am=(valid[:,:,None]&valid[:,None,:])&~torch.eye(k,device=device,dtype=torch.bool)[None]
        terms=F.smooth_l1_loss(output['aux'],truth,reduction='none')
        aux=(terms*am[:,None]).sum()/(am.sum()*truth.shape[1]).clamp_min(1)
    return link+service+.1*aux,dict(link=float(link.detach()),service=float(service.detach()),aux=float(aux.detach()))


def train_one(variant,seed,train,probe,probe_indices):
    p=verify();cfg=p['neural'];folder=ARTIFACTS/'models'/f'v11_{variant}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'model.json';pins={name:sha(ROOT/name) for name in ('model.py','features.py','data.py','train.py')}
    if manifest.exists():
        old=read(manifest)
        if old['source_pins']!=pins:raise ValueError('Trained model implementation changed')
        return manifest
    torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
    net=ForecastGraph(variant).cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);start_step=0;best=-1.;best_step=0;history=[]
    latest=folder/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cuda',weights_only=False)
        if state['source_pins']!=pins:raise ValueError('Resume implementation differs')
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start_step=state['step'];best=state['best'];best_step=state['best_step'];history=state['history'];rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    else:
        with torch.no_grad():
            support=train['link_support'][...,None]
            lp=(train['link_y']*support).sum((0,1))/np.maximum(support.sum((0,1)),1)
            sp=train['service_y'].mean(0)
            net.link.bias.copy_(torch.as_tensor(np.log(np.clip(lp,1e-4,1-1e-4)/(1-np.clip(lp,1e-4,1-1e-4))),device='cuda',dtype=torch.float32))
            net.service[-1].bias.copy_(torch.as_tensor(np.log(np.clip(sp,1e-4,1-1e-4)/(1-np.clip(sp,1e-4,1-1e-4))),device='cuda',dtype=torch.float32))
            net.aux.weight.zero_();net.aux.bias.zero_()
    started=time.monotonic()
    for step in range(start_step+1,cfg['updates']+1):
        net.train();positions=rng.integers(0,len(train['n']),cfg['batch_size'])
        rate=cfg['lr']*min(1,step/250)*(.05+.95*(1+math.cos(math.pi*step/cfg['updates']))/2)
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=net(**tensors(train,positions,variant));loss,losses=loss_for(out,train,positions,net.hybrid,'cuda')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if step%cfg['validate_every']==0 or step==cfg['updates']:
            link,service=infer(net,probe,variant);score=scores(probe,link,service)
            history.append(dict(step=step,score=score,loss=losses,lr=rate,elapsed_s=time.monotonic()-started))
            if score['score']>best:
                best=score['score'];best_step=step
                torch.save(dict(model=net.state_dict(),variant=variant,step=step,seed=seed,source_pins=pins),folder/'best.pt')
            torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,best_step=best_step,history=history,
                source_pins=pins,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),latest)
            write(folder/'history.json',dict(history=history,probe_indices=probe_indices,selection_only=True))
            status('graph_GPU_training',variant=variant,seed=seed,step=step,total=cfg['updates'],best_step=best_step)
            print(variant,seed,step,{k:round(v['ap'],5) for k,v in score.items() if isinstance(v,dict)},'best',best_step,flush=True)
    if pins!={name:sha(ROOT/name) for name in pins}:raise ValueError('Implementation changed during fitting')
    checkpoint=folder/'best.pt'
    write(manifest,dict(family='v11_graph',tag=folder.name,variant=variant,seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        source_pins=pins,selection_score=best,selected_step=best_step,total_updates=cfg['updates'],created_at=now(),
        data_manifest_sha256=sha(ARTIFACTS/'data_manifest.json'),calibration_used=False,confirmation_used=False,
        parameters=sum(v.numel() for v in net.parameters()),production_ready=False))
    del net,optimizer;torch.cuda.empty_cache();return manifest


def main():
    p=verify();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Expected authorized4070SUPER GPU')
    records=read(ARTIFACTS/'data_manifest.json')['records']
    train,_=load_ram([r for r in records if r['role']=='train'])
    probe,probe_indices=load_ram([r for r in records if r['role']=='selection'],True)
    write(ARTIFACTS/'training_plan.json',dict(created_at=now(),train_rows=len(train['n']),probe_rows=len(probe['n']),
        probe_indices=probe_indices,source_pins={name:sha(ROOT/name) for name in ('model.py','features.py','data.py','train.py')},
        objective='Natural-prior BCE for physical/service persistent onsets; hybrid adds0.1 SmoothL1 normalized futureSINR margin',
        checkpoint_score='Mean of linkAP5,linkAP37,serviceAP5,serviceAP37 on fixed selection probe',confirmation_used=False))
    candidates=[]
    for variant in p['variants']:
        members=[train_one(variant,seed,train,probe,probe_indices) for seed in p['seeds']]
        folder=ARTIFACTS/'models'/('v11_'+variant);folder.mkdir(exist_ok=True);manifest=folder/'model.json'
        if not manifest.exists():write(manifest,dict(family='v11_graph_ensemble',tag=folder.name,variant=variant,
            members=[dict(path=str(m),sha256=sha(m),weight=1/len(members)) for m in members],created_at=now()))
        candidates.append(str(manifest))
    write(ARTIFACTS/'trained_models.json',dict(candidates=candidates,created_at=now(),train_rows=len(train['n']),
        probe_rows=len(probe['n']),seeds=p['seeds'],new_neural_models=6))


if __name__=='__main__':main()
