"""Equal-budget continuations with common initialization and random batches."""
from pathlib import Path
import gc
import math
import time
import numpy as np
import torch
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status,PARENT
from .data import arrays
from forecast_graph.model import ForecastGraph,tensors
from forecast_graph.train import loss_for,infer,scores


def load_ram(records,profile,probe=False):
    oldprobe={r['key']:r['positions'] for r in read(PARENT/'training_plan.json')['probe_indices']}
    parts={}
    for item in records:
        positions=np.asarray(oldprobe[item['key']],int) if probe else None
        a=arrays(item,profile,positions)
        for key,value in a.items():
            if key!='query':parts.setdefault(key,[]).append(value)
    return {k:np.concatenate(v) for k,v in parts.items()}


def train_one(profile,seed,initial,train,probe):
    p=verify();folder=ARTIFACTS/'models'/f'v12_{profile}_s{seed}';folder.mkdir(parents=True,exist_ok=True);manifest=folder/'model.json'
    pins={name:sha(ROOT/name) for name in ('train.py','data.py','preview.py','wire.py')}
    if manifest.exists():
        if read(manifest)['source_pins']!=pins:raise ValueError('Existing continuation changed')
        return manifest
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    net=ForecastGraph('hybrid_shared').cuda();warm=read(initial)
    net.load_state_dict(torch.load(warm['checkpoint'],map_location='cuda',weights_only=False)['model'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=.0001)
    latest=folder/'latest.pt';history=[];start=0
    if latest.exists():
        state=torch.load(latest,map_location='cuda',weights_only=False)
        if state['source_pins']!=pins:raise ValueError('Resume implementation changed')
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng'];start=state['step'];best=state['best'];best_step=state['best_step'];history=state['history']
    else:
        a,b=infer(net,probe,'hybrid_shared');score=scores(probe,a,b);best=score['score'];best_step=0
        history.append(dict(step=0,score=score));torch.save(dict(model=net.state_dict(),variant='hybrid_shared',step=0,seed=seed),folder/'best.pt')
    for step in range(start+1,p['updates']+1):
        net.train();ix=rng.integers(0,len(train['n']),p['batch_size']);rate=p['learning_rate']*min(1,step/250)*(.05+.95*(1+math.cos(math.pi*step/p['updates']))/2)
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=net(**tensors(train,ix,'hybrid_shared'));loss,terms=loss_for(out,train,ix,True,'cuda')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite continuation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if step%p['validate_every']==0 or step==p['updates']:
            a,b=infer(net,probe,'hybrid_shared');score=scores(probe,a,b);history.append(dict(step=step,score=score,loss=terms,lr=rate))
            if score['score']>best:
                best=score['score'];best_step=step;torch.save(dict(model=net.state_dict(),variant='hybrid_shared',step=step,seed=seed),folder/'best.pt')
            torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,best_step=best_step,history=history,rng=rng.bit_generator.state,source_pins=pins),latest)
            write(folder/'history.json',dict(history=history,profile=profile,initial=str(initial),selection_only=True))
            status('controlled_plan_continuation',profile=profile,seed=seed,step=step,total=p['updates'],best_step=best_step)
            print(profile,seed,step,{k:round(v['ap'],5) for k,v in score.items() if isinstance(v,dict)},'best',best_step,flush=True)
    if pins!={name:sha(ROOT/name) for name in pins}:raise ValueError('Training source changed midfit')
    cp=folder/'best.pt'
    write(manifest,dict(family='v11_graph',tag=folder.name,variant='hybrid_shared',profile=profile,checkpoint=str(cp),checkpoint_sha256=sha(cp),
        seed=seed,initial_model=str(initial),initial_checkpoint_sha256=warm['checkpoint_sha256'],selected_step=best_step,total_additional_updates=p['updates'],
        selection_score=best,source_pins=pins,created_at=now(),calibration_used=False,confirmation_used=False))
    del net,optimizer;torch.cuda.empty_cache();return manifest


def main():
    p=verify();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    records=read(ARTIFACTS/'data_manifest.json')['records'];all_models=[]
    write(ARTIFACTS/'training_plan.json',dict(created_at=now(),initialization='Per-seed V11hybrid best; same for both profiles',
        checkpoint='Initial0 and every500 up to6000; selection mean fourAPs',probe_indices_sha256=sha(PARENT/'training_plan.json'),
        profile_pairing='Same labels, query ordering, initialization and sampled batches; only plan view changes',
        source_pins={name:sha(ROOT/name) for name in ('train.py','data.py','preview.py','wire.py')},confirmation_used=False))
    for profile in p['profiles']:
        tr=load_ram([i for i in records if i['role']=='train'],profile);val=load_ram([i for i in records if i['role']=='selection'],profile,True)
        members=[]
        for seed,oldseed in zip(p['seeds'],(2026110204,2026110214)):
            initial=PARENT/'models'/f'v11_hybrid_shared_s{oldseed}'/'model.json'
            members.append(train_one(profile,seed,initial,tr,val))
        folder=ARTIFACTS/'models'/('v12_'+profile);folder.mkdir(exist_ok=True);path=folder/'model.json'
        if not path.exists():write(path,dict(family='v11_graph_ensemble',tag=folder.name,variant='hybrid_shared',profile=profile,
            members=[dict(path=str(m),sha256=sha(m),weight=.5) for m in members],created_at=now()))
        all_models.append(str(path));del tr,val;gc.collect()
    write(ARTIFACTS/'trained_models.json',dict(models=all_models,new_models=4,additional_updates_per_model=p['updates'],created_at=now()))


if __name__=='__main__':main()
