"""Controlled residual fits: natural BCE, event objective, added20s memory."""
import gc
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status
from .features import OFFSETS
from .model import EventAdapter
from .episodes import cohort
from .metrics import evaluate_rows


def load(records):
    parts={k:[] for k in ('embedding','logits','y','up','support','uniform','history','ends','bags')};cases=[];row_offset=0;frame_offset=0
    for item in records:
        with np.load(item['path'],allow_pickle=False) as a:
            for k in ('embedding','logits','y','up','support','uniform','history'):parts[k].append(a[k])
            parts['ends'].append(a['ends']+frame_offset)
            bags=a['bags'];parts['bags'].append(np.where(bags>=0,bags+row_offset,-1))
            cases.append((cohort(item),slice(row_offset,row_offset+item['queries'])))
            row_offset+=len(a['ends']);frame_offset+=len(a['history'])
    result={k:np.concatenate(v) for k,v in parts.items()};result['cases']=cases
    return result


class DeviceData:
    def __init__(self,a,member,device='cuda'):
        self.device=device;self.cases=a['cases'];self.uniform=np.flatnonzero(a['uniform']);self.bags=a['bags'];self.n=len(a['ends'])
        self.a={key:torch.as_tensor(a[key],device=device,dtype=torch.bool if key in ('support','up') else torch.long if key=='ends' else torch.float16 if key=='history' else torch.float32)
                for key in ('y','up','support','ends','history')}
        self.a['embedding']=torch.as_tensor(a['embedding'][:,member],device=device);self.a['logits']=torch.as_tensor(a['logits'][:,member],device=device)
        self.offset=torch.as_tensor(OFFSETS,device=device)

    def inputs(self,ix,memory):
        x=self.a;history=x['history'][x['ends'][ix,None]+self.offset[None]].float() if memory else None
        return x['embedding'][ix],x['logits'][ix],history


def infer(model,data,batch=256):
    model.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,data.n,batch):
            ix=torch.arange(start,min(start+batch,data.n),device=data.device)
            p,_=model(*data.inputs(ix,model.variant=='memory_event'));parts.append(torch.sigmoid(p).cpu().numpy())
    return np.concatenate(parts)


def assess(model,data):
    prediction=infer(model,data)
    return evaluate_rows([(ep,prediction[s][:,None,None]) for ep,s in data.cases])


def loss(model,data,uniform,bags,variant):
    device=data.device;mask=bags>=0;flat=np.maximum(bags,0).ravel();rows=np.r_[uniform,flat];ix=torch.as_tensor(rows,device=device)
    logits,residual=model(*data.inputs(ix,variant=='memory_event'));n=len(uniform);ui=ix[:n];a=data.a
    valid=a['support'][ui,None].expand(-1,6).clone();valid[:,2]&=a['up'][ui]
    element=F.binary_cross_entropy_with_logits(logits[:n],a['y'][ui],reduction='none');bce=(element*valid).sum()/valid.sum().clamp_min(1)
    event=bce*0;constraint=bce*0
    if variant!='bce':
        bag=logits[n:,5].reshape(len(bags),4).float();bm=torch.as_tensor(mask,device=device)
        smooth=.25*(torch.logsumexp(bag.masked_fill(~bm,-1e4)/.25,1)-bm.sum(1).clamp_min(1).float().log())
        event=F.softplus(-smooth).mean()
        p=torch.sigmoid(logits[:n,5].float()/.5);y=a['y'][ui,5];v=a['support'][ui]
        tp=(p*y*v).sum();fp=(p*(1-y)*v).sum();constraint=F.relu(.9*fp-.1*tp)/((y*v).sum()+1)
    regularizer=.001*residual.square().mean();total=bce+.5*event+4*constraint+regularizer
    return total,dict(bce=float(bce.detach()),event=float(event.detach()),precision_surrogate=float(constraint.detach()),regularizer=float(regularizer.detach()))


def train_one(variant,seed,tr,val,member):
    p=verify();folder=ARTIFACTS/'models'/f'v13_{variant}_s{seed}';folder.mkdir(parents=True,exist_ok=True);meta=folder/'model.json'
    pins={name:sha(ROOT/name) for name in ('train.py','model.py','metrics.py','features.py','data.py','embedding.py')}
    if meta.exists():
        if read(meta)['source_pins']!=pins:raise ValueError('Completed fit changed')
        return meta
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    net=EventAdapter(variant).cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=.0001)
    score=assess(net,val);best=tuple(score['score']);best_step=0;history=[dict(step=0,selection=score)];start=0
    checkpoint=folder/'best.pt';latest=folder/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cuda',weights_only=False)
        if state['source_pins']!=pins:raise ValueError('Resume sources changed')
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        best=tuple(state['best']);best_step=state['best_step'];history=state['history'];start=state['step']
    else:torch.save(dict(model=net.state_dict(),variant=variant,seed=seed,step=0),checkpoint)
    if not len(tr.bags):raise ValueError('Training has no complete event bags')
    for step in range(start+1,p['updates']+1):
        net.train();u=rng.choice(tr.uniform,size=p['batch_uniform'],replace=True);b=tr.bags[rng.integers(0,len(tr.bags),p['batch_events'])]
        rate=p['learning_rate']*min(1,step/200)*(.05+.95*(1+math.cos(math.pi*step/p['updates']))/2)
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):value,terms=loss(net,tr,u,b,variant)
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite event training loss')
        value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if step%p['validate_every']==0 or step==p['updates']:
            score=assess(net,val);history.append(dict(step=step,selection=score,loss=terms))
            if tuple(score['score'])>best:
                best=tuple(score['score']);best_step=step;torch.save(dict(model=net.state_dict(),variant=variant,seed=seed,step=step),checkpoint)
            torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,best=list(best),best_step=best_step,
                history=history,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),source_pins=pins),latest)
            write(folder/'history.json',dict(history=history,selection_only=True));status('event_recall_training',variant=variant,seed=seed,step=step,total=p['updates'],best_step=best_step)
            print(variant,seed,step,'best',best_step,'selection',score['rule'],score['point'],flush=True)
    if pins!={name:sha(ROOT/name) for name in pins}:raise ValueError('Training source changed')
    write(meta,dict(family='v13_event_adapter',tag=folder.name,variant=variant,member_index=member,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        seed=seed,selected_step=best_step,updates=p['updates'],selection_score=list(best),source_pins=pins,confirmation_used=False,created_at=now()))
    del net,optimizer;torch.cuda.empty_cache();return meta


def main():
    p=verify();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    items=read(ARTIFACTS/'data_manifest.json')['records'];train=load([i for i in items if i['role']=='train']);selection=load([i for i in items if i['role']=='selection'])
    path=ARTIFACTS/'training_plan.json';plan=dict(data_manifest_sha256=sha(ARTIFACTS/'data_manifest.json'),source_pins={name:sha(ROOT/name) for name in ('train.py','model.py','metrics.py','features.py','data.py','embedding.py')},
        training_queries=len(train['ends']),uniform_training_queries=int(train['uniform'].sum()),event_bags=len(train['bags']),selection_queries=len(selection['ends']),protocol_sha256=sha(ARTIFACTS/'protocol.json'))
    if path.exists():
        if read(path)!=plan:raise ValueError('Training plan changed')
    else:write(path,plan)
    members={v:[] for v in p['variants']}
    for member,seed in enumerate(p['seeds']):
        tr=DeviceData(train,member);val=DeviceData(selection,member)
        for variant in p['variants']:members[variant].append(train_one(variant,seed,tr,val,member))
        del tr,val;gc.collect();torch.cuda.empty_cache()
    paths=[]
    for variant,files in members.items():
        folder=ARTIFACTS/'models'/('v13_'+variant);folder.mkdir(exist_ok=True);meta=folder/'model.json'
        if not meta.exists():write(meta,dict(family='v13_event_ensemble',tag=folder.name,variant=variant,members=[dict(path=str(x),sha256=sha(x),weight=.5) for x in files],created_at=now()))
        paths.append(str(meta))
    write(ARTIFACTS/'trained_models.json',dict(models=paths,new_models=6,created_at=now()))


if __name__=='__main__':main()
