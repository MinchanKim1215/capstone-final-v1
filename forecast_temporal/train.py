"""Controlled complete-event end-to-end fine-tuning; common batches and starts."""
import gc
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import ROOT,ARTIFACTS,NAV
from .contracts import read,write,sha,now,verify,status
from .model import TemporalGraph
from .metrics import evaluate_rows
from forecast_recall.episodes import cohort

def load(records):
    parts={k:[] for k in ('node','edge','physics','global_features','index','y','up','support','uniform','bags','logits','n','ns','observer')};cases=[];nr=0;nf=0
    for item in records:
        with np.load(item['path'],allow_pickle=False) as a:
            for k in parts:
                v=a[k]
                if k=='index':v=v+nf
                if k=='bags':v=np.where(v>=0,v+nr,-1)
                parts[k].append(v)
            cases.append((cohort(item),slice(nr,nr+len(a['ends']))));nr+=len(a['ends']);nf+=len(a['node'])
    out={k:np.concatenate(v) for k,v in parts.items()};out['cases']=cases;return out

class DeviceData:
    def __init__(self,a,device='cuda'):
        self.device=device;self.cases=a['cases'];self.uniform=np.flatnonzero(a['uniform']);self.bags=a['bags'];self.n=len(a['y'])
        self.a={}
        for k in ('node','edge','physics','global_features','index','y','up','support','logits','n','ns','observer'):
            dtype=torch.long if k in ('index','n','ns','observer') else torch.bool if k in ('up','support') else torch.float32 if k in ('logits','y') else torch.float16
            self.a[k]=torch.as_tensor(a[k],device=device,dtype=dtype)

    def inputs(self,ix):
        a=self.a;j=a['index'][ix]
        return dict(node=a['node'][j].float(),edge=a['edge'][j].float(),**{k:a[k][ix].float() if k in ('physics','global_features') else a[k][ix] for k in ('physics','global_features','n','ns','observer')})

def infer(net,data,batch=96):
    out=[];net.eval()
    with torch.inference_mode():
        for start in range(0,data.n,batch):
            ix=torch.arange(start,min(start+batch,data.n),device=data.device)
            out.append(torch.sigmoid(net(**data.inputs(ix))['service']).cpu().numpy())
    return np.concatenate(out)

def assess(net,data):
    prediction=infer(net,data)
    return evaluate_rows([(ep,prediction[s][:,None,None]) for ep,s in data.cases])

def loss(net,data,uniform,bags,member):
    mask=bags>=0;ix=torch.as_tensor(np.r_[uniform,np.maximum(bags,0).ravel()],device=data.device);n=len(uniform);a=data.a
    out=net(**data.inputs(ix));z=out['service'];u=ix[:n]
    valid=a['support'][u,None].expand(-1,6).clone();valid[:,2]&=a['up'][u]
    element=F.binary_cross_entropy_with_logits(z[:n],a['y'][u],reduction='none');bce=(element*valid).sum()/valid.sum().clamp_min(1)
    event=bce*0;constraint=bce*0
    if net.temporal_variant!='short_bce':
        bm=torch.as_tensor(mask,device=data.device);bag=z[n:,5].reshape(len(bags),4).float()
        smooth=.25*(torch.logsumexp(bag.masked_fill(~bm,-1e4)/.25,1)-bm.sum(1).clamp_min(1).float().log())
        event=F.softplus(-smooth).mean()
        p=torch.sigmoid(z[:n,5].float()/.5);y=a['y'][u,5];v=a['support'][u]
        tp=(p*y*v).sum();fp=(p*(1-y)*v).sum();constraint=F.relu(.4*fp-.6*tp)/((y*v).sum()+1)
    anchor=.005*((z[:n].float()-a['logits'][u,member]).square()*valid).sum()/valid.sum().clamp_min(1)
    total=bce+.5*event+constraint+anchor
    return total,dict(bce=float(bce.detach()),event=float(event.detach()),precision_surrogate=float(constraint.detach()),anchor=float(anchor.detach()))

def initial(member):return read(NAV/'models/v12_mission/model.json')['members'][member]['path']

def initialize(variant,member,device='cuda'):
    net=TemporalGraph(variant).to(device);m=read(initial(member))
    if sha(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('Warm weights changed')
    net.load_state_dict(torch.load(m['checkpoint'],map_location=device,weights_only=False)['model']);return net

def train_one(variant,seed,member,tr,val):
    p=verify();folder=ARTIFACTS/'models'/f'v14_{variant}_s{seed}';folder.mkdir(parents=True,exist_ok=True);meta=folder/'model.json'
    pins={k:sha(ROOT/k) for k in ('train.py','model.py','metrics.py','features.py','data.py')}
    if meta.exists():
        if read(meta)['source_pins']!=pins:raise ValueError('Completed fit changed')
        return meta
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);net=initialize(variant,member)
    optimizer=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=.0001);latest=folder/'latest.pt';checkpoint=folder/'best.pt';start=0;began=time.monotonic()
    if latest.exists():
        s=torch.load(latest,map_location='cuda',weights_only=False)
        if s['source_pins']!=pins:raise ValueError('Resume implementation changed')
        net.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);rng.bit_generator.state=s['rng'];start=s['step'];best=tuple(s['best']);best_step=s['best_step'];history=s['history']
        torch.set_rng_state(s['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    else:
        score=assess(net,val);best=tuple(score['score']);best_step=0;history=[dict(step=0,selection=score)]
        torch.save(dict(model=net.state_dict(),variant=variant,step=0),checkpoint)
    for step in range(start+1,p['updates']+1):
        net.train();u=rng.choice(tr.uniform,size=p['batch_uniform'],replace=True);b=tr.bags[rng.integers(0,len(tr.bags),p['batch_events'])]
        rate=p['learning_rate']*min(1,step/150)*(.05+.95*(1+math.cos(math.pi*step/p['updates']))/2)
        for g in optimizer.param_groups:g['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):value,terms=loss(net,tr,u,b,member)
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
        value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if step%100==0:status('temporal_graph_training',variant=variant,seed=seed,step=step,total=p['updates'],best_step=best_step,seconds=time.monotonic()-began)
        if step%p['validate_every']==0 or step==p['updates']:
            score=assess(net,val);history.append(dict(step=step,selection=score,loss=terms,seconds=time.monotonic()-began))
            if tuple(score['score'])>best:best=tuple(score['score']);best_step=step;torch.save(dict(model=net.state_dict(),variant=variant,step=step),checkpoint)
            torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,best=list(best),best_step=best_step,history=history,rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),source_pins=pins),latest)
            write(folder/'history.json',dict(history=history,selection_only=True));print(variant,seed,step,'best',best_step,score['point'],flush=True)
    if pins!={k:sha(ROOT/k) for k in pins}:raise ValueError('Source changed mid-training')
    write(meta,dict(family='v14_temporal_graph',tag=folder.name,variant=variant,member_index=member,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        initial_model=str(initial(member)),seed=seed,selected_step=best_step,updates=p['updates'],selection_score=list(best),source_pins=pins,created_at=now(),confirmation_used=False))
    del net,optimizer;torch.cuda.empty_cache();return meta

def main():
    p=verify();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    records=read(ARTIFACTS/'data_manifest.json')['records'];train=load([i for i in records if i['role']=='train']);val=load([i for i in records if i['role']=='selection'])
    plan=dict(data_manifest_sha256=sha(ARTIFACTS/'data_manifest.json'),protocol_sha256=sha(ARTIFACTS/'protocol.json'),source_pins={k:sha(ROOT/k) for k in ('train.py','model.py','metrics.py','data.py','features.py')},
        training_queries=len(train['y']),uniform_training_queries=int(train['uniform'].sum()),event_bags=len(train['bags']),selection_queries=len(val['y']))
    target=ARTIFACTS/'training_plan.json'
    if target.exists() and read(target)!=plan:raise ValueError('Plan changed')
    if not target.exists():write(target,plan)
    tr=DeviceData(train);va=DeviceData(val);del train,val;gc.collect();members={v:[] for v in p['variants']}
    for member,seed in enumerate(p['seeds']):
        for variant in p['variants']:members[variant].append(train_one(variant,seed,member,tr,va))
    models=[]
    for variant,paths in members.items():
        path=ARTIFACTS/'models'/('v14_'+variant)/'model.json';path.parent.mkdir(exist_ok=True)
        if not path.exists():write(path,dict(family='v14_temporal_ensemble',variant=variant,tag=path.parent.name,members=[dict(path=str(x),sha256=sha(x),weight=.5) for x in paths],created_at=now()))
        models.append(str(path))
    write(ARTIFACTS/'trained_models.json',dict(models=models,new_models=6,created_at=now()))

if __name__=='__main__':main()
