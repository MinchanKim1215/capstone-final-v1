"""시험 지역을 읽지 않는 학습. 검증 비행 절반은 선택, 절반은 보정 전용."""
import argparse,json,time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from upgrade.train import configure,lr_at,macro_ap
from upgrade.map_setup import digest
from usable.model import RiskTransformer
from usable.data import Bank,PREPARED

RUNS=ROOT/'results/usable/runs'
CAUSAL_FEATURES=[i for i in range(48) if i not in (12,13,23,25,26)]
VARIANTS={
 'B0_legacy_inputs':{'features':list(range(28)),'multitask':False},
 'B1_observed_radio':{'features':CAUSAL_FEATURES,'multitask':False},
 'B2_multitask':{'features':CAUSAL_FEATURES,'multitask':True},
}


def plan():
    path=ROOT/'configs/usable_training.json'
    spec={'variants':VARIANTS,'updates':4000,'batch_size':256,'learning_rate':.0003,'weight_decay':.0001,
          'validation_every':250,'patience':5,'minimum_updates':1500,'selection':'highest validation-select pooled AP at 5s',
          'first_seed':11,'replications':[22,33],'model':{'d_model':64,'nhead':4,'layers':2,'feedforward':128,'dropout':.1,'length':40},
          'auxiliary_loss':'BCE5 + 0.3*BCE10 + 0.1*Huber(normalized minimum future SINR)',
          'protocol_sha256':read_json(ROOT/'configs/usable_protocol.json')['protocol_sha256'],
          'code_sha256':{n:sha256_file(ROOT/n) for n in ('usable/data.py','usable/model.py','usable/train.py')}}
    spec['plan_sha256']=digest(spec)
    if path.exists():
        old=read_json(path)
        if old!=spec:raise ValueError('동결한 학습 계획과 코드 불일치')
    else:write_json(path,spec)
    return spec


@torch.inference_mode()
def predict_bank(model,bank,batch=512):
    model.eval();out=np.empty((len(bank),3 if model.multitask else 1),np.float32)
    for first in range(0,len(bank),batch):
        x,m,_=bank.batch(np.arange(first,min(first+batch,len(bank))))
        out[first:first+len(x)]=model(x,m).numpy()
    return out


def train(variant,seed):
    conf=plan();s=conf['variants'][variant];dest=RUNS/f'{variant}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'run.json').exists():return read_json(dest/'run.json')
    configure(seed,2);training=Bank('train30',s['features']);val=Bank('val_select',s['features'])
    model=RiskTransformer(features=len(s['features']),multitask=s['multitask'],**conf['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=conf['learning_rate'],weight_decay=conf['weight_decay'])
    rng=np.random.default_rng(seed+2401);order=np.empty(0,int);position=0;best=-1;stale=0;history=[];started=time.perf_counter();cumulative=0
    for step in range(conf['updates']):
        if position+conf['batch_size']>len(order):order=rng.permutation(len(training));position=0
        idx=order[position:position+conf['batch_size']];position+=len(idx)
        x,m,y=training.batch(idx);model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=lr_at(step,conf['updates'],conf['learning_rate'])
        output=model(x,m);loss=torch.nn.functional.binary_cross_entropy_with_logits(output[:,0],y[:,0])
        if s['multitask']:
            loss=loss+.3*torch.nn.functional.binary_cross_entropy_with_logits(output[:,1],y[:,1])+.1*torch.nn.functional.smooth_l1_loss(output[:,2],y[:,2])
        if not torch.isfinite(loss):raise RuntimeError('학습 손실 비정상')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();cumulative+=float(loss.detach())
        if (step+1)%conf['validation_every']==0:
            logits=predict_bank(model,val);prob=torch.sigmoid(torch.from_numpy(logits[:,0])).numpy()
            ap=float(average_precision_score(val.selected[:,4],prob));ma,by=macro_ap(val.selected[:,4],prob,val.region)
            row={'step':step+1,'loss':cumulative/conf['validation_every'],'validation_pooled_ap':ap,'validation_macro_ap':ma,'by_region':by,'elapsed_s':time.perf_counter()-started};history.append(row);cumulative=0
            if ap>best+1e-6:
                best=ap;stale=0;best_step=step+1
                torch.save({'state_dict':model.state_dict(),'spec':model.spec,'feature_indices':s['features'],
                            'normalizer':read_json(PREPARED/'normalizer.json'),'variant':variant,'seed':seed,
                            'step':best_step,'plan_sha256':conf['plan_sha256']},dest/'best.pt')
                np.save(dest/'val_select_logits.npy',logits)
            else:stale+=1
            write_json(dest/'history.json',history)
            print(json.dumps({'run':dest.name,'step':step+1,'val_ap':round(ap,5),'macro':round(ma,5),'best_step':best_step,'elapsed_s':round(time.perf_counter()-started,1)}),flush=True)
            if step+1>=conf['minimum_updates'] and stale>=conf['patience']:break
    result={'variant':variant,'seed':seed,'best_step':best_step,'updates':step+1,'validation_select_pooled_ap':best,
            'elapsed_s':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters()),
            'training_samples':len(training),'test_used':False,'plan_sha256':conf['plan_sha256'],'checkpoint_sha256':sha256_file(dest/'best.pt')}
    write_json(dest/'run.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=list(VARIANTS),required=True);parser.add_argument('--seed',type=int,default=11);a=parser.parse_args();train(a.variant,a.seed)
