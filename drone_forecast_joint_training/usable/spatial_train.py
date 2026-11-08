"""검증 자료에서 확인한 공간 선행 정보 부족에 대한 추가 비교. 시험 미사용."""
import argparse,json,time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from upgrade.map_setup import digest
from upgrade.train import configure,lr_at,macro_ap
from usable.data import Bank,PREPARED
from usable.model import RiskTransformer
from usable.train import RUNS,CAUSAL_FEATURES,predict_bank

FEATURES=CAUSAL_FEATURES+list(range(48,60));VARIANT='B3_beacon_map_multitask'


class SpatialBank(Bank):
    def __init__(self,name,features=FEATURES):
        super().__init__(name,list(range(48)))
        self.spatial_values=np.load(PREPARED/name/'spatial_values.npy',mmap_mode='r')
        self.spatial_mask=np.load(PREPARED/name/'spatial_mask.npy',mmap_mode='r')
        self.features=np.asarray(features,int);norm=read_json(PREPARED/'normalizer_spatial.json')
        self.mean=np.asarray(norm['mean'],np.float32)[self.features];self.std=np.asarray(norm['std'],np.float32)[self.features]
    def batch(self,idx):
        x=np.concatenate((self.values[idx],self.spatial_values[idx]),axis=-1)[:,:,self.features]
        m=np.concatenate((self.mask[idx],self.spatial_mask[idx]),axis=-1)[:,:,self.features]
        x=np.where(m,np.clip((x-self.mean)/self.std,-8,8),0).astype(np.float32)
        return torch.from_numpy(x),torch.from_numpy(m),torch.from_numpy(self.selected[idx,4:7].astype(np.float32))


def plan():
    path=ROOT/'configs/usable_spatial_training.json'
    out={'variant':VARIANT,'features':FEATURES,'updates':4000,'batch_size':256,'learning_rate':.0003,'weight_decay':.0001,
         'validation_every':250,'patience':5,'minimum_updates':1500,
         'model':{'d_model':64,'nhead':4,'layers':2,'feedforward':128,'dropout':.1,'length':40},
         'multitask':True,'auxiliary_loss':'BCE5 + 0.3*BCE10 + 0.1*Huber(normalized future minimum SINR)',
         'selection_metric':'validation-select pooled AP at 5s; final comparison against B0/B1/B2',
         'reason':'validation-only diagnosis: signal levels cannot anticipate nearby geometric shielding without fresh peer telemetry and static map extrapolation',
         'test_predictions_computed':False,'first_seed':11,'replications':[22,33],
         'code_sha256':{n:sha256_file(ROOT/n) for n in ('usable/spatial.py','usable/spatial_train.py')}}
    out['plan_sha256']=digest(out)
    if path.exists() and read_json(path)!=out:raise ValueError('추가 비교 계획 동결 위반')
    write_json(path,out);return out


def train(seed=11):
    conf=plan();folder=RUNS/f'{VARIANT}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():return read_json(folder/'run.json')
    configure(seed,2);training=SpatialBank('train30');val=SpatialBank('val_select')
    model=RiskTransformer(features=len(FEATURES),multitask=True,**conf['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=conf['learning_rate'],weight_decay=conf['weight_decay'])
    rng=np.random.default_rng(seed+2401);order=np.empty(0,int);position=0;best=-1.;stale=0;history=[];cumulative=0.;started=time.perf_counter()
    for step in range(conf['updates']):
        if position+conf['batch_size']>len(order):order=rng.permutation(len(training));position=0
        idx=order[position:position+conf['batch_size']];position+=len(idx);x,m,y=training.batch(idx)
        model.train();opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=lr_at(step,conf['updates'],conf['learning_rate'])
        z=model(x,m);loss=torch.nn.functional.binary_cross_entropy_with_logits(z[:,0],y[:,0])+.3*torch.nn.functional.binary_cross_entropy_with_logits(z[:,1],y[:,1])+.1*torch.nn.functional.smooth_l1_loss(z[:,2],y[:,2])
        if not torch.isfinite(loss):raise RuntimeError('학습 손실 오류')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();cumulative+=float(loss.detach())
        if (step+1)%250==0:
            logits=predict_bank(model,val);prob=torch.sigmoid(torch.from_numpy(logits[:,0])).numpy();ap=float(average_precision_score(val.selected[:,4],prob));macro,by=macro_ap(val.selected[:,4],prob,val.region)
            row={'step':step+1,'loss':cumulative/250,'validation_pooled_ap':ap,'validation_macro_ap':macro,'by_region':by,'elapsed_s':time.perf_counter()-started};history.append(row);cumulative=0
            if ap>best+1e-6:
                best=ap;stale=0;best_step=step+1
                torch.save({'state_dict':model.state_dict(),'spec':model.spec,'feature_indices':FEATURES,
                            'normalizer':read_json(PREPARED/'normalizer_spatial.json'),'variant':VARIANT,'seed':seed,'step':best_step,
                            'plan_sha256':conf['plan_sha256']},folder/'best.pt');np.save(folder/'val_select_logits.npy',logits)
            else:stale+=1
            write_json(folder/'history.json',history);print(json.dumps({'run':folder.name,'step':step+1,'val_ap':round(ap,5),'macro':round(macro,5),'best_step':best_step,'elapsed_s':round(time.perf_counter()-started,1)}),flush=True)
            if step+1>=conf['minimum_updates'] and stale>=conf['patience']:break
    out={'variant':VARIANT,'seed':seed,'best_step':best_step,'updates':step+1,'validation_select_pooled_ap':best,
         'parameters':sum(p.numel() for p in model.parameters()),'training_samples':len(training),'test_used':False,
         'elapsed_s':time.perf_counter()-started,'plan_sha256':conf['plan_sha256'],'checkpoint_sha256':sha256_file(folder/'best.pt')}
    write_json(folder/'run.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=11);a=p.parse_args();train(a.seed)
