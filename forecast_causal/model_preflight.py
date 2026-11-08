"""Actual tree fits and GPU gradients on a TRAIN-only fixture, no study models."""
import numpy as np
import torch
import lightgbm as lgb
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now
from .model import CausalAdapter
from .inference import columns
from .neural_train import load,DeviceData,loss

PIN_NAMES=('model.py','inference.py','neural_train.py','tree_train.py','metrics.py','features.py')
def pins():return {k:sha(ROOT/k) for k in PIN_NAMES}

def main():
    torch.set_num_threads(2);target=ARTIFACTS/'model_preflight.json';current=pins()
    if target.exists():
        old=read(target)
        if old['passed'] and old['source_pins']==current:return old
    item=read(ARTIFACTS/'data/969533adae14a5e6b101.json');a=load([item]);d=DeviceData(a,0);ix=torch.arange(3,device='cuda');errors={};gradients={}
    for variant in ('gru_base','gru_network'):
        torch.manual_seed(2026110316);net=CausalAdapter(variant).cuda();net.eval()
        with torch.no_grad():z,_=net(*d.inputs(ix));errors[variant]=float((z-d.a['baseline_logits'][ix]).abs().max())
        if errors[variant]!=0:raise ValueError('Initial model changed current predictor')
        net.train();opt=torch.optim.AdamW(net.parameters(),lr=.0003);u=np.resize(d.uniform,256);bags=np.resize(d.bags,(32,4))
        for k in range(3):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):v,_=loss(net,d,u,bags,variant)
            v.backward()
            if not torch.isfinite(v) or any(not torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):raise ValueError('Invalid GRU gradient')
            opt.step()
        gradients[variant]=dict(rf=float(next(net.rf_memory.parameters()).grad.abs().sum()),network=None if variant=='gru_base' else float(next(net.network_memory.parameters()).grad.abs().sum()))
        if gradients[variant]['rf']<=0 or (variant=='gru_network' and gradients[variant]['network']<=0):raise ValueError('Observed histories not learning')
        del net,opt;torch.cuda.empty_cache()
    with np.load(item['path']) as z:raw={k:z[k] for k in ('base','network','physics','y')}
    tree_dimensions={}
    for variant in ('tree_base','tree_network','tree_physics','tree_both'):
        x=columns(raw,variant);model=lgb.train(dict(objective='binary',num_leaves=7,min_data_in_leaf=10,num_threads=2,verbosity=-1,seed=2026110316),lgb.Dataset(x,label=raw['y'][:,5]),num_boost_round=12);p=model.predict(x,num_threads=2)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid tree probability')
        tree_dimensions[variant]=x.shape[1]
    if tree_dimensions!=dict(tree_base=432,tree_network=1548,tree_physics=486,tree_both=1602):raise ValueError('The ablations do not share the same432observed base fields')
    value=dict(passed=True,created_at=now(),training_fixture_only=True,not_in_model_catalogue=True,zero_residual_logit_error=errors,actual_GPU_gradients=gradients,actual_LightGBM_dimensions=tree_dimensions,source_pins=current)
    write(target,value);print(value,flush=True);return value

if __name__=='__main__':main()
