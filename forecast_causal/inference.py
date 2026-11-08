"""Predictors accept no packet truth, future archives, map ID or private seed."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import lightgbm as lgb
from .contracts import read,sha
from .model import CausalAdapter
from .features import sequence as network_sequence
from forecast_recall.features import sequence as rf_sequence
from forecast_peer.inference import teacher
from forecast_temporal.inference import parent_entry

def columns(arrays,variant):
    # Tree thresholds use deterministic observed features. The cached neural
    # reference probabilities remain available for baselines/blends, but tiny
    # CPU/GPU differences in those values must not flip a tree branch.
    parts=[arrays['base'][:,:432]]
    if variant in ('tree_network','tree_both'):parts.append(arrays['network'])
    if variant in ('tree_physics','tree_both'):parts.append(arrays['physics'])
    return np.concatenate(parts,-1).astype(np.float32) if len(parts)>1 else parts[0]

@lru_cache(maxsize=16)
def network(path,device):
    m=read(path)
    if sha(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('GRUweights changed')
    net=CausalAdapter(m['variant']).to(device);net.load_state_dict(torch.load(m['checkpoint'],map_location=device,weights_only=False)['model']);net.eval();return net

@lru_cache(maxsize=16)
def boosters(path):
    m=read(path);out={}
    for h,item in m['heads'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Treeweights changed')
        out[int(h)]=lgb.Booster(model_str=Path(item['path']).read_text(encoding='utf-8'))
    return out

def predict(path,inputs,device='cuda',batch=256):
    m=read(path)
    if m['family']=='v16_ensemble':
        parts=[]
        for member in m['members']:
            if sha(member['path'])!=member['sha256']:raise ValueError('Expert ensemble changed')
            parts.append(predict(member['path'],inputs,device,batch)*member['weight'])
        return sum(parts)
    if m['family']=='v16_tree':
        x=columns(inputs,m['variant']);p=np.array(inputs['baseline'],copy=True)
        for head,booster in boosters(str(path)).items():p[:,head]=booster.predict(x,num_threads=2)
        return p
    net=network(str(path),device);member=m['member_index'];out=[]
    with torch.inference_mode():
        for start in range(0,len(inputs['ends']),batch):
            s=slice(start,start+batch);ends=inputs['ends'][s];e=torch.as_tensor(inputs['embedding'][s,member],device=device);z=torch.as_tensor(inputs['logits'][s,member],device=device)
            r=torch.as_tensor(rf_sequence(inputs['history'],ends),device=device);n=torch.as_tensor(network_sequence(inputs['network_history'],ends),device=device);base=teacher(member,e,z,r,device)
            value,_=net(e,base,r,n);out.append(torch.sigmoid(value).cpu().numpy())
    return np.concatenate(out)

def baseline(item,scope):
    with np.load(item['path'],allow_pickle=False) as a:return a['baseline'][:,None,None]

def pin(path):
    p=Path(path);m=read(p);files={str(p):sha(p)}
    if m['family']=='v16_ensemble':
        for member in m['members']:files.update(pin(member['path'])['files'])
    elif m['family']=='v16_tree':
        for h in m['heads'].values():files[h['path']]=sha(h['path'])
    else:files[m['checkpoint']]=sha(m['checkpoint'])
    return dict(path=str(p),files=files)
