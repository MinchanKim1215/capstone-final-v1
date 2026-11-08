"""Frozen V13 logits plus label-free direct-peer residual prediction."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from . import PARENT
from .contracts import read,sha
from .model import PeerAdapter
from .features import sequence
from forecast_recall.features import sequence as global_sequence
from forecast_recall.inference import network as old_network,parent_entry as nav_entry
from forecast_temporal.inference import parent_entry

def teacher(member,embedding,logits,history,device):
    ensemble=read(parent_entry()['candidate']['model']);m=ensemble['members'][member]
    if sha(m['path'])!=m['sha256']:raise ValueError('Teacher ensemble changed')
    net=old_network(m['path'],device)
    with torch.inference_mode():return net(embedding,logits,history)[0].clone()

def baseline(item,scope):
    if scope=='confirmation':
        if sha(item['reference_path'])!=item['reference_sha256']:raise ValueError('Blind holdout baseline changed')
        with np.load(item['reference_path']) as a:return a['prediction']
    e=parent_entry();s=read(e['selection_summary'] if scope=='selection' else e['calibration_summary']);case=next(c for c in s['cases'] if c['key']==item['key'])
    if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Old probability changed')
    with np.load(case['predictions_path']) as a:return a['prediction']

@lru_cache(maxsize=16)
def network(path,device):
    m=read(path)
    if sha(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('Peer weight changed')
    net=PeerAdapter(m['variant']).to(device);net.load_state_dict(torch.load(m['checkpoint'],map_location=device,weights_only=False)['model']);net.eval();return net

def predict(path,inputs,device='cuda',batch=256):
    m=read(path)
    if m['family']=='v15_peer_ensemble':
        parts=[]
        for member in m['members']:
            if sha(member['path'])!=member['sha256']:raise ValueError('Peer ensemble changed')
            parts.append(predict(member['path'],inputs,device,batch)*member['weight'])
        return sum(parts)
    net=network(str(path),device);member=m['member_index'];parts=[]
    with torch.inference_mode():
        for start in range(0,len(inputs['ends']),batch):
            s=slice(start,start+batch);ends=inputs['ends'][s];e=torch.as_tensor(inputs['embedding'][s,member],device=device);z=torch.as_tensor(inputs['logits'][s,member],device=device)
            gh=torch.as_tensor(global_sequence(inputs['history'],ends),device=device);ph=torch.as_tensor(sequence(inputs['peer_history'],ends),device=device)
            base=teacher(member,e,z,gh,device);n=torch.as_tensor(inputs['n'][s],device=device,dtype=torch.long);ns=torch.as_tensor(inputs['ns'][s],device=device,dtype=torch.long)
            value,_=net(e,base,gh,ph,n,ns);parts.append(torch.sigmoid(value).cpu().numpy())
    return np.concatenate(parts)

def pin(path):
    p=Path(path);m=read(p);files={str(p):sha(p)}
    if m['family']=='v15_peer_ensemble':
        for member in m['members']:files.update(pin(member['path'])['files'])
    else:files[m['checkpoint']]=sha(m['checkpoint'])
    return dict(path=str(p),files=files)
