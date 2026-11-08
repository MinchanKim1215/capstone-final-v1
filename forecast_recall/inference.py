"""Label-free residual inference and immutable baseline cache access."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from . import ARTIFACTS,PARENT
from .contracts import read,sha
from .features import sequence
from .model import EventAdapter


def parent_entry():
    f=read(PARENT/'freeze.json');choice=f['decisions']['service']['candidate']
    return next(i for i in f['candidates'] if i['candidate']==choice)


def baseline(item,scope):
    if scope=='confirmation':
        if sha(item['reference_path'])!=item['reference_sha256']:raise ValueError('Fresh baseline reference changed')
        with np.load(item['reference_path'],allow_pickle=False) as a:return a['prediction']
    entry=parent_entry();summary=read(entry['selection_summary'] if scope=='selection' else entry['calibration_summary'])
    case=next(c for c in summary['cases'] if c['key']==item['key'])
    if case['source_sha256']!=item['source_record']['array_sha256'] or sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Parent belongs to a different flight')
    with np.load(case['predictions_path'],allow_pickle=False) as a:return a['prediction']


@lru_cache(maxsize=16)
def network(path,device):
    m=read(path)
    if sha(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('Adapter weight changed')
    net=EventAdapter(m['variant']).to(device);net.load_state_dict(torch.load(m['checkpoint'],map_location=device,weights_only=False)['model']);net.eval();return net


def predict(path,inputs,device='cuda',batch=256):
    m=read(path)
    if m['family']=='v13_event_ensemble':
        out=[]
        for member in m['members']:
            if sha(member['path'])!=member['sha256']:raise ValueError('Ensemble member changed')
            out.append(predict(member['path'],inputs,device,batch)*member['weight'])
        return sum(out)
    net=network(str(path),device);member=m['member_index'];parts=[]
    with torch.inference_mode():
        for start in range(0,len(inputs['ends']),batch):
            stop=start+batch;e=torch.as_tensor(inputs['embedding'][start:stop,member],device=device)
            z=torch.as_tensor(inputs['logits'][start:stop,member],device=device)
            h=torch.as_tensor(sequence(inputs['history'],inputs['ends'][start:stop]),device=device) if m['variant']=='memory_event' else None
            out,_=net(e,z,h);parts.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(parts)


def pin(path):
    path=Path(path);m=read(path);files={str(path):sha(path)}
    if m['family']=='v13_event_ensemble':
        for member in m['members']:files.update(pin(member['path'])['files'])
    else:files[m['checkpoint']]=sha(m['checkpoint'])
    return dict(path=str(path),files=files)
