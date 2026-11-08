"""Label-free new graph prediction and frozen V13 reference."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from . import ARTIFACTS,PARENT
from .contracts import read,sha
from .model import TemporalGraph

def parent_entry():
    f=read(PARENT/'freeze.json');return next(i for i in f['candidates'] if i['candidate']==f['decision']['candidate'])

def baseline(item,scope):
    if scope=='confirmation':
        if sha(item['reference_path'])!=item['reference_sha256']:raise ValueError('Fresh V13 risk changed')
        with np.load(item['reference_path']) as a:return a['prediction']
    entry=parent_entry();s=read(entry['selection_summary'] if scope=='selection' else entry['calibration_summary']);c=next(c for c in s['cases'] if c['key']==item['key'])
    if c['source_sha256']!=item['source_record']['array_sha256'] or sha(c['predictions_path'])!=c['prediction_sha256']:raise ValueError('Wrong frozen cohort')
    with np.load(c['predictions_path']) as a:return a['prediction']

@lru_cache(maxsize=16)
def network(path,device):
    m=read(path)
    if sha(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('Temporal weights changed')
    net=TemporalGraph(m['variant']).to(device);net.load_state_dict(torch.load(m['checkpoint'],map_location=device,weights_only=False)['model']);net.eval();return net

def predict(path,arrays,device='cuda',batch=96):
    m=read(path)
    if m['family']=='v14_temporal_ensemble':
        parts=[]
        for member in m['members']:
            if sha(member['path'])!=member['sha256']:raise ValueError('Temporal ensemble changed')
            parts.append(predict(member['path'],arrays,device,batch)*member['weight'])
        return sum(parts)
    net=network(str(path),device);parts=[];size=len(arrays['index'])
    with torch.inference_mode():
        for start in range(0,size,batch):
            s=slice(start,start+batch);j=arrays['index'][s]
            inputs=dict(node=torch.as_tensor(arrays['node'][j],device=device,dtype=torch.float32),edge=torch.as_tensor(arrays['edge'][j],device=device,dtype=torch.float32))
            inputs.update({k:torch.as_tensor(arrays[k][s],device=device,dtype=torch.long if k in ('n','ns','observer') else torch.float32) for k in ('physics','global_features','n','ns','observer')})
            parts.append(torch.sigmoid(net(**inputs)['service']).cpu().numpy())
    return np.concatenate(parts)

def pin(path):
    p=Path(path);m=read(p);files={str(p):sha(p)}
    if m['family']=='v14_temporal_ensemble':
        for member in m['members']:files.update(pin(member['path'])['files'])
    else:files[m['checkpoint']]=sha(m['checkpoint'])
    return dict(path=str(p),files=files)
