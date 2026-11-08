"""Hash-checked neural/tree ensembles, consuming only computed causal features."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import lightgbm as lgb
from .contracts import read,sha
from .model import ForecastGraph,tensors


@lru_cache(maxsize=16)
def network(path,device):
    meta=read(path)
    if sha(meta['checkpoint'])!=meta['checkpoint_sha256']:raise ValueError('Neural checkpoint changed')
    net=ForecastGraph(meta['variant']).to(device)
    net.load_state_dict(torch.load(meta['checkpoint'],map_location=device,weights_only=False)['model']);net.eval();return net


@lru_cache(maxsize=32)
def booster(path):return lgb.Booster(model_str=Path(path).read_text(encoding='utf-8'))


def predict(model,arrays,record=None,ep=None,city=None,device='cuda',tree_features=None,batch=128):
    meta=read(model);family=meta['family']
    if family.endswith('_ensemble'):
        if family=='v11_tree_ensemble' and tree_features is None:
            from .tree import features
            tree_features=features(record,arrays,ep,city)
        predictions=[]
        for member in meta['members']:
            if sha(member['path'])!=member['sha256']:raise ValueError('Ensemble member changed')
            predictions.append((predict(member['path'],arrays,record,ep,city,device,tree_features,batch),member['weight']))
        return {task:sum(value[task]*weight for value,weight in predictions) for task in ('link','service')}
    count=len(arrays['n'])
    if family=='v11_graph':
        net=network(str(model),device);parts={task:[] for task in ('link','service')}
        with torch.no_grad():
            for start in range(0,count,batch):
                ix=np.arange(start,min(count,start+batch));out=net(**tensors(arrays,ix,meta['variant'],device))
                for task in parts:parts[task].append(torch.sigmoid(out[task]).cpu().numpy())
        return {task:np.concatenate(value) for task,value in parts.items()}
    if family=='v11_tree':
        if tree_features is None:
            from .tree import features
            tree_features=features(record,arrays,ep,city)
        lx,sx=tree_features;n=lx.shape[1]
        result=dict(link=np.full((count,16,6),np.nan,np.float32),service=np.full((count,6),np.nan,np.float32))
        for task,x in [('link',lx.reshape(-1,lx.shape[-1])),('service',sx)]:
            x=x[:,:meta['input_dimensions'][task]]
            for head,item in meta['heads'][task].items():
                if sha(item['path'])!=item['sha256']:raise ValueError('Tree head changed')
                p=booster(item['path']).predict(x,num_threads=2).astype(np.float32)
                if task=='link':result[task][:,:n,int(head)]=p.reshape(count,n)
                else:result[task][:,int(head)]=p
        return result
    raise ValueError('Unknown V11 model family: '+family)


def pin(path):
    meta=read(path);files={str(Path(path).resolve()):sha(path)}
    if meta['family'].endswith('_ensemble'):
        for member in meta['members']:files.update(pin(member['path'])['files'])
    elif meta['family']=='v11_graph':files[meta['checkpoint']]=sha(meta['checkpoint'])
    elif meta['family']=='v11_tree':
        for heads in meta['heads'].values():
            for item in heads.values():files[item['path']]=sha(item['path'])
    return dict(path=str(Path(path).resolve()),files=files)
