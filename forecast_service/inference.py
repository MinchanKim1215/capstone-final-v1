"""Read-only predictors and complete recursive weight pins for both tasks."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import lightgbm as lgb
from .contracts import read,sha


def pin(path):
    path=str(Path(path).resolve());meta=read(path);files={path:sha(path)}
    if 'heads' in meta:
        for item in meta['heads'].values():
            if sha(item['path'])!=item['sha256']:raise ValueError('Tree weights changed')
            files[item['path']]=item['sha256']
    elif 'members' in meta:
        for member in meta['members']:
            if sha(member['path'])!=member['model_sha256']:raise ValueError('Ensemble member changed')
            files.update(pin(member['path'])['files'])
    elif meta['family']=='service_proxy':
        files.update(pin(meta['link_model'])['files'])
    elif 'checkpoint' in meta:
        if sha(meta['checkpoint'])!=meta['sha256']:raise ValueError('Neural weights changed')
        files[meta['checkpoint']]=meta['sha256']
    else:raise ValueError('Unknown model representation')
    return dict(path=path,sha256=sha(path),files=files)


@lru_cache(maxsize=32)
def boosters(path):
    meta=read(path)
    return {int(h):lgb.Booster(model_str=Path(item['path']).read_text(encoding='utf-8')) for h,item in meta['heads'].items()}


def tree_only(path):
    meta=read(path)
    return 'heads' in meta or ('members' in meta and all(tree_only(m['path']) for m in meta['members']))


def tree_predict(path,features):
    meta=read(path)
    if 'members' in meta:
        arrays=[(tree_predict(m['path'],features),m['weight']) for m in meta['members']]
        total=sum(np.where(np.isfinite(p),p,0)*w for p,w in arrays);weights=sum(np.isfinite(p)*w for p,w in arrays)
        return np.divide(total,weights,out=np.full_like(total,np.nan),where=weights>0)
    x=features[:,:meta.get('dimension',features.shape[1])];out=np.full((len(x),6),np.nan,np.float32)
    for head,model in boosters(str(path)).items():out[:,head]=model.predict(x,num_threads=2)
    return out
