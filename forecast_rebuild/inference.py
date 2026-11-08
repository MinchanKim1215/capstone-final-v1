"""Label-free model loading and window prediction."""
from pathlib import Path
import numpy as np
import torch
from .contracts import read,sha,BASELINE
from .features import windows
from .models import TemporalPredictor,ResidualTransformer


def coherent(probability):
    p=np.asarray(probability,np.float32).copy()
    # Preserve the two benchmark/operational scores exactly. Restrict only
    # optional subset windows so displayed probabilities respect containment.
    p[...,0]=np.fmin(p[...,0],np.fmin(p[...,1],p[...,2]))
    p[...,1]=np.fmin(p[...,1],p[...,2])
    p[...,3]=np.fmax(p[...,3],np.fmax(p[...,2],p[...,5]))
    p[...,4]=np.fmin(p[...,4],np.fmin(p[...,2],p[...,5]))
    return p


class Predictor:
    def __init__(self,model='baseline',device='cuda'):
        self.device=device if device!='cuda' or torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(1)
        if model=='baseline':
            from adaptive_joint_expanded.model import Controller
            value=torch.load(BASELINE,map_location='cpu',weights_only=False)
            self.network=Controller(warm_start=False).to(self.device)
            self.network.load_state_dict(value['model'],strict=True);self.network.eval()
            self.meta=dict(family='baseline',tag='frozen_v7_64',sha256=sha(BASELINE),use_energy=False)
        else:
            self.path=Path(model);self.meta=read(self.path)
            if self.meta['family']=='lightgbm':
                import lightgbm as lgb
                self.trees={}
                for h,item in self.meta['heads'].items():
                    if sha(item['path'])!=item['sha256']:raise ValueError('Tree file differs from its model manifest')
                    self.trees[int(h)]=lgb.Booster(model_str=Path(item['path']).read_text(encoding='utf-8'))
            elif self.meta['family'] in ('tcn','residual_transformer','dense_residual'):
                if sha(self.meta['checkpoint'])!=self.meta['sha256']:raise ValueError('Neural weight file changed')
                value=torch.load(self.meta['checkpoint'],map_location='cpu',weights_only=False)
                if self.meta['family']=='dense_residual':
                    from .models_dense import DenseResidual
                    self.network=DenseResidual(value['static_dim'],value['width'],False).to(self.device)
                else:self.network=(ResidualTransformer(value['static_dim'],value['width'],False) if self.meta['family']=='residual_transformer'
                                   else TemporalPredictor(value['static_dim'],value['width'])).to(self.device)
                self.network.load_state_dict(value['model']);self.network.eval()
            elif self.meta['family']=='ensemble':
                for member in self.meta['members']:
                    if sha(member['path'])!=member['model_sha256']:raise ValueError('Ensemble member manifest changed')
                self.members=[(Predictor(m['path'],self.device),m['weight']) for m in self.meta['members']]
            else:raise ValueError('Unknown prediction family')

    def predict(self,episode,rows):
        rows=np.asarray(rows,int);family=self.meta['family']
        if family=='baseline':
            from adaptive_joint_expanded.targets import probabilities
            v,m,c=windows(episode.values,episode.masks,episode.intents,episode.ends,rows)
            t,r,j=rows.T;g=np.asarray(episode.geometry[t,r,j],np.float32)
            with torch.inference_mode():
                _,q=self.network.forecaster.encode(torch.as_tensor(v.astype(np.float32),device=self.device),
                    torch.as_tensor(m,device=self.device),torch.as_tensor(c.astype(np.float32),device=self.device),
                    torch.as_tensor(g,device=self.device))
                scores=probabilities(q).cpu().numpy();timing=torch.softmax(q,-1).cpu().numpy()
            out=np.full((len(rows),6),np.nan,np.float32)
            out[:,2]=scores[:,0];out[:,4]=scores[:,1];out[:,5]=scores[:,2]
            return out,timing
        if family=='ensemble':
            outputs=[(member.predict(episode,rows)[0],weight) for member,weight in self.members]
            total=sum(np.where(np.isfinite(p),p,0)*weight for p,weight in outputs)
            weights=sum(np.isfinite(p)*weight for p,weight in outputs)
            return np.divide(total,weights,out=np.full_like(total,np.nan),where=weights>0),None
        x=episode.features(rows,self.meta['use_energy'],self.meta.get('use_plan',False))
        if family=='lightgbm':
            out=np.full((len(rows),6),np.nan,np.float32)
            for head,booster in self.trees.items():out[:,head]=booster.predict(x,num_threads=2)
            return out,None
        seq=episode.sequence(rows,self.meta['use_energy'])
        x=np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
        with torch.inference_mode():
            any_logits,next_logits=self.network(torch.as_tensor(seq,device=self.device),torch.as_tensor(x,device=self.device))
            out=torch.sigmoid(any_logits).cpu().numpy();timing=torch.softmax(next_logits,-1).cpu().numpy()
        return coherent(out),timing
