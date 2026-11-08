"""119fields per peer; all masked RF is removed before normalization."""
import numpy as np
from forecast_rebuild.features import SCALE,INTENT_SCALE,normalize_energy
DIM=119
MAX_N=10
OFFSETS=np.arange(-19,1)

def steps(values,masks,intents,energy,n,ns):
    v=np.asarray(values,np.float32);m=np.asarray(masks,bool);c=np.asarray(intents,np.float32)/INTENT_SCALE
    x=np.clip(np.where(m,v,0)/SCALE,-8,8);e=normalize_energy(energy);em=np.isfinite(e);e=np.clip(np.nan_to_num(e,nan=0.,posinf=0.,neginf=0.),-8,8)
    # Own fields0/1/5/6 are issuing-state fields. Peer path fields require flag8.
    for j in (2,3,4,7):c[:,:,j]=np.where(c[:,:,8]>.5,c[:,:,j],0)
    role=np.zeros((len(v),n,2),np.float32);role[:,:ns,0]=1;role[:,ns:,1]=1
    a=np.concatenate([x,m.astype(np.float32),np.clip(c,-8,8),e,em.astype(np.float32),role],-1)
    if a.shape!=(len(v),n,DIM) or not np.isfinite(a).all():raise ValueError('Invalid observed peer representation')
    out=np.zeros((len(v),MAX_N,DIM),np.float16);out[:,:n]=a
    return out

def sequence(history,endpoints):
    e=np.asarray(endpoints,int)//2-1;ix=e[:,None]+OFFSETS[None]
    if ix.min()<0 or ix.max()>=len(history):raise ValueError('Causal peer-history boundary violation')
    return np.asarray(history[ix],np.float32)
