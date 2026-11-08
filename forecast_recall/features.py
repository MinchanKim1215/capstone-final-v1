"""480 causal local aggregates perframe; only past20one-second samples."""
import numpy as np
from forecast_rebuild.features import SCALE,INTENT_SCALE,normalize_energy
DIM=480
OFFSETS=np.arange(-39,0,2)


def stats(a,known,axis=1):
    count=known.sum(axis);v=np.where(known,a,0);mean=v.sum(axis)/np.maximum(count,1)
    sd=np.sqrt(np.maximum(0,(v*v).sum(axis)/np.maximum(count,1)-mean*mean))
    low=np.where(known,a,np.inf).min(axis);high=np.where(known,a,-np.inf).max(axis)
    return tuple(np.where(count>0,x,0) for x in (mean,sd,low,high))


def steps(values,masks,intents,energy,n,ns):
    v=np.asarray(values,np.float32)/SCALE;m=np.asarray(masks,bool).copy();c=np.asarray(intents,np.float32)/INTENT_SCALE
    if v.ndim!=3 or v.shape[1:]!=(n,48):raise ValueError('Only one relay local history accepted')
    m[:,ns]=False;parts=list(stats(v,m));parts.append(m.mean(1))
    for lo,hi in [(0,ns),(ns,n)]:
        mean,sd,_,_=stats(v[:,lo:hi],m[:,lo:hi]);parts.extend([mean,sd])
    cm=np.ones(c.shape,bool);cm[:,ns]=False
    for j in (2,3,4,7):cm[:,:,j] &= c[:,:,8]>.5
    mean,_,low,high=stats(c,cm);parts.extend([mean,low,high])
    e=normalize_energy(energy);em=np.isfinite(e);em[:,ns]=False
    mean,sd,_,_=stats(np.nan_to_num(e),em);parts.extend([mean,sd,em.mean(1)])
    parts.append(np.tile([ns/6,(n-ns)/7,n/10],(len(v),1)))
    x=np.concatenate(parts,-1).astype(np.float32)
    if x.shape!=(len(v),DIM) or not np.isfinite(x).all():raise ValueError('Invalid local context')
    return np.clip(x,-100,100).astype(np.float16)


def sequence(step,endpoints):
    end=np.asarray(endpoints,int);ix=end[:,None]+OFFSETS[None]
    if ix.min()<0 or ix.max()>=len(step):raise ValueError('Local history boundary violation')
    return np.asarray(step[ix],np.float32)
