"""RF intervals [q/2,(q+1)/2); predictions use frames strictly before endpoint."""
import numpy as np

WINDOWS=((0,2),(0,6),(0,10),(0,16),(6,8),(6,14))
NAMES=('any_lt1s','any_lt3s','any_lt5s','any_lt8s','any_3to4s','any_3to7s')


def onsets(links):
    up=np.asarray(links,bool)
    if up.ndim!=3 or up.shape[1]!=up.shape[2]:raise ValueError('Square RF link trajectories required')
    result=np.zeros_like(up)
    result[1:-1]=up[:-2]&~up[1:-1]&~up[2:]
    return result


def labels(links,endpoints,n_search):
    up=np.asarray(links,bool);ends=np.asarray(endpoints,int)
    if not 1<=n_search<up.shape[1] or np.any(ends<1) or np.any(ends>len(up)):
        raise ValueError('Invalid role split or endpoint')
    starts=onsets(up)
    nr,n=up.shape[1]-n_search,up.shape[1]
    y=np.zeros((len(ends),nr,n,len(WINDOWS)),np.float32)
    valid=np.zeros((len(ends),nr,n),bool)
    next_bin=np.full(valid.shape,16,np.int64)
    live=np.zeros_like(valid)
    for t,end in enumerate(ends):
        if end+16>=len(up):continue
        valid[t]=True
        valid[t,np.arange(nr),np.arange(n_search,n)]=False
        live[t]=up[end-1,n_search:]&valid[t]
        for h,(low,high) in enumerate(WINDOWS):y[t,...,h]=starts[end+low:end+high,n_search:].any(0)
        future=starts[end:end+16,n_search:]
        next_bin[t]=np.where(future.any(0),future.argmax(0),16)
    return dict(targets=y,valid=valid,currently_up=live,next_bin=next_bin)


def past_support(values,masks,endpoints,n_search):
    values=np.asarray(values);masks=np.asarray(masks,bool);ends=np.asarray(endpoints,int)
    x=values[ends-1];m=masks[ends-1]
    old=m[...,41]&m[...,46]&(x[...,46]>.5)
    available=m[...,30]&m[...,34]&m[...,35]&(x[...,30]<=10)
    for out in (old,available):out[:,np.arange(out.shape[1]),np.arange(n_search,n_search+out.shape[1])]=False
    return dict(legacy=old,available=available)
