"""One relay's past-only view; hidden packet/radio/trajectory arrays forbidden."""
import numpy as np
from forecast_rebuild.features import SCALE,INTENT_SCALE,normalize_energy
from usable.beacons import data_threshold_db

BASE_DIM=384
GRAPH_DIM=48
VERSION='relay_local_past3s_received_search_pair_gis_0_3_7s_v1'


def stats(a,known,axis):
    count=known.sum(axis);mean=np.where(known,a,0).sum(axis)/np.maximum(count,1)
    std=np.sqrt(np.maximum(0,np.where(known,a*a,0).sum(axis)/np.maximum(count,1)-mean*mean))
    low=np.where(known,a,np.inf).min(axis);high=np.where(known,a,-np.inf).max(axis)
    return tuple(np.where(count>0,x,np.nan) for x in (mean,std,low,high))


def base(episode,rows,n_search):
    rows=np.asarray(rows,int);t,r=rows.T;ends=np.asarray(episode.ends)[t];n=episode.values.shape[2]
    ix=ends[:,None]+np.arange(-6,0)[None,:]
    if np.any(ix<0):raise ValueError('Insufficient causal history')
    v=np.asarray(episode.values[ix,r[:,None]],np.float32)/SCALE
    m=np.array(episode.masks[ix,r[:,None]],bool);self_peer=r+n_search
    m[np.arange(len(rows))[:,None],np.arange(6)[None,:],self_peer[:,None]]=False
    current=v[:,-1];known=m[:,-1]
    parts=list(stats(current,known,1));parts.append(known.mean(1))
    mean,std,_,_=stats(v,m,(1,2));parts.extend([mean,std])
    c=np.asarray(episode.intents[ends-1,r],np.float32)/INTENT_SCALE
    cm=np.ones_like(c,bool);cm[np.arange(len(rows)),self_peer]=False
    for field in (2,3,4,7):cm[:,:,field] &= c[:,:,8]>.5
    cmean,_,clow,chigh=stats(c,cm,1);parts.extend([cmean,clow,chigh])
    energy=normalize_energy(episode.energy[ends-1,r]);em=np.isfinite(energy)
    em[np.arange(len(rows)),self_peer]=False;emean,estd,_,_=stats(np.nan_to_num(energy),em,1)
    parts.extend([emean,estd,em.mean(1),np.tile([n_search/6,(n-n_search)/7,n/10],(len(rows),1))])
    result=np.concatenate(parts,-1).astype(np.float32)
    assert result.shape==(len(rows),BASE_DIM),result.shape
    result[~np.isfinite(result)]=np.nan;return result


def graph(episode,rows,n_search,city):
    rows=np.asarray(rows,int);t,r=rows.T;ends=np.asarray(episode.ends)[t]
    x=np.asarray(episode.values[ends-1,r],np.float64);m=np.asarray(episode.masks[ends-1,r],bool)
    c=np.asarray(episode.intents[ends-1,r],np.float64);energy=np.asarray(episode.energy[ends-1,r],np.float64)
    # Row0 is a search peer, never the local relay's empty self row. Its own
    # position and velocity fields are measured on this relay, not on peer0.
    own=x[:,0,12:14];position=own[:,None,:]+x[:,:n_search,34:36]
    velocity=x[:,:n_search,:2]+x[:,:n_search,36:38]
    good=m[:,:n_search][:,:,[0,1,12,13,30,34,35,36,37]].all(-1)&(x[:,:n_search,30]<=2)
    planned=(c[:,:n_search,8]>.5)&(c[:,:n_search,4]<=2)&good
    remaining=np.clip(3-c[:,:n_search,4],0,3)
    offset=c[:,:n_search,2:4];length=np.linalg.norm(offset,axis=-1)
    offset=offset*np.minimum(1,5*remaining/np.maximum(length,1e-9))[...,None]
    sensor=(energy[:,:n_search,4]>.5)&np.isfinite(energy[:,:n_search,1])&(energy[:,:n_search,2]<=2)
    noise=np.where(sensor,energy[:,:n_search,1],-107.)
    a,b=np.triu_indices(n_search,1);pair_known=good[:,a]&good[:,b];parts=[]
    scenario=city._forecast_scenario;threshold=float(data_threshold_db())
    for horizon in (0.,3.,7.):
        progress=np.minimum(1,horizon/np.maximum(remaining,.001))
        commanded=offset*progress[...,None]+velocity*np.maximum(0,horizon-remaining)[...,None]
        move=np.where(planned[...,None],commanded,velocity*horizon)
        future=np.clip(position+move,-195,195)
        pa=np.concatenate([future[:,a],np.full((*future[:,a].shape[:-1],1),50.)],-1)
        pb=np.concatenate([future[:,b],np.full((*future[:,b].shape[:-1],1),50.)],-1)
        loss=np.full(pair_known.shape,np.nan);los=np.full_like(loss,np.nan)
        if pair_known.any():
            l,s,_=city.radio.propagation(pa[pair_known],pb[pair_known],scenario);loss[pair_known]=l;los[pair_known]=s
        margin=10-loss-np.maximum(noise[:,a],noise[:,b])-threshold
        lf=stats(np.nan_to_num(loss)/100,pair_known,1);mf=stats(np.nan_to_num(margin)/30,pair_known,1)
        adj=np.zeros((len(rows),n_search,n_search),bool);edge=pair_known&(margin>=0)
        adj[:,a,b]=edge;adj[:,b,a]=edge;ii=np.arange(n_search);adj[:,ii,ii]=True
        for k in range(n_search):adj |= adj[:,:,k,None]&adj[:,None,k,:]
        connected=np.where(good.all(1),adj.all((1,2)).astype(float),np.nan)
        known_count=pair_known.sum(1);los_mean=np.where(pair_known,los,0).sum(1)/np.maximum(known_count,1)
        edge_fraction=edge.sum(1)/np.maximum(known_count,1)
        parts.extend([np.stack(lf,-1),np.stack(mf,-1),np.stack([np.where(known_count>0,los_mean,np.nan),connected,
            np.where(known_count>0,edge_fraction,np.nan),pair_known.mean(1)],-1)])
    distance=np.linalg.norm(position[:,a]-position[:,b],axis=-1)/400
    relative=np.sum((position[:,a]-position[:,b])*(velocity[:,a]-velocity[:,b]),axis=-1)/np.maximum(distance*400,1e-9)/10
    parts.extend([np.stack(stats(distance,pair_known,1),-1),np.stack(stats(relative,pair_known,1),-1),
        np.stack([good.mean(1),np.where(good,x[:,:n_search,30],0).sum(1)/np.maximum(good.sum(1),1)/10,planned.mean(1),sensor.mean(1)],-1)])
    result=np.concatenate(parts,-1).astype(np.float32);assert result.shape==(len(rows),GRAPH_DIM),result.shape
    result[~np.isfinite(result)]=np.nan;return result


def build(episode,rows,n_search,city):return np.concatenate([base(episode,rows,n_search),graph(episode,rows,n_search,city)],-1)
