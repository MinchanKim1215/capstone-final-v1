"""Causal observation perturbations, with consistent GIS/intent recomputation."""
from pathlib import Path
import shutil
import numpy as np
from .contracts import read,write,digest,sha,now,SEED
from forecast_rebuild.dataset import Episode,ARRAYS
from forecast_rebuild.planning import profile,public_context
from cooperative.fast_features import geometry_profile

VERSION='v9-consistent-noisy-belief-v1'


def noisy_energy(energy,n_search,rng,mode):
    out=np.array(energy,dtype=np.float32)
    if mode=='missing':out[:]=np.nan;out[...,3:5]=0;return out
    if mode=='clean':return out
    frames,nr,n=out.shape[:3];error=rng.normal(0,2.5,(frames,n))
    for r in range(nr):
        out[:,r,:,0]+=error[:,r+n_search,None]
        stamp=np.arange(frames)[:,None]+1-np.nan_to_num(out[:,r,:,2],nan=frames+1)*2
        stamp=np.clip(np.rint(stamp).astype(int),0,frames-1)
        out[:,r,:,1]+=error[stamp,np.arange(n)[None,:]]
    return out


def shift_positions(values,masks,intents,n_search,bias):
    v=np.array(values,dtype=np.float32);c=np.array(intents,dtype=np.float32);m=np.asarray(masks,bool)
    n=v.shape[2];a=bias[n_search:][None,:,None,:];delta=bias[None,None,:,:]-a
    for k in (0,1):v[...,12+k]=np.where(m[...,12+k],v[...,12+k]+a[...,k],v[...,12+k])
    for start in (16,34):
        known=m[...,start:start+2].all(-1)
        shifted=v[...,start:start+2]+delta
        v[...,start:start+2]=np.where(known[...,None],shifted,v[...,start:start+2])
        distance=np.linalg.norm(v[...,start:start+2],axis=-1)
        v[...,start+4]=np.where(known&m[...,start+4],distance,v[...,start+4])
        radial=(v[...,start:start+2]*v[...,start+2:start+4]).sum(-1)/np.maximum(distance,1)
        v[...,start+5]=np.where(known&m[...,start+5],radial,v[...,start+5])
    c[...,:2]-=a;c[...,6]=np.linalg.norm(c[...,:2],axis=-1)
    received=c[...,8]>.5
    c[...,2:4]=np.where(received[...,None],c[...,2:4]-bias[None,None,:,:],c[...,2:4])
    c[...,7]=np.where(received,np.linalg.norm(c[...,2:4],axis=-1),c[...,7])
    for r in range(n-n_search):c[:,r,r+n_search]=np.asarray(intents[:,r,r+n_search])
    # Values written to disk are the representation consumed by training and
    # live inference. Derive geometry from those same quantized observations.
    return v.astype(np.float16),c.astype(np.float16)


def recompute_radio_geometry(city,v,m):
    import shapely
    x=np.array(v,dtype=np.float32)
    high=np.asarray([g for g,h in zip(city.geometries,city.heights) if h>=40],object)
    if len(high):
        points=shapely.points(x[...,12:14].reshape(-1,2));tree=shapely.STRtree(high)
        near=shapely.distance(points,high[tree.nearest(points)]).reshape(x.shape[:3])
    else:near=np.full(x.shape[:3],np.sqrt(2)*400)
    x[...,27]=np.where(m[...,27],near,x[...,27])
    ix=np.argwhere(np.asarray(m[...,16:27]).all(-1))
    for start in range(0,len(ix),4096):
        t,r,j=ix[start:start+4096].T;z=x[t,r,j]
        a=np.c_[z[:,12:14],z[:,2]];b=a+np.c_[z[:,16:18],np.zeros(len(t))]
        va=np.c_[z[:,:2],np.zeros(len(t))];vb=va+np.c_[z[:,18:20],np.zeros(len(t))]
        loss,los,_=city.radio.propagation(a,b,city._forecast_scenario)
        aa=a+5*va;bb=b+5*vb;aa[:,:2]=np.clip(aa[:,:2],-195,195);bb[:,:2]=np.clip(bb[:,:2],-195,195)
        loss5,los5,_=city.radio.propagation(aa,bb,city._forecast_scenario)
        x[t,r,j,22:27]=np.c_[loss,loss5,los,los5,loss5-loss]
    return x.astype(np.float16)


def create(source_record,record,rows):
    folder=Path(record['cache_path']);manifest=folder/'manifest.json';setting=record['belief_profile']
    contract=dict(version=VERSION,source_key=source_record['key'],source_sha256=source_record['array_sha256'],profile=setting,
                  rows_sha256=digest(np.asarray(rows).tolist()))
    if manifest.exists():
        if read(manifest)['contract']!=contract:raise ValueError('Belief augmentation contract changed')
        return record
    episode=Episode(source_record);rng=np.random.default_rng(int(digest(['belief-v9',SEED,record['key']])[:16],16))
    city,nominal=public_context(source_record);sigma=setting['gps_axis_sigma_m']
    if sigma:
        bias=rng.normal(0,sigma,(episode.values.shape[2],2))
        values,intents=shift_positions(episode.values,episode.masks,episode.intents,record['n_search'],bias)
        values=recompute_radio_geometry(city,values,episode.masks)
    else:values=np.asarray(episode.values);intents=np.asarray(episode.intents)
    energy=noisy_energy(episode.energy,record['n_search'],rng,setting['energy_mode']).astype(np.float16)
    geometry=np.zeros_like(episode.geometry,dtype=np.float32)
    for t in np.unique(rows[:,0]):
        end=int(episode.ends[t]);selected=rows[rows[:,0]==t]
        for r in np.unique(selected[:,1]):
            peers=selected[selected[:,1]==r,2]
            geometry[t,r,peers]=geometry_profile(city,values[end-1,r,peers].astype(np.float32),episode.masks[end-1,r,peers],intents[end-1,r,peers].astype(np.float32))
    planned=profile(city,nominal,values,episode.masks,intents,episode.ends,record['n_search'],only_rows=rows)
    folder.mkdir(parents=True,exist_ok=True)
    changed=dict(values=values,intents=intents,geometry=geometry,energy=energy)
    for name in ARRAYS:
        target=folder/(name+'.npy')
        if name in changed:np.save(target,changed[name],allow_pickle=False)
        else:shutil.copyfile(Path(source_record['cache_path'])/(name+'.npy'),target)
    np.save(folder/'planning.npy',planned,allow_pickle=False)
    write(folder/'planning_manifest.json',dict(source_sha256=record['array_sha256'],rows_sha256=contract['rows_sha256'],public_plan_only=True,actual_future_positions_used=False))
    write(manifest,dict(contract=contract,created_at=now(),physical_labels_unchanged=True,unknown_peer_intents_preserved=True))
    return record

