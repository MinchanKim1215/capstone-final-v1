"""Selection-only measurement stress. Physical outcomes and policy stay fixed.

These are diagnostic assumptions, not measured device error distributions.
Recompute every position-derived feature from the same noisy belief: retaining
clean GIS geometry alongside noisy positions would understate the degradation.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,digest,now,maps
from .dataset import Episode
from .planning import public_context,profile
from .inference import Predictor
from .metrics import binary,alert_counts,combine_alerts
from .calibration import transform,alarm_support


def energy_error(episode,rng,sigma=2.5,missing=False):
    energy=np.array(episode.energy,dtype=np.float32)
    if missing:
        energy[:]=np.nan;energy[...,3:5]=0;return energy
    frames,nr,n=energy.shape[:3];noise=rng.normal(0,sigma,(frames,n))
    for r in range(nr):
        observer=r+episode.record['n_search']
        energy[:,r,:,0]+=noise[:,observer,None]
        # A received old sample retains the error it had at measurement time.
        # Do not add a new independent error whenever the cache is consulted.
        stamp=np.arange(frames)[:,None]+1-np.nan_to_num(energy[:,r,:,2],nan=frames+1)*2
        stamp=np.clip(np.rint(stamp).astype(int),0,frames-1)
        energy[:,r,:,1]+=noise[stamp,np.arange(n)[None,:]]
    return energy


def biased_positions(episode,rng,sigma=2.):
    from cooperative.fast_features import geometry_profile
    import shapely
    changed=copy.copy(episode);city,nominal=public_context(episode.record)
    v=np.array(episode.values,dtype=np.float32);c=np.array(episode.intents,dtype=np.float32)
    ns=episode.record['n_search'];nr,n=v.shape[1:3]
    # A fixed bias per drone preserves velocity; independent of future truth.
    bias=rng.normal(0,sigma,(n,2));a=bias[ns:][None,:,None,:]
    delta=bias[None,None,:,:]-a
    v[...,12:14]+=a
    for start in (16,34):
        v[...,start:start+2]+=delta
        distance=np.linalg.norm(v[...,start:start+2],axis=-1)
        v[...,start+4]=distance
        v[...,start+5]=(v[...,start:start+2]*v[...,start+2:start+4]).sum(-1)/np.maximum(distance,1)
    c[...,:2]-=a;c[...,2:4]-=bias[None,None,:,:]
    c[...,6]=np.linalg.norm(c[...,:2],axis=-1);c[...,7]=np.linalg.norm(c[...,2:4],axis=-1)
    high=np.asarray([g for g,h in zip(city.geometries,city.heights) if h>=40],object)
    if len(high):
        points=shapely.points(v[...,12:14].reshape(-1,2));tree=shapely.STRtree(high)
        v[...,27]=shapely.distance(points,high[tree.nearest(points)]).reshape(v.shape[:3])
    else:v[...,27]=np.sqrt(2)*400
    good=np.asarray(episode.masks[...,16:27]).all(-1)
    ix=np.argwhere(good)
    for start in range(0,len(ix),4096):
        t,r,j=ix[start:start+4096].T;x=v[t,r,j]
        own=np.c_[x[:,12:14],x[:,2]];peer=own+np.c_[x[:,16:18],np.zeros(len(t))]
        va=np.c_[x[:,:2],np.zeros(len(t))];vb=va+np.c_[x[:,18:20],np.zeros(len(t))]
        loss,los,_=city.radio.propagation(own,peer,city._forecast_scenario)
        own5=own+5*va;peer5=peer+5*vb
        own5[:,:2]=np.clip(own5[:,:2],-195,195);peer5[:,:2]=np.clip(peer5[:,:2],-195,195)
        loss5,los5,_=city.radio.propagation(own5,peer5,city._forecast_scenario)
        v[t,r,j,22:27]=np.c_[loss,loss5,los,los5,loss5-loss]
    changed.values=v;changed.intents=c
    changed.geometry=np.zeros_like(episode.geometry,dtype=np.float32)
    for t,end in enumerate(episode.ends):
        for r in range(nr):changed.geometry[t,r]=geometry_profile(city,v[end-1,r],episode.masks[end-1,r],c[end-1,r])
    changed.planning=profile(city,nominal,v,episode.masks,c,episode.ends,ns)
    return changed


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--policy',required=True);a=p.parse_args()
    summary=read(a.summary)
    if summary['scope']!='selection':raise ValueError('Stress development uses selection, never confirmation for tuning')
    policy=read(a.policy);registry=maps();data=read(summary['dataset_path'])
    rows=[r for r in data['rows'] if r['role']=='selection']
    # Prespecified two geographies: first alphabetical legacy and added map.
    selected=[]
    for added in (False,True):
        regions=sorted({r['region'] for r in rows if (registry[r['region']]['source_dataset']!='legacy_v6')==added})
        if regions:selected.append(regions[0])
    rows=[r for r in rows if r['region'] in selected]
    predictor=Predictor(summary['model_manifest'],'cuda');out=[]
    folder=ARTIFACTS/'stress'/summary['model_tag'];folder.mkdir(parents=True,exist_ok=True)
    for kind in ('clean','energy_missing','energy_noise_2p5db','position_bias_2m'):
        case_metrics=[];alarms=[]
        for record in rows:
            episode=Episode(record);rng=np.random.default_rng(int(digest([record['key'],kind])[:16],16))
            if kind=='position_bias_2m':episode=biased_positions(episode,rng)
            elif kind.startswith('energy_'):
                episode.energy=energy_error(episode,rng,missing=kind=='energy_missing')
            prediction=np.full((*episode.valid.shape,6),np.nan,np.float32)
            eligible=np.argwhere(episode.valid)
            for start in range(0,len(eligible),768):
                batch=eligible[start:start+768];t,r,j=batch.T
                prediction[t,r,j]=predictor.predict(episode,batch)[0]
            support=episode.valid&episode.legacy_support
            live=support&episode.currently_up
            case_metrics.append(dict(key=record['key'],region=record['region'],kind=record['spec']['kind'],
                next5=binary(episode.targets[...,2][live],prediction[...,2][live]),
                any3to7=binary(episode.targets[...,5][support],prediction[...,5][support])))
            calibrated=transform(prediction,policy['parameters'])
            alarms.append(alert_counts(episode.links,episode.ends,calibrated[...,5],alarm_support(episode,policy.get('require_energy',False)),
                                        record['n_search'],policy['threshold']))
            print(kind,record['region'],record['spec']['kind'],flush=True)
        result=dict(profile=kind,cases=case_metrics,calibrated_alarms=combine_alerts(alarms),
                    macro_case_ap5=float(np.mean([r['next5']['ap'] for r in case_metrics])),
                    macro_case_ap37=float(np.mean([r['any3to7']['ap'] for r in case_metrics])))
        out.append(result);write(folder/'results.json',dict(profiles=out,regions=selected,created_at=now(),
            scope='Six selection flights only. No device-specific error distribution, no confirmation tuning.',
            physical_outcomes_unchanged=True,clean_position_geometry_reused_in_position_stress=False))
    print(folder/'results.json')


if __name__=='__main__':main()
