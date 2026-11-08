"""Reconstruct only issued navigation history; never replay future actions."""
from functools import lru_cache
import numpy as np
from adaptive_joint_expanded import ARTIFACTS as NAV_ARTIFACTS
from adaptive_joint_expanded.baseline import ensure_nominal,paired_shadow
from adaptive_joint_expanded.contracts import DEPLOYMENT_BY_KEY
from swarm_sim.config import Scenario
from usable.collect import prepare_city_without_unused_radio_cache
from cooperative.motion import motion_stream
from forecast_prevention.diagnostics import connected,outage_runs
from final_v1.causes import radio_samples,pattern
from . import WORKSPACE,LEGACY
from .protocol import read,sha,maps,digest,CONFIG


@lru_cache(maxsize=4)
def context(metadata_path):
    meta=read(metadata_path)
    if 'mission' in meta:
        meta=dict(meta['mission'],spec=meta['spec'],archive_path=meta['archive_path'],
                  archive_sha256=meta['archive_sha256'])
    if 'archive_path' not in meta:
        # Feature caches retain the original V7 source record.
        rec=meta.get('source_record',meta.get('source',{}))
        original=rec.get('metadata_path')
        if original:
            return context(original)
        raise ValueError('Expected a raw flight metadata record')
    if sha(meta['archive_path'])!=meta['archive_sha256']:
        raise ValueError('Changed source flight')
    registry=maps()[meta['spec']['region']]
    path=LEGACY/registry['path']
    if sha(path)!=registry['sha256']:
        raise ValueError('Changed map')
    scenario=Scenario(direction=meta['spec']['direction'],motion_dt_s=3).validate()
    city=prepare_city_without_unused_radio_cache(path,scenario,NAV_ARTIFACTS/'cache/city')
    city.radio.enable_cache()
    fleet=DEPLOYMENT_BY_KEY[meta['deployment']]
    anchor=ensure_nominal(city,DEPLOYMENT_BY_KEY['v8_s3_r5_40'],scenario)
    with np.load(anchor,allow_pickle=False) as data:
        anchor_duration=(len(data['nodes'])-1)*3.
    rng=np.random.default_rng(meta['spec']['seed'])
    xy=city.target_points[int(rng.integers(len(city.target_points)))]
    kind=meta['spec']['kind']
    if kind=='clear':
        begin=end=power=0.
    elif kind=='continuous':
        begin=0.;end=1e9;power=float(rng.uniform(-3,3))
    else:
        begin=float(rng.uniform(.15,.45)*anchor_duration)
        end=float(rng.uniform(.65,.9)*anchor_duration)
        power=float(rng.uniform(-3,3))
    jammer=dict(jammer_xy_m=xy.tolist(),tx_power_dbm=power,intermittent_start_s=begin,intermittent_stop_s=end)
    with np.load(meta['archive_path'],allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ('nodes','commands')}
        arrays['positions']=z['positions'].copy() if 'positions' in z else city.points[z['nodes']]
        arrays['links']=z['links'].copy() if 'links' in z else z['actual_links'].copy()
    return meta,city,scenario,fleet,jammer,arrays


def anchors(entry):
    meta,city,scenario,fleet,jammer,z=context(entry['path'])
    runs=outage_runs(connected(z['links'],fleet.ns))
    eligible=[(a,b) for a,b in runs if a>=66 and (a*.5-12)//3*3+36 < meta['duration_s']]
    pool=[]
    if eligible:
        frames=np.unique(np.concatenate([np.arange(a-12,a+1) for a,_ in eligible]))
        m,g=radio_samples(city,scenario,z['positions'],frames,meta['spec']['seed'],
                          meta['spec']['kind'],jammer,fleet.ns)
        np.testing.assert_array_equal(g,z['links'][frames])
        for a,b in eligible:
            trace=m[np.searchsorted(frames,np.arange(a-12,a+1))]
            start=float(np.floor((a*.5-12)/3)*3)
            pool.append(dict(start_s=start,reference_onset_s=a*.5,source_pattern=pattern(trace),anchor_kind='outage'))
    ordered=sorted(pool,key=lambda e:digest([entry['path'],e['start_s'],'anchor']))
    chosen=[];types=set()
    for e in ordered+ordered:
        if len(chosen)>=3:
            break
        if any(abs(e['start_s']-o['start_s'])<24 for o in chosen):
            continue
        if e['source_pattern'] not in types or len(types)>=len({q['source_pattern'] for q in pool}):
            chosen.append(e);types.add(e['source_pattern'])
    # A quiet source interval prevents the experiment from containing cuts only.
    network=connected(z['links'],fleet.ns)
    quiet=[t for t in np.arange(33,meta['duration_s']-36,3) if network[int((t-6)*2):int((t+30)*2)].all()
           and all(abs(t-e['start_s'])>=30 for e in chosen)]
    if quiet:
        t=min(quiet,key=lambda t:digest([entry['path'],float(t),'quiet']))
        chosen.append(dict(start_s=float(t),reference_onset_s=None,source_pattern='quiet',anchor_kind='quiet'))
    result=[]
    for e in chosen:
        key=digest([meta['spec'],e['start_s'],'fast-branch'])[:20]
        result.append(dict(e,key=key,source_path=entry['path'],source_sha256=entry['sha256'],spec=meta['spec'],
                           fine_seed=int(digest([key,'fine'])[:8],16)%(2**31-1000)))
    return result


def effective(command,proposal,city,ns,k,baseline_duration):
    c=np.array(command,float,copy=True)
    lag=np.linalg.norm(city.points[proposal['nodes'][:ns]]-proposal['nominal'][:ns],axis=1).max()/40.
    if np.any(c) and (k*3>baseline_duration*1.25 or lag>1.5 or proposal['phase']==2):
        c[:]=0
    return c


class Program:
    """Published held-command plan, with no source links or future commands."""
    def __init__(self,city,ns,nominal,search_goals,initial_offsets):
        self.city=city;self.ns=ns
        self.nominal=np.asarray(nominal,float)
        self.search_goals=np.asarray(search_goals,np.int32)
        self.initial_offsets=np.asarray(initial_offsets,float)

    def goals(self,time_s,offsets):
        k=min(int((time_s+1e-8)//3),len(self.nominal)-1)
        target=self.nominal[k].copy()
        target[self.ns:]+=offsets
        target=np.clip(target,-195,195)
        ids=self.city.tree.query(target)[1].astype(np.int32)
        ids[:self.ns]=self.search_goals[k]
        return ids

    def schedule(self,time_s,offsets,horizon_s=8.):
        times=time_s+np.arange(0,horizon_s+.001,.1)
        steps=np.minimum(((times+1e-8)//3).astype(int),len(self.nominal)-1)
        targets=self.nominal[steps].copy()
        targets[:,self.ns:]+=offsets[None]
        ids=self.city.tree.query(np.clip(targets,-195,195).reshape(-1,2))[1].reshape(len(times),-1).astype(np.int32)
        ids[:,:self.ns]=self.search_goals[steps]
        return ids


def program_for(case):
    return _program_for(case['source_path'],float(case['start_s']))


@lru_cache(maxsize=4)
def _program_for(source_path,start_s):
    meta,city,scenario,fleet,jammer,z=context(source_path)
    step=int(round(start_s/3))
    stream=motion_stream(city,fleet.design(),scenario)
    proposal=next(stream)
    for k,command in enumerate(z['commands'][:step]):
        c=effective(command,proposal,city,fleet.ns,k,meta['baseline_duration_s'])
        event=stream.send(c)
        np.testing.assert_array_equal(event['new'],z['nodes'][k+1])
        proposal=next(stream)
    initial=z['nodes'][step].copy()
    np.testing.assert_array_equal(proposal['nodes'],initial)
    held=effective(z['commands'][step],proposal,city,fleet.ns,step,meta['baseline_duration_s'])
    nominal=[];search=[]
    for j in range(13):
        nominal.append(proposal['nominal'].copy())
        c=effective(held,proposal,city,fleet.ns,step+j,meta['baseline_duration_s'])
        event=stream.send(c)
        search.append(event['new'][:fleet.ns].copy())
        try:
            proposal=next(stream)
        except StopIteration:
            break
    return initial,Program(city,fleet.ns,nominal,search,held[fleet.ns:]),(meta,city,scenario,fleet,jammer)


def fine_shadow(case,n):
    """OU conditional bridges, exact at every archived half-second endpoint."""
    start=int(round(case['start_s']*2))
    steps=int(CONFIG['branch_duration_s']*10)
    coarse=paired_shadow(case['spec']['seed'],start+steps//5+2,n,.5,1.5,4.)
    output=np.empty((steps+1,n,n))
    rng=np.random.default_rng(case['fine_seed'])
    sigma2=1.5**2
    for interval in range(steps//5):
        x=coarse[start+interval].copy();target=coarse[start+interval+1]
        output[interval*5]=x
        for j in range(1,5):
            remaining=(5-j)*.1
            rho=np.exp(-.1/4.)
            r=np.exp(-remaining/4.)
            variance=sigma2*(1-rho*rho)
            total=sigma2*(1-(rho*r)**2)
            gain=variance*r/total
            mean=rho*x+gain*(target-r*rho*x)
            sd=np.sqrt(max(0.,variance-gain*r*variance))
            draw=np.zeros((n,n))
            ii,jj=np.tril_indices(n,-1)
            draw[ii,jj]=rng.normal(size=len(ii));draw[jj,ii]=draw[ii,jj]
            x=mean+sd*draw
            output[interval*5+j]=x
    output[-1]=coarse[start+steps//5]
    np.testing.assert_array_equal(output[::5],coarse[start:start+steps//5+1])
    return output

