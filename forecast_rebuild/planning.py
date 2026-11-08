"""Public nominal-route lookahead inferred from received positions, not future truth."""
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from .features import HORIZONS

VERSION='public_nominal_phase_from_local_past_v1'
DIMENSION=82


def names():
    return (['nominal_loss_'+str(i) for i in range(13)]+['nominal_los_'+str(i) for i in range(13)]+
            ['nominal_margin_'+str(i) for i in range(13)]+['nominal_outage_reference_'+str(i) for i in range(13)]+
            ['nominal_phase_spread','nominal_match_error','nominal_fresh_searches','nominal_supported']+
            ['velocity_loss_'+str(i) for i in range(13)]+['velocity_los_'+str(i) for i in range(13)])


def nominal_positions(path,phase,ids):
    phase=np.clip(np.asarray(phase),0,len(path)-1)
    low=np.floor(phase).astype(int);high=np.minimum(low+1,len(path)-1);fraction=phase-low
    return path[low,ids]*(1-fraction[...,None])+path[high,ids]*fraction[...,None]


def infer_phase(nominal,x,m,n_search,previous,clock_s):
    estimates=[];errors=[]
    # At most the known elapsed mission time can have passed. A small public
    # planning tolerance covers different nominal collision waiting times.
    maximum=min(len(nominal)-2,int(clock_s/3*1.25+10))
    if previous is None:minimum=0
    else:minimum=max(0,int(previous)-2);maximum=min(maximum,int(previous)+8)
    if maximum<minimum:return None
    ix=np.arange(minimum,maximum+1)
    own=x[0,12:14]
    for j in range(n_search):
        if not m[j,[0,1,12,13,30,34,35,36,37]].all() or x[j,30]>2:continue
        position=own+x[j,34:36];velocity=x[j,:2]+x[j,36:38]
        a=nominal[ix,j];b=nominal[np.minimum(ix+1,len(nominal)-1),j];d=b-a
        square=(d*d).sum(-1);f=np.clip(((position-a)*d).sum(-1)/np.maximum(square,1e-9),0,1)
        projected=a+d*f[:,None];distance=np.linalg.norm(projected-position,axis=-1)
        speed=np.linalg.norm(velocity)
        if speed>.2:
            cosine=(d*velocity).sum(-1)/(np.sqrt(square)*speed+1e-9)
            distance=distance+2*(1-np.clip(cosine,-1,1))
        best=int(np.argmin(distance))
        if distance[best]<=20:
            estimates.append(float(ix[best]+f[best]));errors.append(float(distance[best]))
    if not estimates:return None
    phase=float(np.median(estimates))
    if previous is not None:phase=max(previous-.1,phase)
    return phase,float(np.std(estimates)),float(np.mean(errors)),len(estimates)


def profile(city,nominal,values,masks,intents,ends,n_search,only_rows=None):
    """All arguments are a public plan and observed/past telemetry.

    The nominal plan is computed before flight without RF outcomes. Its phase
    is inferred from only this observer's received search-drone states. Future
    protection pauses, actual future positions and future PPO actions are not
    available here. These are explicitly alternative forecast hypotheses.
    """
    nr,n=values.shape[1:3];out=np.full((len(ends),nr,n,DIMENSION),np.nan,np.float32)
    previous=[None]*nr;valid_rows=[];states=[];quality=[];phases=[]
    chosen=None if only_rows is None else {tuple(map(int,row)) for row in only_rows}
    nominal=np.asarray(nominal,np.float64)
    for t,end in enumerate(ends):
        x=np.asarray(values[end-1],np.float64);m=np.asarray(masks[end-1],bool)
        c=np.asarray(intents[end-1],np.float64)
        for r in range(nr):
            inferred=infer_phase(nominal,x[r],m[r],n_search,previous[r],float(end)*.5)
            if inferred is not None:previous[r]=inferred[0]
            for j in range(n):
                if j==r+n_search or (chosen is not None and (t,r,j) not in chosen):continue
                if not m[r,j,[0,1,2,12,13,30,34,35,36,37]].all() or x[r,j,30]>2:continue
                own=np.r_[x[r,j,12:14],50.]
                peer=own+np.r_[x[r,j,34:36],0.]
                own_v=np.r_[x[r,j,:2],0.]
                peer_v=own_v+np.r_[x[r,j,36:38],0.]
                rem=float(np.clip(c[r,j,5],0,3));prem=float(np.clip(3-c[r,j,4],0,3)) if c[r,j,8]>.5 else 0.
                states.append((own,peer,own_v,peer_v,rem,prem))
                phases.append(inferred[0] if inferred is not None else np.nan)
                quality.append([inferred[1]/3,inferred[2]/20,inferred[3]/max(1,n_search),1.] if inferred is not None else [0,0,0,0])
                valid_rows.append((t,r,j))
    if not valid_rows:return out
    rows=np.asarray(valid_rows,int);phase=np.asarray(phases);quality=np.asarray(quality,np.float32)
    for start in range(0,len(rows),384):
        stop=min(start+384,len(rows));part=rows[start:stop];q=quality[start:stop];phase_part=phase[start:stop]
        a=np.stack([s[0] for s in states[start:stop]]);b=np.stack([s[1] for s in states[start:stop]])
        va=np.stack([s[2] for s in states[start:stop]]);vb=np.stack([s[3] for s in states[start:stop]])
        rem=np.array([s[4] for s in states[start:stop]]);prem=np.array([s[5] for s in states[start:stop]])
        loss_now=city.radio.propagation(a,b,city._forecast_scenario)[0]
        ca=np.clip(a[:,None,:]+HORIZONS[None,:,None]*va[:,None,:],[-195,-195,50],[195,195,50])
        cb=np.clip(b[:,None,:]+HORIZONS[None,:,None]*vb[:,None,:],[-195,-195,50],[195,195,50])
        cvloss,cvlos,_=city.radio.propagation(ca.reshape(-1,3),cb.reshape(-1,3),city._forecast_scenario)
        block=np.full((len(part),DIMENSION),np.nan,np.float32)
        block[:,56:69]=(cvloss.reshape(-1,13)-loss_now[:,None])/30
        block[:,69:82]=cvlos.reshape(-1,13)
        block[:,52:56]=q
        supported=np.isfinite(phase_part)
        if supported.any():
            selected=part[supported];phase_good=phase_part[supported]
            horizons=HORIZONS[None,:]/3
            ia=selected[:,1]+n_search;ib=selected[:,2]
            initial=nominal_positions(nominal,phase_good[:,None]+np.minimum(HORIZONS[None,:],rem[supported,None])/3,ia[:,None])
            future=nominal_positions(nominal,phase_good[:,None]+horizons,ia[:,None])
            pa=a[supported,None,:]+np.minimum(HORIZONS[None,:],rem[supported,None])[...,None]*va[supported,None,:]
            pa[...,:2]+=future-initial
            initial_b=nominal_positions(nominal,phase_good[:,None]+np.minimum(HORIZONS[None,:],prem[supported,None])/3,ib[:,None])
            future_b=nominal_positions(nominal,phase_good[:,None]+horizons,ib[:,None])
            pb=b[supported,None,:]+np.minimum(HORIZONS[None,:],prem[supported,None])[...,None]*vb[supported,None,:]
            pb[...,:2]+=future_b-initial_b
            pa[...,:2]=np.clip(pa[...,:2],-195,195);pb[...,:2]=np.clip(pb[...,:2],-195,195)
            loss,los,_=city.radio.propagation(pa.reshape(-1,3),pb.reshape(-1,3),city._forecast_scenario)
            delta=loss.reshape(-1,13)-loss_now[supported,None]
            margin=np.asarray(values[np.asarray(ends)[selected[:,0]]-1,selected[:,1],selected[:,2],41],np.float64)
            sigma=np.sqrt(1.5**2*(1-np.exp(-2*HORIZONS/4))+.5**2)
            block[supported,:13]=delta/30;block[supported,13:26]=los.reshape(-1,13)
            block[supported,26:39]=(margin[:,None]-delta)/30
            block[supported,39:52]=ndtr((delta-margin[:,None])/sigma)
        out[part[:,0],part[:,1],part[:,2]]=block
    return out


def public_context(record):
    from adaptive_joint_expanded.contracts import entry_for,LEGACY,DEPLOYMENT_BY_KEY
    from adaptive_joint_expanded import ARTIFACTS as PARENT_ARTIFACTS
    from adaptive_joint_expanded.baseline import ensure_nominal
    from usable.collect import prepare_city_without_unused_radio_cache
    from swarm_sim.config import Scenario
    entry=entry_for(record['region'])
    scenario=Scenario(duration_s=7200,motion_dt_s=3,direction=record['spec']['direction']).validate()
    city=prepare_city_without_unused_radio_cache(LEGACY/entry['path'],scenario,PARENT_ARTIFACTS/'cache/city')
    city._forecast_scenario=scenario
    path=ensure_nominal(city,DEPLOYMENT_BY_KEY[record['deployment']],scenario)
    with np.load(path,allow_pickle=False) as z:nominal=city.points[z['nodes']]
    return city,nominal
