"""One-second causal prediction stream and raw three-second policy inputs."""
import time
import numpy as np
import torch
from cooperative.fast_features import geometry_profile
from .features import local_candidates_extended, OFFSETS
from .model import tensor_observation
from .targets import probabilities
from cooperative.continuity_control import transform_observation


class Observer:
    def __init__(self, model=None):
        self.model = model
        self.values=[]; self.masks=[]; self.intents=[]
        self.predictions=[]; self.prediction_support=[]; self.prediction_times=[]; self.prediction_geometry=[]
        self.inference_ms=[]

    def bind(self, city, deployment):
        self.city, self.deployment = city, deployment

    def inputs(self):
        d = self.deployment
        frames = len(self.values)
        outputs=[]
        for data,last,dtype in [(self.values,48,np.float32),(self.masks,48,bool),(self.intents,9,np.float32)]:
            a=np.asarray(data[-40:],dtype=dtype).reshape(-1,d.nr,d.n,last)
            if len(a)<40:
                a=np.concatenate([np.zeros((40-len(a),d.nr,d.n,last),dtype=dtype),a])
            outputs.append(a.transpose(1,2,0,3).copy())
        support=np.zeros((d.nr,d.n),bool)
        geo=np.zeros((d.nr,d.n,46),np.float32)
        if frames:
            x=np.asarray(self.values[-1],np.float32);m=self.masks[-1]
            intent=np.asarray(self.intents[-1],np.float32)
            support=m[...,41]&m[...,46]&(x[...,46]>.5)&(frames>=40)
            support[np.arange(d.nr),np.arange(d.ns,d.n)]=False
            geo=geometry_profile(self.city,x.reshape(-1,48),m.reshape(-1,48),intent.reshape(-1,9)).reshape(d.nr,d.n,46)
        return dict(values=outputs[0],masks=outputs[1],intents=outputs[2],geometry=geo,support=support)

    def observe(self, now, values, masks, intents):
        d=self.deployment
        if values.shape!=(d.n,d.n,48) or masks.shape!=values.shape or intents.shape!=(d.n,d.n,9):
            raise ValueError('RF/intent dimensions differ from the selected fleet')
        if not np.isclose(now,(len(self.values)+1)*.5):
            raise ValueError('RF frames must arrive once, in causal half-second order')
        self.values.append(np.asarray(values[d.ns:],np.float16).copy())
        self.masks.append(np.asarray(masks[d.ns:],bool).copy())
        self.intents.append(np.asarray(intents[d.ns:],np.float16).copy())
        if not np.isfinite(self.values[-1]).all() or not np.isfinite(self.intents[-1]).all():
            raise FloatingPointError('Nonfinite/overflowed causal RF input')
        if self.model is None or len(self.values)<40 or len(self.values)%2:
            return dict(status='recording',links=[])
        tick=time.perf_counter()
        inputs=self.inputs()
        device=next(self.model.parameters()).device
        with torch.no_grad():
            b=tensor_observation(inputs,device)
            _,timing=self.model.forecaster.encode(b['values'],b['masks'],b['intents'],b['geometry'])
            p=probabilities(timing).cpu().numpy()
        self.inference_ms.append((time.perf_counter()-tick)*1000)
        self.predictions.append(p)
        self.prediction_support.append(inputs['support'].copy())
        self.prediction_times.append(float(now))
        self.prediction_geometry.append(inputs['geometry'].copy())
        links=[]
        for r in range(d.nr):
            for j in range(d.n):
                if j==r+d.ns:continue
                known=bool(inputs['support'][r,j])
                links.append(dict(source=int(r+d.ns),peer=int(j),status='ready' if known else 'unknown_or_stale',
                                  risk_5s=float(p[r,j,0]) if known else None,
                                  risk_onset_3to7s=float(p[r,j,2]) if known else None))
        return dict(status='ready',links=links)


def policy_inputs(env):
    d=env.deployment
    env.city._cooperative_scenario=env.scenario
    context=[];candidates=[];masks=[];teachers=[];gains=[]
    for r,i in enumerate(range(d.ns,d.n)):
        missing=np.full(d.n,np.nan)
        a,b,m,t,g=local_candidates_extended(env.city,i,env.last_x[i],env.last_mask[i],
                    env.proposal['nominal'][i],env.k*3.,missing.copy(),missing.copy(),True,d.ns)
        if env.proposal['phase']==2:
            m[1:]=False
        context.append(a);candidates.append(b);masks.append(m);teachers.append(t);gains.append(g)
    # The selected v5 probability profile contains no fixed relay-count loop;
    # reuse it directly instead of silently switching physical objectives.
    c,f,m,t=transform_observation((np.asarray(context),np.asarray(candidates),np.asarray(masks),np.asarray(teachers)),
                                 env.previous_actions,'probability')
    out=env.predictor.inputs()
    out.update(context=np.asarray(c,np.float32),candidates=np.asarray(f,np.float32),
               action_mask=np.asarray(m,bool),teacher=np.asarray(t,np.float32),
               gain=np.asarray(gains,np.float32),
               design=np.array([d.n/10,d.ns/6,d.nr/5,d.spacing/100,d.formation=='v'],np.float32))
    if out['context'].shape!=(d.nr,34):
        raise ValueError('Dynamic local context contract changed')
    return out


def critic_state(env):
    """CTDE only: current global state. Never supplied to actor or selector."""
    d=env.deployment
    p=env.city.points[env.nodes]
    previous=OFFSETS[env.previous_actions]
    stats=env.stats[-1] if env.stats else np.zeros(2)
    off=env.p_now[~np.eye(d.n,dtype=bool)]
    state=np.r_[p.mean(0)/200,p.std(0)/200,p.min(0)/200,p.max(0)/200,
                 p[:d.ns].mean(0)/200,p[:d.ns].std(0)/200,
                 p[d.ns:].mean(0)/200,p[d.ns:].std(0)/200,
                 stats[0],stats[1],env.packets.lengths.mean()/64,
                 env.packets.ic[1]/max(1,env.packets.ic[0]),
                 env.proposal['observed_cells']/env.proposal['observable_cells'],
                 env.proposal['phase']/2,env.k/2400,d.n/10,d.ns/6,d.nr/5,
                 min(env.min_separation,50)/50,min(env.reward_age,10)/10,
                 np.linalg.norm(previous,axis=-1).mean()/60,
                 np.linalg.norm(previous,axis=-1).max()/60,off.mean(),off.min()]
    if state.shape!=(32,) or not np.isfinite(state).all():
        raise ValueError('Invalid centralized CURRENT state')
    return state.astype(np.float32)
