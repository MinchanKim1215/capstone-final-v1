"""Causal physics shield with the same rule at every tested planning cadence."""
from time import perf_counter
import numpy as np
import shapely
from adaptive_joint_expanded.features import THRESHOLD,widest
from forecast_shield.forecaster import geometry
from .protocol import CONFIG

TIMES=np.r_[np.arange(0,1.01,.1),np.arange(1.5,8.01,.5)]
WEIGHTS=np.r_[np.diff(TIMES),.5]


def runs(margin):
    starts=[];elapsed=0.;begin=None
    for i in range(len(margin)-1):
        if margin[i]<0:
            if begin is None:
                begin=float(TIMES[i])
            elapsed+=TIMES[i+1]-TIMES[i]
            if elapsed>=1.-1e-8 and (not starts or starts[-1]!=begin):
                starts.append(begin)
        else:
            elapsed=0.;begin=None
    return starts


def margins(city,scenario,xy,estimate,ns):
    loss,_=geometry(city,scenario,xy)
    residual=np.nan_to_num(np.clip(estimate.power-(10.-loss[0]),-5.,5.))
    directed=10.-loss+residual*np.exp(-TIMES[:,None,None]/4.)-estimate.noise[None,None,:]-THRESHOLD
    uncertainty=np.where(estimate.measured,0.,1.5)
    directed-=np.maximum(uncertainty[None,:,None],uncertainty[None,None,:])
    directed-=.25*np.maximum(0.,TIMES[:,None,None]-3.)
    graph=np.minimum(directed,directed.swapaxes(-1,-2))
    net=widest(graph,ns).min(axis=-1)
    return net,graph


def critical_endpoints(graph,ns):
    """Weakest edge on the widest path of the least-connected terminal pair."""
    i,j=np.triu_indices(ns,1)
    k=int(np.argmin(widest(graph,ns)))
    source,target=int(i[k]),int(j[k])
    n=len(graph);quality=np.full(n,-np.inf);parent=np.full(n,-1,int);done=np.zeros(n,bool)
    quality[source]=np.inf
    for _ in range(n):
        node=int(np.argmax(np.where(done,-np.inf,quality)))
        done[node]=True
        if node==target:
            break
        candidate=np.minimum(quality[node],graph[node])
        better=(candidate>quality)&~done
        parent[better]=node;quality[better]=candidate[better]
    edges=[];node=target
    while node!=source and parent[node]>=0:
        p=int(parent[node]);edges.append((float(graph[p,node]),p,node));node=p
    if not edges:
        return source,target
    _,a,b=min(edges)
    return a,b


class Controller:
    def __init__(self,city,scenario,program,fleet,latency):
        self.city=city;self.scenario=scenario;self.program=program;self.ns=fleet.ns
        self.latency=latency
        self.last_change=-1e9;self.projected_onset=None
        self.events=[];self.calls=[];self.assessments=0;self.candidates=0
        self.critical_nodes=(fleet.ns,0)
        self.published_onset=None

    def path(self,motion,now,offsets,applied):
        schedule=self.program.schedule(now,offsets)
        committed=int(round(self.latency*10))
        if committed:
            schedule[:committed]=self.program.schedule(now,applied)[:committed]
        return motion.preview(schedule,TIMES)[0]

    def decide(self,motion,now,requested,applied,estimate,emergency=False):
        tick=perf_counter();self.assessments+=1
        try:
            if not estimate.local_rf:
                return requested,None
            if now-self.last_change < (CONFIG['emergency_minimum_gap_s'] if emergency else CONFIG['minimum_target_hold_s'])-1e-8:
                return requested,None
            base_xy=self.path(motion,now,requested,applied)
            base,graph=margins(self.city,self.scenario,base_xy,estimate,self.ns)
            self.critical_nodes=critical_endpoints(graph[int(np.argmin(base))],self.ns)
            source_runs=runs(base)
            self.projected_onset=now+source_runs[0] if source_runs else None
            loss=float(np.dot(np.maximum(0.,CONFIG['margin_target_db']-base),WEIGHTS))
            # The same physical trigger is used in all cadence arms.
            if not source_runs or loss<CONFIG['minimum_gain_db_seconds']:
                return requested,None
            best=None;seen={base_xy.tobytes()}
            for r in range(len(requested)):
                options=[np.zeros(2),self.program.initial_offsets[r]]
                options.extend(requested[r]+d for d in np.array([[20,0],[-20,0],[0,20],[0,-20]]))
                for offset in options:
                    if np.any(np.abs(offset)>40.00001) or np.array_equal(offset,requested[r]):
                        continue
                    k=min(int((now+1e-8)//3),len(self.program.nominal)-1)
                    target=self.program.nominal[k,self.ns+r]+offset
                    if np.any(np.abs(target)>195.00001) or shapely.intersects_xy(self.city.flight_blocker,*target):
                        continue
                    candidate=requested.copy();candidate[r]=offset
                    xy=self.path(motion,now,candidate,applied)
                    key=xy.tobytes()
                    if key in seen:
                        continue
                    seen.add(key);self.candidates+=1
                    alt,g=margins(self.city,self.scenario,xy,estimate,self.ns)
                    early=TIMES<=3.
                    if (alt[early]<np.minimum(0.,base[early])-CONFIG['early_tolerance_db']).any():
                        continue
                    alt_runs=runs(alt)
                    if len(alt_runs)>len(source_runs):
                        continue
                    cost=float(np.dot(np.maximum(0.,CONFIG['margin_target_db']-alt),WEIGHTS))
                    gain=loss-cost
                    if gain<CONFIG['minimum_gain_db_seconds']:
                        continue
                    distance=float(np.linalg.norm(xy[-1]-base_xy[-1],axis=1).sum())
                    value=gain-.02*distance-.10*np.linalg.norm(offset-requested[r])/20.
                    if best is None or value>best[0]:
                        best=(value,candidate,r,gain,alt)
            if best is None:
                return requested,None
            _,candidate,r,gain,alt=best
            event=dict(time_s=round(now,6),emergency=emergency,reason='recovery' if base[0]<0 else 'prevention',
                       changed_relay=self.ns+r,old_offset=requested[r].tolist(),new_offset=candidate[r].tolist(),
                       predicted_onsets_s=source_runs,expected_gain_db_s=gain,
                       fresh_measurement_s=estimate.latest_stamp_s,local_drop_db=estimate.drop_db,
                       base_margin_db=np.round(base,3).tolist(),new_margin_db=np.round(alt,3).tolist())
            self.last_change=now;self.events.append(event)
            return candidate,event
        finally:
            self.calls.append((perf_counter()-tick)*1000.)

    def emergency_due(self,now,estimate):
        near=self.published_onset is not None and 0.<=self.published_onset-now<=1.5
        return estimate.drop_db>=CONFIG['emergency_drop_db'] or near
