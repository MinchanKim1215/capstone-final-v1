"""Bounded-acceleration motion on safe graph edges, with timed reservations.

Every leg starts/ends at rest. Target changes never teleport an airborne drone
or change its direction mid-edge. A new edge is reserved against the complete
remaining trajectories of other drones. All code is independent of radio truth.
"""
import numpy as np
from numba import njit

DT=.1
VMAX=5.
AMAX=2.5
SEPARATION=5.
RESERVATION_GUARD=.26


@njit(cache=True)
def duration(length):
    if length <= VMAX*VMAX/AMAX:
        return 2*np.sqrt(length/AMAX)
    return 2*VMAX/AMAX+(length-VMAX*VMAX/AMAX)/VMAX


@njit(cache=True)
def profile(length, elapsed):
    """Exact rest-to-rest trapezoid/triangle position and velocity."""
    if length < 1e-10:
        return 0.,0.
    peak=min(VMAX,np.sqrt(length*AMAX))
    ramp=peak/AMAX
    total=2*ramp+max(0.,length-peak*peak/AMAX)/peak
    t=max(0.,min(elapsed,total))
    if t <= ramp:
        return .5*AMAX*t*t,AMAX*t
    if t < total-ramp:
        return .5*peak*ramp+peak*(t-ramp),peak
    rest=total-t
    return length-.5*AMAX*rest*rest,AMAX*rest


@njit(cache=True)
def poses(points,start,end,elapsed, future=0.):
    n=len(start)
    xy=np.empty((n,2));vel=np.zeros((n,2))
    for i in range(n):
        delta=points[end[i]]-points[start[i]]
        length=np.sqrt((delta*delta).sum())
        if length<1e-10:
            xy[i]=points[end[i]]
        else:
            distance,speed=profile(length,elapsed[i]+future)
            xy[i]=points[start[i]]+delta*(distance/length)
            vel[i]=delta*(speed/length)
    return xy,vel


@njit(cache=True)
def reserve_ok(points,start,end,elapsed,i,target):
    delta=points[target]-points[start[i]]
    length=np.sqrt((delta*delta).sum())
    total=duration(length)
    # Relative speed is <=10 m/s. At 0.05 s spacing the nearest grid time
    # is <=0.025 s away, so the 0.26 m guard bounds unsampled separation.
    samples=int(np.ceil((total+5.)/.05))+1
    for k in range(samples):
        t=k*.05
        distance,_=profile(length,t)
        x=points[start[i]]+delta*(distance/max(length,1e-10))
        for j in range(len(start)):
            if j==i:
                continue
            d=points[end[j]]-points[start[j]]
            l=np.sqrt((d*d).sum())
            s,_=profile(l,elapsed[j]+t)
            y=points[start[j]]+d*(s/max(l,1e-10))
            if np.sqrt(((x-y)**2).sum()) < SEPARATION+RESERVATION_GUARD-1e-8:
                return False
    return True


@njit(cache=True)
def advance(points, predecessor, start, end, elapsed, goals, age, ns, dt=DT):
    """Reserve new edges at rest, then integrate all already reserved motion."""
    n=len(start)
    blocked=0
    for i in range(n):
        if start[i]!=end[i]:
            length=np.sqrt(((points[end[i]]-points[start[i]])**2).sum())
            if elapsed[i]>=duration(length)-1e-9:
                start[i]=end[i]
                elapsed[i]=0.
    order=np.argsort(-(age+(np.arange(n)<ns)*10000.))
    for i in order:
        if start[i]!=end[i] or start[i]==goals[i]:
            continue
        target=predecessor[goals[i],start[i]]
        if target<0:
            blocked+=1
            continue
        if reserve_ok(points,start,end,elapsed,i,target):
            end[i]=target
            elapsed[i]=0.
            age[i]=0.
        else:
            blocked+=1
    for i in range(n):
        elapsed[i]+=dt
        age[i]+=dt
    return blocked


@njit(cache=True)
def forecast(points, predecessor, start, end, elapsed, age, goals_schedule, ns, times, dt=DT):
    """Actual motion equations, prospective targets only; no RF or archives."""
    a=start.copy();b=end.copy();e=elapsed.copy();wait=age.copy()
    count=int(np.rint(times[-1]/dt))+1
    out=np.empty((len(times),len(start),2));vel=np.empty_like(out)
    j=0
    for step in range(count):
        if j<len(times) and abs(step*dt-times[j])<1e-6:
            out[j],vel[j]=poses(points,a,b,e)
            j+=1
        if step+1<count:
            advance(points,predecessor,a,b,e,goals_schedule[min(step,len(goals_schedule)-1)],wait,ns,dt)
    return out,vel


class Fleet:
    def __init__(self,city,nodes,ns):
        self.city=city;self.ns=ns
        self.start=np.asarray(nodes,np.int32).copy();self.end=self.start.copy()
        self.elapsed=np.zeros(len(nodes));self.age=np.arange(len(nodes))*.001
        self.blocked=0

    def positions(self):
        return poses(self.city.points,self.start,self.end,self.elapsed)

    def step(self,goals):
        self.blocked+=advance(self.city.points,self.city.predecessor,self.start,self.end,self.elapsed,
                              np.asarray(goals,np.int32),self.age,self.ns)
        return self.positions()

    def preview(self,goals,times):
        return forecast(self.city.points,self.city.predecessor,self.start,self.end,self.elapsed,
                        self.age,np.asarray(goals,np.int32),self.ns,np.asarray(times,float))

