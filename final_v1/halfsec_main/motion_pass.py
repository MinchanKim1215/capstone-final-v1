"""Pass-through motion for the default half-second controller (final_v1_fast/motion.py에서 갈라짐).

원본(final_v1_fast/motion.py)은 이전 0.5초 연구가 해시로 고정해 두었으므로 수정하지 않고, 기본 제어만 이 파일을 쓴다.

원본과 다른 점 (나머지 식과 상수는 원본과 같다)
1. 정지한 기체가 다음 구간을 고를 때, 목표까지의 최단 경로를 따라 일직선이 이어지는 데까지 한 구간으로 예약한다
   (pick_run). 수색기는 최대 RUN_LIMIT_SEARCH m, 중계기는 RUN_LIMIT_RELAY m. 예약이 안 되면 한 칸씩 줄인다.
   속도는 원본과 같은 정지-가속(2.5 m/s²)-최고 5 m/s-감속-정지 식을 긴 구간 전체에 적용한다.
2. 예약 검사(reserve_ok)는 원본처럼 '내 이동 시간 + 5초'가 아니라 '다른 기체의 남은 이동이 모두 끝날 때까지 + 5초'를
   본다. 한 칸(최대 4.75초)일 때는 둘이 같지만, 긴 구간에서는 다른 기체가 그 뒤에도 움직일 수 있기 때문이다.

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
    horizon=total
    for j in range(len(start)):
        if j!=i and start[j]!=end[j]:
            dj=points[end[j]]-points[start[j]]
            remaining=duration(np.sqrt((dj*dj).sum()))-elapsed[j]
            if remaining>horizon:
                horizon=remaining
    samples=int(np.ceil((horizon+5.)/.05))+1
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
        target=pick_run(points,predecessor,start,end,elapsed,i,goals[i],
                        RUN_LIMIT_SEARCH if i<ns else RUN_LIMIT_RELAY)
        if target==-2:
            blocked+=1
            continue
        if target>=0:
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



# 한 번에 예약하는 직선 구간의 최대 길이 (m)
RUN_LIMIT_SEARCH=200.0
RUN_LIMIT_RELAY=50.0


@njit(cache=True)
def pick_run(points,predecessor,start,end,elapsed,i,goal,limit):
    """최단 경로를 따라 일직선이 이어지는 가장 먼 지점부터, 예약되는 지점을 고른다 (-2: 경로 없음, -1: 예약 불가)."""
    a=start[i]
    first=predecessor[goal,a]
    if first<0:
        return -2
    chain=np.empty(64,np.int64)
    chain[0]=first
    count=1
    d0=points[first]-points[a]
    total=np.sqrt((d0*d0).sum())
    u=d0/max(total,1e-10)
    last=first
    while last!=goal and count<64:
        nxt=predecessor[goal,last]
        if nxt<0:
            break
        d=points[nxt]-points[last]
        l=np.sqrt((d*d).sum())
        if l<1e-10 or (d[0]*u[0]+d[1]*u[1])/l<0.9999 or total+l>limit+1e-9:
            break
        chain[count]=nxt
        count+=1
        total+=l
        last=nxt
    for k in range(count-1,-1,-1):
        if reserve_ok(points,start,end,elapsed,i,chain[k]):
            return chain[k]
    return -1
