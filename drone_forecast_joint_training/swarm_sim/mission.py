"""전체 경로와 관측 가능한 실외 셀을 완료하는 통신 비의존 대형 이동.

PIBT의 우선순위 상속과 되돌림 아이디어를 적용하고 선형 이동의 5m
분리 거리를 추가 검사한다. 원 논문 전체 알고리즘의 재현 또는 완전성
보장을 주장하지 않는다. 통신 장애 예측이나 위험 기반 재배치는 없다.
"""
# 현재 전체 수색의 경로·대형 이동 담당 파일입니다.
# 기준 경로 수색 → 잔여 관측 가능 셀 수색 → 30초 공유의 순서로 움직입니다.
# 중계 드론은 센서 관측을 하지 않지만, 수색 드론도 패킷을 중계할 수 있습니다.
# PIBT의 우선순위 상속·되돌림 아이디어를 변형해 충돌을 피합니다.
# 이 이동은 통신 품질을 보지 않는 기본 수색입니다. 예측이나 통신 위험 기반 재배치는 아직 없습니다.
from pathlib import Path
import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from numba import njit
from .city import City
from .motion import fleet_offsets, swept_minimum
from .radio import BuildingRadio

MOVE_DT=3.0
MAX_MISSION=7200.0
SHARE_SECONDS=30.0

# 50m고도·10m격자·25m센서의 고정 조건으로 지도와 캐시를 준비합니다.
# 캐시 이름에 지도 해시와 주요 공간값을 넣고, 허용하지 않은 조건은 오류로 차단합니다.
# observable은 가능한 모든 비행 노드에서 한 번이라도 관측되는 실외 셀의 합집합입니다.
def prepared_city(path,scenario,cache_dir):
    if not (scenario.altitude_m==50 and scenario.grid_m==10 and scenario.sensor_radius_m==25
      and scenario.sensor_occlusion and scenario.height_case=='delivered' and scenario.horizontal_clearance_m==5
      and scenario.vertical_clearance_m==0 and scenario.frequency_hz==2.4e9 and scenario.radio_model=='bullington'):
        raise ValueError('전체 임무 캐시는 고도50·격자10·센서25·전달된높이·여유5·2.4GHz·Bullington 고정 조건입니다.')
    city=City(path,scenario)
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    tag=f'{city.key}_{city.sha256[:12]}_{scenario.altitude_m:g}_{scenario.grid_m:g}_{scenario.sensor_radius_m:g}'
    file=cache_dir/(tag+'.npz')
    if file.exists():
        z=np.load(file);city.loss_cache=z['loss'];city.los_cache=z['los'];city.visible=z['visible']
    else:
        n=len(city.points);loss=np.empty((n,n),np.float32);los=np.empty((n,n),bool)
        # 같은 고도 A2A, 최대 수평 거리 566m, 2.4GHz에서 40m 미만 지붕은
        # 회절의 nu<-0.78 범위이다. 설정을 바꾸면 이 최적화를 적용하지 않는다.
        # 50m·2.4GHz·지도최대대각거리조건에서40m미만지붕의회절손실은이근사의0구간입니다.
        # 고도/주파수/크기를바꾸면이캐시최적화를그대로사용하면안됩니다.
        use=(scenario.altitude_m==50 and scenario.frequency_hz==2.4e9)
        keep=city.heights>=40 if use else np.ones(len(city.heights),bool)
        radio=BuildingRadio(np.asarray(city.geometries,dtype=object)[keep],city.heights[keep])
        p=np.column_stack([city.points,np.full(n,scenario.altitude_m)])
        for start in range(0,n,16):
            stop=min(n,start+16)
            tx=np.repeat(p[start:stop],n,axis=0);rx=np.tile(p,(stop-start,1))
            a,b,_=radio.propagation(tx,rx,scenario)
            loss[start:stop]=a.reshape(stop-start,n);los[start:stop]=b.reshape(stop-start,n)
        visible=np.zeros((n,len(city.target_points)),bool)
        for i in range(n):visible[i,city.sensing(p[i],scenario)]=True
        import os
        # 캐시를임시이름에완성한뒤교체해병렬프로세스가미완성캐시를읽는것을줄입니다.
        temp=file.with_name(file.stem+f'.tmp.{os.getpid()}.npz')
        np.savez_compressed(temp,loss=loss,los=los,visible=visible)
        temp.replace(file)
        city.loss_cache=loss;city.los_cache=los;city.visible=visible
    # 100%완료의분모는모든노드에서보이는고정셀집합입니다. 설계가잘안보는셀을매번빼는것이아닙니다.
    city.observable=np.any(city.visible,axis=0)
    matrix=city.graph.toarray()>0
    city.neighbors=np.full((len(city.points),9),-1,np.int32)
    for i in range(len(city.points)):
        v=np.r_[i,np.flatnonzero(matrix[i])];city.neighbors[i,:len(v)]=v
    return city

# 두 드론의 동시 선형 이동 a→b와 c→d에서 상대거리의 구간 최소값을 계산합니다.
# 끝점만 떨어져 있어도 이동 중 교차하면 거부합니다. minsep는 현재5m입니다.
@njit(cache=False)
def compatible(a,b,c,d,points,minsep):
    r=points[a]-points[c];dv=(points[b]-points[a])-(points[d]-points[c])
    dd=dv[0]*dv[0]+dv[1]*dv[1]
    t=0.0 if dd<1e-20 else min(1.0,max(0.0,-(r[0]*dv[0]+r[1]*dv[1])/dd))
    r=r+t*dv
    return r[0]*r[0]+r[1]*r[1]>=minsep*minsep-1e-8

# 우선 이동 드론의 목표 칸을 다른 드론이 막으면 그 드론도 움직이도록 재귀적으로 요청합니다.
# 충돌 없이 해결되지 않으면 snapshot으로 되돌리고 다른 이동 후보를 시도합니다.
# PIBT 아이디어를 변형한 코드이며 이 지도에서 항상 도달한다는 보장은 없습니다.
@njit(cache=False)
def inherit_move(i,nodes,goals,nxt,busy,occupant,neighbors,distance,points,minsep,depth):
    if depth>len(nodes):return False
    busy[i]=True
    options=neighbors[nodes[i]]
    costs=np.full(len(options),1e20)
    for j in range(len(options)):
        if options[j]>=0:costs[j]=distance[options[j],goals[i]]+0.001*(options[j]==nodes[i])
    order=np.argsort(costs)
    for j in order:
        node=options[j]
        if node<0:continue
        snapshot=nxt.copy();busy_snapshot=busy.copy();ok=True
        for k in range(len(nodes)):
            if k!=i and nxt[k]>=0 and not compatible(nodes[i],node,nodes[k],nxt[k],points,minsep):ok=False;break
        if not ok:continue
        nxt[i]=node
        other=occupant[node]
        if other>=0 and other!=i and nxt[other]<0:
            if busy[other]:ok=False
            else:ok=inherit_move(other,nodes,goals,nxt,busy,occupant,neighbors,distance,points,minsep,depth+1)
        if ok:
            busy[i]=False
            return True
        nxt[:]=snapshot;busy[:]=busy_snapshot
    nxt[i]=nodes[i];busy[i]=False
    return False

# 대기 나이와 목표 거리를 반영한 우선순위 순서로 전체 이동을 정합니다.
# 마지막으로 모든 드론 쌍의 연속 최소거리를 검사하고 실패하면 이번 이동 전체를 대기로 돌립니다.
@njit(cache=False)
def move_group(nodes,goals,priorities,neighbors,distance,points,minsep):
    nxt=np.full(len(nodes),-1,np.int32);busy=np.zeros(len(nodes),np.bool_)
    occupant=np.full(len(points),-1,np.int32)
    for i in range(len(nodes)):occupant[nodes[i]]=i
    for i in np.argsort(-priorities):
        if nxt[i]<0:inherit_move(i,nodes,goals,nxt,busy,occupant,neighbors,distance,points,minsep,0)
    for i in range(len(nodes)):
        if nxt[i]<0:nxt[i]=nodes[i]
    for i in range(len(nodes)):
        for j in range(i):
            if not compatible(nodes[i],nxt[i],nodes[j],nxt[j],points,minsep):return nodes.copy()
    return nxt

# Hungarian 할당으로 목표 슬롯을 서로 다른 안전 격자 노드에 매칭합니다.
# 가장 가까운 노드만 개별 선택하면 여러 드론이 같은 칸을 원할 수 있어 중복을 줄입니다.
def unique_goals(city,desired):
    cost=np.linalg.norm(desired[:,None,:]-city.points[None,:,:],axis=2)
    _,cols=linear_sum_assignment(cost)
    return cols.astype(np.int32)

# 전체 왕복 경로와 관측 가능한 셀100%를 완료한 뒤30초 공유하는 궤적을 만듭니다.
# 통신 성공 확률은 입력으로 받지 않으며, 잔여 셀은 새 관측 이득/(10+최단거리) 자체 탐욕 기준으로 탐색합니다.
# 7200초까지 끝나지 않으면 미완료로 기록합니다. 반환 nodes는 시각별 격자 노드, phase는0/1/2단계입니다.
def full_mission(city,design,scenario):
    offset=fleet_offsets(design,scenario.direction)
    if np.any(np.ptp(offset,axis=0)>380):raise ValueError('명목 대형 폭 380m 초과')
    reference=city.route[0].copy();cursor=1
    desired=reference+offset;nodes=unique_goals(city,desired)
    trajectories=[nodes.copy()];desired_log=[desired.copy()];phases=[0]
    observed=np.any(city.visible[nodes[:design.n_search]],axis=0)
    age=np.arange(len(nodes),dtype=float)*1e-3
    goal=-1;route_end=None;coverage_end=None;share_left=None
    mindistance=float('inf');travel=np.zeros(len(nodes));wait=0;clamps=0
    route_length=float(np.linalg.norm(np.diff(city.route,axis=0),axis=1).sum())
    for step in range(1,int(MAX_MISSION/MOVE_DT)+1):
        phase=0
        if cursor<len(city.route):
            remaining=scenario.reference_speed_mps*MOVE_DT
            while remaining>1e-9 and cursor<len(city.route):
                target=city.route[cursor];d=np.linalg.norm(target-reference)
                if d<=remaining+1e-9:reference=target.copy();remaining-=d;cursor+=1
                else:reference+=(target-reference)*remaining/d;remaining=0
            if cursor==len(city.route):route_end=step*MOVE_DT
        else:
            phase=1
            # 기준경로가끝났어도잔여관측가능셀을실제로찾아가며수색합니다. 공유단계는그뒤입니다.
            missing=city.observable&~observed
            if not np.any(missing):
                if coverage_end is None:coverage_end=(step-1)*MOVE_DT;share_left=int(SHARE_SECONDS/MOVE_DT)
                phase=2
            else:
                if goal<0 or nodes[0]==goal or not np.any(city.visible[goal]&missing):
                    gains=np.sum(city.visible[:,missing],axis=1)
                    # 새로관측할셀수와이동거리의탐욕절충입니다. 프로젝트자체휴리스틱이며통신예측재배치가아닙니다.
                    score=gains/(10.0+city.distance[nodes[0]])
                    goal=int(np.argmax(score))
                # anchor follows a safe shortest path to a remaining visible cell;
                # other drones keep their geometrical offsets to that anchor.
                reference=city.points[city.next_node(nodes[0],goal)].copy()-offset[0]
        desired=reference[None,:]+offset
        bounded=np.clip(desired,-200+scenario.grid_m/2,200-scenario.grid_m/2)
        clamps+=int(np.any(bounded!=desired,axis=1).sum())
        goals=unique_goals(city,bounded)
        if phase==1 and goal>=0:
            # Preserve cleanup anchor target rather than allow assignment to steal it.
            goals[0]=city.next_node(nodes[0],goal)
            if len(set(goals))<len(goals):goals=unique_goals(city,bounded)
        age+=1
        # 오래대기한드론에우선권을주고거리항으로순서를구분합니다. 성공적으로목표에도달하면나이를0으로초기화합니다.
        priorities=age+city.distance[nodes,goals]*0.001
        new=move_group(nodes,goals,priorities,city.neighbors,city.distance,city.points,scenario.min_separation_m)
        oldp,newp=city.points[nodes],city.points[new]
        minimum,_=swept_minimum(oldp,newp)
        if minimum<scenario.min_separation_m-1e-6:raise RuntimeError('연속 이동 분리 거리 위반')
        mindistance=min(mindistance,minimum)
        change=np.linalg.norm(newp-oldp,axis=1)
        if np.max(change)>scenario.speed_mps*MOVE_DT+1e-6:raise RuntimeError('최대 속도 위반')
        travel+=change;wait+=int(np.sum(change<1e-9));age[new==goals]=0
        nodes=new;observed|=np.any(city.visible[nodes[:design.n_search]],axis=0)
        trajectories.append(nodes.copy());desired_log.append(desired.copy());phases.append(phase)
        if phase==2:
            share_left-=1
            if share_left<=0:break
    # 비행임무완료와공유90%달성은별도입니다. 후자는full_run의mission_success에서검사합니다.
    completed=bool(route_end is not None and np.all(observed[city.observable]))
    if completed and coverage_end is None:coverage_end=(len(trajectories)-1)*MOVE_DT
    arr=np.asarray(trajectories,np.int32)
    # A graph edge is already swept-polygon checked in City; confirm each used edge.
    for a,b in zip(arr[:-1],arr[1:]):
        if np.any((a!=b)&(np.asarray(city.graph[a,b]).ravel()==0)):raise RuntimeError('비행 그래프 외 이동')
    return {'nodes':arr,'desired':np.asarray(desired_log),'phase':np.array(phases),
      'duration_s':(len(arr)-1)*MOVE_DT,'mission_completed':completed,
      'route_completion_s':route_end,'coverage_completion_s':coverage_end,'route_length_m':route_length,
      'minimum_swept_separation_m':mindistance,'total_travel_m':float(travel.sum()),
      'search_travel_m':float(travel[:design.n_search].sum()),'relay_travel_m':float(travel[design.n_search:].sum()),
      'stationary_agent_frames':wait,'boundary_clamp_agent_frames':clamps,'obstacle_violations':0}
