"""기하학적 대형 추종, 장애물 우회와 충돌 대기. 통신 위험 기반 이동은 하지 않는다."""
# formation_offsets와 fleet_offsets는 현재 전체 임무에서도 쓰는 명목 배치 계산입니다.
# swept_minimum은 움직이는 두 드론의 구간 전체 최소거리를 계산합니다.
# 아래 FormationMotion 클래스는 초기 짧은 시험용이며, 현재 전체 수색은 mission.py를 사용합니다.
# 명목 간격은 장애물 없는 기준 슬롯의 간격이고, 실제 비행 중 거리는 따로 측정합니다.
import numpy as np
import shapely

# 대형 중심에서 각 슬롯까지의 평면 좌표를 만듭니다. 횡렬·종렬·V·격자·원형의5종입니다.
# 원형의 spacing은 인접점 현 길이이며, 모든 대형을 평균 중심0으로 옮깁니다.
def formation_offsets(n, spacing, name):
    if name == "line_cross":
        p = np.column_stack([np.zeros(n),np.arange(n)*spacing])
    elif name == "line_along":
        p = np.column_stack([np.arange(n)*spacing,np.zeros(n)])
    elif name == "v":
        p = np.array([[0.,0.]]+[[-((i+1)//2),(-1 if i%2 else 1)*((i+1)//2)] for i in range(1,n)])*spacing/np.sqrt(2)
    elif name == "grid":
        width = int(np.ceil(np.sqrt(n)))
        p = np.array([[i%width,i//width] for i in range(n)],float)*spacing
    elif name == "ring":
        radius = spacing/(2*np.sin(np.pi/n))
        angle = np.arange(n)*2*np.pi/n
        p = radius*np.column_stack([np.cos(angle),np.sin(angle)])
    else:
        raise ValueError(name)
    return p-p.mean(axis=0)

# 수색 슬롯을 먼저, 전용 중계 슬롯을 뒤에 배치해 역할 인덱스를 통일합니다.
# 현재 search/backbone에서는 수색군 대형과 가장 먼 수색쌍 사이의 중계 슬롯을 함께 만듭니다.
# backbone은 기준 배치 이름이고 네트워크 프로토콜 이름이 아닙니다.
def fleet_offsets(design, direction):
    ns,nr = design.n_search,design.n_relays
    if design.formation_scope == "fleet":
        full = formation_offsets(design.n_drones,design.spacing_m,design.formation)
        # 전용 중계 슬롯은 전체 대형을 따라 균등 분산한다. 앞 Ns개가 수색 역할이다.
        relays = sorted({int(i*design.n_drones/nr) for i in range(nr)}) if nr else []
        search = [i for i in range(design.n_drones) if i not in relays]
        offsets = full[search+relays]
    else:
        search = formation_offsets(ns,design.spacing_m,design.formation)
        relay = []
        if nr and design.relay_layout == "center":
            width = int(np.ceil(np.sqrt(nr)))
            relay = np.array([[i%width,i//width] for i in range(nr)],float)*10.0
            relay -= relay.mean(axis=0); relay[:,1] += 8.0
        elif nr:
            pair = np.unravel_index(np.argmax(np.linalg.norm(search[:,None]-search[None,:],axis=2)),(ns,ns))
            first,last = search[pair[0]],search[pair[1]]
            vector = last-first; unit = vector/max(np.linalg.norm(vector),1e-9)
            normal = np.array([-unit[1],unit[0]])
            relay = np.array([first+(i+1)/(nr+1)*vector+normal*(8.0 if i%2==0 else -8.0) for i in range(nr)])
        offsets = np.vstack([search,np.asarray(relay).reshape(-1,2)]) if nr else search
    if direction == "north_south":
        offsets = offsets[:,[1,0]]
    return offsets

# r(τ)=r₀+τΔr에서 τ*=clip(−r₀·Δr/|Δr|²,0,1)을 사용해 이동 구간의 최소거리를 계산합니다.
def swept_minimum(old,new,active=None):
    ids = np.arange(len(old)) if active is None else np.flatnonzero(active)
    if len(ids)<2:
        return float("inf"),None
    a,b = np.triu_indices(len(ids),1); ii,jj=ids[a],ids[b]
    relative = old[ii]-old[jj]
    delta = (new[ii]-old[ii])-(new[jj]-old[jj])
    norm = np.sum(delta*delta,axis=1)
    tau = np.clip(-np.sum(relative*delta,axis=1)/np.maximum(norm,1e-20),0,1)
    distance = np.linalg.norm(relative+tau[:,None]*delta,axis=1)
    k = int(np.argmin(distance))
    return float(distance[k]),(int(ii[k]),int(jj[k]))

# 초기 짧은 시험용 이동 클래스입니다. 현재 전체 수색은 mission.full_mission을 사용합니다.
class FormationMotion:
    # 기준 경로의 시작점과 대형 슬롯을 만들고 안전 간격을 만족하는 초기 노드를 할당합니다.
    def __init__(self, city,design,scenario):
        self.city,self.design,self.scenario = city,design,scenario
        self.offsets = fleet_offsets(design,scenario.direction)
        if np.any(np.ptp(self.offsets,axis=0)>380):
            raise ValueError("명목 대형의 폭이 380m를 넘어 400m 지도에 배치할 수 없습니다.")
        lengths=np.linalg.norm(np.diff(city.route,axis=0),axis=1)
        cumulative=np.r_[0,np.cumsum(lengths)]
        target=scenario.route_start_fraction*cumulative[-1]
        segment=min(int(np.searchsorted(cumulative,target,side="right")-1),len(lengths)-1)
        alpha=(target-cumulative[segment])/max(lengths[segment],1e-12)
        self.reference=city.route[segment]*(1-alpha)+city.route[segment+1]*alpha
        self.route_cursor=segment+1
        desired = self.reference[None,:]+self.offsets
        used=[]; node_ids=[]
        for point in desired:
            order = np.argsort(np.linalg.norm(city.points-point,axis=1))
            selected = next((int(i) for i in order if all(np.linalg.norm(city.points[i]-city.points[j])>=scenario.min_separation_m for j in used)),None)
            if selected is None:
                raise ValueError("안전 간격을 만족하는 초기 배치를 찾지 못했습니다.")
            used.append(selected);node_ids.append(selected)
        self.nodes=np.array(node_ids,int)
        self.next_nodes=self.nodes.copy()
        self.positions=city.points[self.nodes].copy()
        self.active=np.ones(design.n_drones,dtype=bool)
        self.distance=np.zeros(design.n_drones)
        self.wait_count=0;self.boundary_clamps=0
        self.path_cursor_at_end=False

    # 기준점을 정해진 속도로 경로 위에서 진행시킵니다. 통신 상태는 보지 않습니다.
    def advance_reference(self,dt):
        remaining=self.scenario.reference_speed_mps*dt
        while remaining>1e-10 and self.route_cursor<len(self.city.route):
            target=self.city.route[self.route_cursor]
            vector=target-self.reference;distance=np.linalg.norm(vector)
            if distance<=remaining+1e-10:
                self.reference=target.copy();remaining-=distance;self.route_cursor+=1
            else:
                self.reference+=vector/distance*remaining;remaining=0
        self.path_cursor_at_end=self.route_cursor>=len(self.city.route)

    # 명목 목표를 안전 격자로 옮겨 이동하고, 충돌 구간은 대기로 막습니다.
    # 현재 전체 임무의 협력 이동과 다른 초기 방식이므로 이 함수의 결과를 현재 전수 결과와 혼합하지 않습니다.
    def step(self,target_positions=None):
        s=self.scenario;old=self.positions.copy()
        self.advance_reference(s.motion_dt_s)
        desired=self.reference[None,:]+self.offsets if target_positions is None else np.asarray(target_positions,float)
        if desired.shape!=self.positions.shape or not np.isfinite(desired).all():
            raise ValueError("각 드론의 목표 평면 좌표 N×2가 필요합니다.")
        half_grid=s.grid_m/2
        clamped=np.clip(desired,np.array(self.city.bounds[:2])+half_grid,
                        np.array(self.city.bounds[2:])-half_grid)
        self.boundary_clamps+=int(np.any(clamped!=desired,axis=1).sum())
        proposals=old.copy()
        for i in range(len(old)):
            if not self.active[i]:continue
            if np.linalg.norm(old[i]-self.city.points[self.next_nodes[i]])<1e-8:
                self.nodes[i]=self.next_nodes[i]
                goal=self.city.nearest(clamped[i])
                self.next_nodes[i]=self.city.next_node(self.nodes[i],goal)
            target=self.city.points[self.next_nodes[i]]
            vector=target-old[i];distance=np.linalg.norm(vector)
            if distance>1e-10:
                proposals[i]=old[i]+vector/distance*min(distance,s.speed_mps*s.motion_dt_s)
        # 한 프레임의 동시 선형 이동 구간을 검사해 교차·추월 충돌도 막는다.
        for _ in range(len(old)*len(old)):
            minimum,pair=swept_minimum(old,proposals,self.active)
            if minimum>=s.min_separation_m-1e-8:break
            i,j=pair
            proposals[i]=old[i];proposals[j]=old[j]
        if swept_minimum(old,proposals,self.active)[0]<s.min_separation_m-1e-8:
            proposals=old.copy()
        lines=shapely.linestrings(np.stack([old,proposals],axis=1))
        moved=np.linalg.norm(proposals-old,axis=1)>1e-8
        if np.any(shapely.intersects(lines[moved],self.city.flight_blocker)):
            raise RuntimeError("장애물을 통과하는 이동 구간이 생성됐습니다.")
        self.wait_count+=int((self.active & (np.linalg.norm(proposals-old,axis=1)<1e-9)).sum())
        self.distance+=np.linalg.norm(proposals-old,axis=1)
        self.positions=proposals
        return old,proposals,desired

    # 평면 위치에 공통 고도 열을 붙여 채널·센서가 쓰는 N×3좌표로 바꿉니다.
    def positions_3d(self):
        return np.column_stack([self.positions,np.full(len(self.positions),self.scenario.altitude_m)])
