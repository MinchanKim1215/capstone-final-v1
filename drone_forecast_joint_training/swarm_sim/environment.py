"""지형→관측→무선 링크→패킷→이동을 연결하는 기본 시뮬레이터.

예측·강화학습·통신 위험 기반 재배치는 포함하지 않는다.
같은 seed는 다른 설계에도 동일한 간섭원 위치·채널 난수 접두부를 제공한다.
"""
# 초기 짧은 시간 시험용 step 환경입니다. 현재 전체 임무 실험의 실행 입구가 아닙니다.
# 현재 결과는 full_mission/full_run에서 만들어졌으므로 이 파일만 바꿔도 전체 결과가 바뀌지는 않습니다.
# 관측·행동·다음 상태 API를 설계할 때 참고할 수 있지만 미래의 예측·재배치 학습 환경은 별도 통합이 필요합니다.
from dataclasses import asdict
import numpy as np
from .city import City
from .motion import FormationMotion
from .network import PacketNetwork
from .metrics import EpisodeMetrics


# 초기 step 환경입니다. 현재 전체 임무는 이 클래스가 아니라 full_run에서 실행합니다.
class SwarmEnvironment:
    # 배치·조건을 검사하고 지도·난수·캡처 옵션을 준비합니다.
    def __init__(self,map_path,design,scenario,seed=0,capture=False,city=None):
        self.d,self.s=design.validate(),scenario.validate()
        self.seed=int(seed);self.city=city or City(map_path,scenario);self.capture=capture
        self.reset()

    # 채널·간섭·MAC 난수를 분리해 초기화하고 수색 관측과 지식 집합을 비웁니다.
    def reset(self):
        child=np.random.SeedSequence(self.seed).spawn(3)
        self.channel_rng=np.random.default_rng(child[0]);self.external_rng=np.random.default_rng(child[1])
        self.mac_rng=np.random.default_rng(child[2])
        self.motion=FormationMotion(self.city,self.d,self.s)
        self.observed=[set() for _ in range(self.d.n_search)]
        self.known=[set() for _ in range(self.d.n_drones)]
        self.network=PacketNetwork(self.d,self.s,self.mac_rng,self.known,self.capture)
        self.metrics=EpisodeMetrics(self.d,self.s,self.city)
        p=self.city.target_points[self.external_rng.integers(len(self.city.target_points))]
        self.jammer_xyz=np.r_[p,self.s.jammer_height_m]
        self.shadow=np.zeros((10,10));self.t=0.0;self.frames=[];self.link_rows=[];self.failed=False
        self.last_state=None;self.last_state_time=None
        return self.observation()

    # 개발용 전역 정보를 돌려줍니다. 실제 분산 정책에 그대로 제공하면 이상적 정보 가정을 만들게 됩니다.
    def observation(self):
        # 개발용 전역 관측이다. 향후 분산 알고리즘에는 실제 수신한 로컬 정보만 노출해야 한다.
        return {"t":self.t,"positions":self.motion.positions_3d().copy(),
                "roles":["search"]*self.d.n_search+["relay"]*self.d.n_relays,
                "state":self.last_state,"state_time":self.last_state_time,
                "state_age_s":self.t-self.last_state_time if self.last_state_time is not None else None}

    # 센서 관측 → 채널 → 패킷 전달 → 이동 → 지표 기록의 초기 한 주기입니다.
    # 반환 지표는 그 주기 처리 결과이므로 같은 시각의 사전 예측 입력과 구분합니다.
    def step(self,target_positions=None):
        s=self.s;d=self.d;t=self.t
        if t>=s.duration_s-1e-10:raise RuntimeError("실험 구간이 끝났습니다.")
        if s.relay_failure_fraction is not None and d.n_relays and not self.failed and t>=s.relay_failure_fraction*s.duration_s:
            node=d.n_search;self.motion.active[node]=False;self.network.fail_node(node);self.failed=True
        position=self.motion.positions_3d()
        for i in range(d.n_search):
            cells=self.city.sensing(position[i],s)
            self.observed[i].update(map(int,cells));self.known[i].update(map(int,cells))
        rho=np.exp(-s.motion_dt_s/s.shadow_correlation_s)
        draw=self.channel_rng.normal(size=(10,10));draw=(draw+draw.T)/np.sqrt(2)
        self.shadow=rho*self.shadow+np.sqrt(1-rho*rho)*s.shadow_sigma_db*draw
        np.fill_diagonal(self.shadow,0)
        on=(s.jammer_power_dbm is not None and s.jammer_start_fraction*s.duration_s<=t<s.jammer_end_fraction*s.duration_s)
        if on:on=bool(self.external_rng.random()<s.jammer_duty)
        state=self.city.radio.state(position,s,self.shadow[:d.n_drones,:d.n_drones],self.jammer_xyz,on)
        state["adjacency"] &= self.motion.active[:,None]&self.motion.active[None,:]
        self.network.advance(t,t+s.motion_dt_s,state)
        old,new,desired=self.motion.step(target_positions)
        self.metrics.push(t,state,self.motion,old,new,desired,self.observed,self.known,self.network,on)
        self.last_state=state;self.last_state_time=t
        if self.capture:
            self.frames.append({"t":round(t,6),"positions":old.tolist(),"desired":desired.tolist(),
                                "edges":np.argwhere(np.triu(state["adjacency"],1)).tolist(),
                                "observed_cells":sorted(set().union(*self.observed)),
                                "metrics":self.metrics.rows[-1]})
            for i,j in zip(*np.triu_indices(d.n_drones,1)):
                self.link_rows.append({"t":round(t,6),"i":int(i),"j":int(j),
                                       "distance_m":float(np.linalg.norm(position[i]-position[j])),
                                       "role_i":"search" if i<d.n_search else "relay",
                                       "role_j":"search" if j<d.n_search else "relay",
                                       "rx_ij_dbm":float(state["rx_dbm"][i,j]),"rx_ji_dbm":float(state["rx_dbm"][j,i]),
                                       "los":int(state["los"][i,j]),"path_loss_db":float(state["loss_db"][i,j]),
                                       "sinr_ij_db":float(state["sinr_db"][i,j]),"sinr_ji_db":float(state["sinr_db"][j,i]),
                                       "ps_ij":float(state["ps_hop"][i,j]),"ps_ji":float(state["ps_hop"][j,i]),
                                       "usable":int(state["adjacency"][i,j]),"jammer_on":int(on)})
        self.t+=s.motion_dt_s
        return self.observation(),self.metrics.rows[-1],self.t>=s.duration_s-1e-10

    # 고정 시간의 초기 시험을 끝내고 새 패킷 생성을 멈춘 뒤 기존 패킷을 TTL까지 정산합니다.
    def run(self):
        while self.t<self.s.duration_s-1e-10:self.step()
        # 생성된 패킷의 운명을 비교할 수 있도록 생성 중지 후 TTL까지 배출한다.
        # 배출 구간은 위치와 채널을 고정하며 수색/연결성 지표 구간에는 넣지 않는다.
        final=self.city.radio.state(self.motion.positions_3d(),self.s,self.shadow[:self.d.n_drones,:self.d.n_drones],self.jammer_xyz,False)
        final["adjacency"] &= self.motion.active[:,None]&self.motion.active[None,:]
        end=self.t+self.s.packet_lifetime_s+self.s.motion_dt_s
        while self.t<end-1e-10:
            self.network.advance(self.t,self.t+self.s.motion_dt_s,final,generate=False);self.t+=self.s.motion_dt_s
        metrics={**self.metrics.finalize(self.motion),**self.network.finalize()}
        if metrics["packets_pending"]:raise RuntimeError("배출 이후에도 패킷이 남았습니다.")
        return {"region":self.city.key,"seed":self.seed,"design":asdict(self.d),"scenario":asdict(self.s),
                "map":self.city.summary(),"jammer_xyz":self.jammer_xyz.tolist(),"metrics":metrics,
                "frames":self.frames,"link_rows":self.link_rows,"packet_events":self.network.events,
                "target_points":self.city.target_points.tolist() if self.capture else [],
                "effective_heights_m":self.city.heights.tolist() if self.capture else [],
                "no_fly_geometry":self.city.flight_blocker.__geo_interface__ if self.capture else None}
