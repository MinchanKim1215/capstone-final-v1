"""슬롯 기반 패킷 큐·중계·재전송. 그래프 연결과 실제 전달을 분리한다.

TDMA는 이상적인 중앙 시간 동기화 가정이고, ALOHA는 간섭 민감도 비교다.
경로는 이상적인 주기적 전역 스냅샷에서 계산한다. OLSR 구현이 아니다.
수색 드론도 중계 가능하다. 전용 중계 드론만 센서 수집을 하지 않는다.
"""
# 초기 단위 검사와 수식 대조용 객체 기반 패킷 엔진입니다.
# 현재 512설계 전수 비교는 이 클래스 대신 fast_network.py를 사용합니다.
# 설정을 바꿀 수 있는 이 코드가 존재해도 현재 결과에 ALOHA·드론 이탈을 포함했다는 뜻은 아닙니다.
# 수색 드론도 중계할 수 있고, 목적지 수신만 종단 간 성공으로 계산합니다.
from dataclasses import dataclass
from collections import Counter
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .radio import packet_success, mw_to_dbm


# Jain=(Σx)²/(nΣx²) 공정성입니다. 이 초기 함수는 모두0이면 정의할 분모가 없어None을 반환합니다.
def jain(values):
    a=np.asarray(values,float)
    denominator=len(a)*np.sum(a*a)
    return float(a.sum()**2/denominator) if denominator>0 else None


# 패킷의 최초 송신자·원래 목적지·생성시각·셀ID·현재노드·재시도·홉수를 보관합니다.
@dataclass
class Packet:
    packet_id:int
    src:int
    dst:int
    created:float
    cells:tuple
    current:int
    hop_attempts:int=0
    hops:int=0
    relay_hops:int=0


# 초기 객체 기반 엔진입니다. 현재 전체 실험은 fast_network.packet_run을 사용합니다.
class PacketNetwork:
    # 방향별 수색 흐름, 노드별 큐, 라우팅 표, 지연·전송·AoI 집계 상태를 초기화합니다.
    def __init__(self,design,scenario,rng,known,capture=False):
        self.d,self.s,self.rng=design,scenario,rng
        self.known=known
        self.queues=[[] for _ in range(design.n_drones)]
        self.next_hop=np.full((design.n_drones,design.n_drones),-1,int)
        self.active=np.ones(design.n_drones,bool)
        self.next_packet=0; self.generated=0; self.delivered=0
        self.drops=Counter(); self.flow_generated=Counter(); self.flow_delivered=Counter()
        self.delay=[]; self.hops=[]; self.relay_hops=[]
        self.delivered_timely=0;self.delivered_in_window=0
        self.tx_attempts=0;self.hop_successes=0;self.retries=0
        self.no_route_slots=0;self.route_updates=0;self.max_queue=0
        self.last_route_time=-math.inf;self.next_created=0.0;self.slot_index=0
        self.rate=(scenario.total_packets_per_s if scenario.traffic_mode=="fixed_total"
                   else scenario.per_search_packets_per_s*design.n_search)
        self.interval=1/self.rate
        ns=design.n_search
        self.flows=[(i,(i+k)%ns) for k in range(1,ns) for i in range(ns)]
        self.last_timestamp=np.zeros((ns,ns))
        self.ever_received=np.eye(ns,dtype=bool)
        self.aoi_sum=0.0;self.aoi_count=0;self.aoi_peak=0.0
        self.cell_cursor=np.zeros(ns,int)
        self.capture=capture;self.events=[]

    # 패킷을 원인별로 유실 처리합니다. 화면 캡처 시 원인도 기록합니다.
    def drop(self,p,reason):
        self.drops[reason]+=1
        if self.capture:
            self.events.append({"kind":"drop","id":p.packet_id,"reason":reason})

    # 큐64개 한도를 검사하고 오래된 생성시각부터 처리하도록 삽입합니다. TTL이 같아EDF순서와 같습니다.
    def enqueue(self,node,p):
        if len(self.queues[node])>=self.s.queue_limit:
            self.drop(p,"queue_overflow");return False
        # 생성 시각 순서의 EDF 큐. 오래된 중계 패킷을 먼저 처리한다.
        q=self.queues[node];key=(p.created,p.packet_id)
        k=len(q)
        while k and (q[k-1].created,q[k-1].packet_id)>key:k-=1
        q.insert(k,p)
        self.max_queue=max(self.max_queue,len(q));return True

    # 초기 엔진의 이탈 진단용입니다. 해당 드론 큐를 유실 처리하고 경로를 다시 계산하게 합니다.
    def fail_node(self,node):
        self.active[node]=False
        for p in self.queues[node]:self.drop(p,"node_failure")
        self.queues[node].clear();self.last_route_time=-math.inf

    # 현재 가용 그래프에서 최소홉 또는 성공확률 역수 비용 최단 경로를 만듭니다.
    # 이상적인 전역 링크 정보 스냅샷이며 라우팅 제어 패킷 교환은 구현하지 않습니다.
    def update_routes(self,state,t):
        if t-self.last_route_time < self.s.route_update_s-1e-10:return
        adj=state["adjacency"] & self.active[:,None] & self.active[None,:]
        ii,jj=np.nonzero(adj)
        costs=np.ones(len(ii)) if self.s.routing=="hops" else 1/np.maximum(state["ps_hop"][ii,jj],1e-9)
        graph=csr_matrix((costs,(ii,jj)),shape=adj.shape)
        _,pred=shortest_path(graph,directed=True,return_predecessors=True)
        # 뒤집은 경로의 predecessor는 목적지 방향의 첫 홉이다. 여기서는 명시적으로 역추적한다.
        self.next_hop.fill(-1)
        for source in range(len(adj)):
            for dest in range(len(adj)):
                if source==dest:continue
                node=dest
                for _ in range(len(adj)):
                    previous=int(pred[source,node])
                    if previous<0:break
                    if previous==source:
                        self.next_hop[source,dest]=node;break
                    node=previous
        self.last_route_time=t;self.route_updates+=1

    # 정해진 생성 시각까지 새 패킷을 만들고 수색 드론이 실제 알고 있는 셀ID만 담습니다.
    def generate_until(self,t):
        while self.next_created<=t+1e-10 and self.next_created<self.s.duration_s-1e-10:
            src,dst=self.flows[self.next_packet%len(self.flows)]
            cells=sorted(self.known[src]);count=min(len(cells),self.s.packet_payload_bytes//8)
            if count:
                start=self.cell_cursor[src]%len(cells)
                selected=tuple(cells[(start+k)%len(cells)] for k in range(count))
                self.cell_cursor[src]+=count
            else:selected=()
            p=Packet(self.next_packet,src,dst,self.next_created,selected,src)
            self.next_packet+=1;self.generated+=1;self.flow_generated[src,dst]+=1
            self.enqueue(src,p);self.next_created+=self.interval

    # 슬롯별 유실·송신·성공/실패·중계·목적지수신·AoI를 처리합니다.
    # ALOHA 분기는 초기 민감도 코드이고 현재 선정 실험에서는 이상적TDMA만 사용했습니다.
    def advance(self,start,end,state,generate=True):
        s=self.s; n=self.d.n_drones
        self.update_routes(state,start)
        slots=int(round((end-start)/s.slot_s))
        for k in range(slots):
            t=start+k*s.slot_s
            if generate:self.generate_until(t)
            for q in self.queues:
                while q and t-q[0].created>=s.packet_lifetime_s-1e-10:
                    self.drop(q.pop(0),"expired")
            candidates=[]
            chosen=[self.slot_index%n] if s.mac=="tdma" else list(np.flatnonzero(self.rng.random(n)<s.aloha_attempt_probability))
            self.slot_index+=1
            for src in chosen:
                if not self.active[src] or not self.queues[src]:continue
                p=self.queues[src][0];dest=int(self.next_hop[src,p.dst])
                if dest<0 or not self.active[dest]:
                    self.no_route_slots+=1;continue
                candidates.append((src,dest,p))
            transmitters={a for a,_,_ in candidates};outcomes=[]
            for src,dest,p in candidates:
                self.tx_attempts+=1;self.retries+=int(p.hop_attempts>0);p.hop_attempts+=1
                if s.mac=="tdma":prob=float(state["ps_hop"][src,dest])
                else:
                    others=[a for a in transmitters if a!=src]
                    inter=float(state["rx_mw"][others,dest].sum()) if others else 0.0
                    sinr=mw_to_dbm(state["rx_mw"][src,dest]/(state["denominator_mw"][dest]+inter))
                    pdata=float(packet_success(sinr,8*(s.packet_payload_bytes+s.packet_header_bytes),s.bandwidth_hz,s.bit_rate_bps))
                    pack=float(packet_success(state["sinr_db"][dest,src],8*s.ack_bytes,s.bandwidth_hz,s.bit_rate_bps))
                    prob=0.0 if dest in transmitters else pdata*pack
                success=bool(self.rng.random()<prob)
                outcomes.append((src,dest,p,success))
                if self.capture:
                    self.events.append({"kind":"hop","t":round(t+s.slot_s,6),"id":p.packet_id,
                                        "from":src,"to":dest,"success":success})
            # 모든 동시 시도의 결과를 먼저 정한 뒤 큐를 갱신한다. 같은 슬롯에서 연속 중계하지 않는다.
            for src,dest,p,success in outcomes:
                if success:
                    self.queues[src].pop(0);self.hop_successes+=1
                    p.hops+=1;p.relay_hops+=int(src>=self.d.n_search);p.current=dest;p.hop_attempts=0
                    arrival=t+s.slot_s
                    if arrival-p.created>s.packet_lifetime_s+1e-10:
                        self.drop(p,"expired");continue
                    if dest==p.dst:
                        self.delivered+=1;self.flow_delivered[p.src,p.dst]+=1
                        delay=arrival-p.created;self.delay.append(delay);self.hops.append(p.hops);self.relay_hops.append(p.relay_hops)
                        self.delivered_timely+=int(delay<=s.timely_deadline_s+1e-10)
                        self.delivered_in_window+=int(arrival<=s.duration_s+1e-10)
                        self.known[dest].update(p.cells)
                        self.last_timestamp[p.src,dest]=max(self.last_timestamp[p.src,dest],p.created)
                        self.ever_received[p.src,dest]=True
                    else:self.enqueue(dest,p)
                elif p.hop_attempts>=s.max_hop_attempts:
                    self.queues[src].pop(0);self.drop(p,"retry_limit")
        if end<=s.duration_s+1e-10:
            ages=np.array([end-self.last_timestamp[a,b] for a,b in self.flows])
            self.aoi_sum+=float(ages.mean());self.aoi_count+=1
            self.aoi_peak=max(self.aoi_peak,float(ages.max()))

    # 생성 패킷 보존식을 확인하고 PDR·성공 조건부 지연·홉·공정성·유실·AoI를 분리해 반환합니다.
    def finalize(self):
        pending=sum(map(len,self.queues))
        if self.generated!=self.delivered+sum(self.drops.values())+pending:
            raise RuntimeError("패킷 보존식이 깨졌습니다.")
        ratios=[self.flow_delivered[f]/self.flow_generated[f] for f in self.flows if self.flow_generated[f]]
        delay=np.array(self.delay)
        return {
            "packets_generated":self.generated,"packets_delivered":self.delivered,"packets_pending":pending,
            "pdr":self.delivered/self.generated if self.generated else None,
            "timely_pdr":self.delivered_timely/self.generated if self.generated else None,
            "goodput_kbps":self.delivered_in_window*self.s.packet_payload_bytes*8/self.s.duration_s/1000,
            "delay_mean_s":float(delay.mean()) if len(delay) else None,
            "delay_p95_s":float(np.percentile(delay,95)) if len(delay) else None,
            "delay_max_s":float(delay.max()) if len(delay) else None,
            "delivered_hops_mean":float(np.mean(self.hops)) if self.hops else None,
            "dedicated_relay_hops_mean":float(np.mean(self.relay_hops)) if self.relay_hops else None,
            "flow_pdr_jain":jain(ratios),"flow_pdr_min":float(min(ratios)) if ratios else None,
            "flow_pdr_std":float(np.std(ratios)) if ratios else None,
            "tx_attempts":self.tx_attempts,"hop_successes":self.hop_successes,
            "hop_success_rate":self.hop_successes/self.tx_attempts if self.tx_attempts else None,
            "retries":self.retries,"tx_per_delivered":self.tx_attempts/self.delivered if self.delivered else None,
            "wire_traffic_kbytes":self.tx_attempts*(self.s.packet_payload_bytes+self.s.packet_header_bytes+self.s.ack_bytes)/1000,
            "queue_max_packets":self.max_queue,"no_route_slots":self.no_route_slots,"route_updates":self.route_updates,
            "drop_expired":self.drops["expired"],"drop_retry_limit":self.drops["retry_limit"],
            "drop_queue_overflow":self.drops["queue_overflow"],"drop_node_failure":self.drops["node_failure"],
            "aoi_mean_s":self.aoi_sum/self.aoi_count if self.aoi_count else None,"aoi_peak_s":self.aoi_peak,
            "never_received_flow_fraction":float(np.mean([not self.ever_received[a,b] for a,b in self.flows])),
        }
