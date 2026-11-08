"""이상적 TDMA의 실제 추상 패킷 큐와 재시도 및 목적지 수신을 계산한다.

IP/Wi-Fi 구현은 아니다. 슬롯 5ms, 고정 총 30packet/s, EDF FIFO,
큐64, 홉별4회, 수명2s, 64개 셀 ID 페이로드라는 명시된 가정이다.
"""
# 현재 전체 임무용 고속 패킷 엔진입니다. Numba가 반복 계산을 컴파일합니다.
# probabilities: 건물 손실·외부 간섭·시간 상관 섀도잉에서 홉 성공 확률을 만듭니다.
# packet_run: 5ms TDMA 슬롯마다 생성·큐 대기·중계·재시도·수신·유실을 처리합니다.
# 이 파일의 10dBm·−107dBm·4608bit·112bit 등은 고정 비교 조건이며 full_run에서 변경을 차단합니다.
# IP/Wi-Fi/OLSR/AODV 스택과 전역 링크 정보 공유 비용은 모사하지 않습니다.
import math
import numpy as np
from numba import njit

# 고정 B/Rb=1 조건의 BPSK 패킷 성공 확률입니다. db는SINR(dB), n_bits는데이터 또는ACK길이입니다.
@njit(cache=True)
def bpsk_probability(db,n_bits):
    eb=10.0**(max(-100.,min(100.,db))/10.0)
    ber=0.5*math.erfc(math.sqrt(eb))
    return math.exp(n_bits*math.log1p(-min(ber,0.5)))

# 배열 loss[시간,송신,수신]과 interference[시간,수신]에서 홉 성공 확률을 계산합니다.
# 표준편차1.5dB·상관시간4초의 섀도잉을 넣고 전방 데이터4608bit×역방향ACK112bit의 성공을 곱합니다.
# seed가 같아도 N에 따라 난수 사용 순서가 달라지므로 모든 링크의 잡음이 설계 간 똑같다고 보장하지는 않습니다.
@njit(cache=True)
def probabilities(loss,interference,dt,seed,sigma=1.5,tau=4.,received_log=None,sinr_log=None):
    np.random.seed(seed)
    nt,n,_=loss.shape;p=np.zeros_like(loss)
    # rho=exp(−dt/tau), X_t=rho·X_(t−1)+sigma·sqrt(1−rho²)·정규난수의시간상관모델입니다.
    shadow=np.zeros((n,n));rho=math.exp(-dt/tau);sd=sigma*math.sqrt(1-rho*rho)
    for t in range(nt):
        rx=np.zeros((n,n));den=np.zeros(n)
        for j in range(n):den[j]=10.0*math.log10(10**(-107/10)+interference[t,j])
        for i in range(n):
            for j in range(i):
                shadow[i,j]=rho*shadow[i,j]+sd*np.random.normal()
                shadow[j,i]=shadow[i,j]
        for i in range(n):
            for j in range(n):
                rx[i,j]=10.-loss[t,i,j]-shadow[i,j]-den[j]
                # 선택적 학습 로그입니다. 난수 추출·성공 확률·기존 반환값은 바꾸지 않습니다.
                if received_log is not None:received_log[t,i,j]=10.-loss[t,i,j]-shadow[i,j]
                if sinr_log is not None:sinr_log[t,i,j]=rx[i,j]
        for i in range(n):
            for j in range(n):
                if i!=j:p[t,i,j]=bpsk_probability(rx[i,j],4608)*bpsk_probability(rx[j,i],112)
    return p

# 양방향 홉 성공 확률≥0.9를 가용 링크로 두고 전역 그래프를 만듭니다.
# Floyd–Warshall로 최소홉과 Σ(1/성공확률) 최소 비용 경로를 계산합니다.
# 수색군 전체 연결과 모든N대 연결을 구분하며, 연결되어도 큐·수명 때문에 PDR은 더 낮을 수 있습니다.
@njit(cache=True)
def graph_statistics(p,ns):
    n=len(p);a=(p>=.9)&(p.T>=.9)
    hops=np.full((n,n),1e9);distance=np.full((n,n),1e9);nxt=np.full((n,n),-1,np.int32)
    for i in range(n):
        hops[i,i]=0;distance[i,i]=0
        for j in range(n):
            if a[i,j]:hops[i,j]=1;distance[i,j]=1/max(p[i,j],1e-20);nxt[i,j]=j
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if hops[i,k]+hops[k,j]<hops[i,j]:hops[i,j]=hops[i,k]+hops[k,j]
                if distance[i,k]+distance[k,j]<distance[i,j]:
                    distance[i,j]=distance[i,k]+distance[k,j];nxt[i,j]=nxt[i,k]
    label=np.full(n,-1,np.int32);sizes=np.zeros(n);components=0
    for i in range(n):
        if label[i]<0:
            for j in range(n):
                if hops[i,j]<1e8:label[j]=components;sizes[components]+=1
            components+=1
    same=True;pair=0.;eff=0.;hh=0.;hc=0;edges=0;isolated=0
    for i in range(n):
        degree=0
        for j in range(n):
            if a[i,j]:degree+=1;edges+=1
            if i!=j and hops[i,j]<1e8:eff+=1/hops[i,j]
        if degree==0:isolated+=1
    for i in range(ns):
        for j in range(i):
            if hops[i,j]<1e8:pair+=1;hh+=hops[i,j];hc+=1
            else:same=False
    stats=np.array([1.0 if same else 0.0,pair/(ns*(ns-1)/2),1.0 if components==1 else 0.0,components,
       np.max(sizes)/n,edges/n,isolated/n,edges/(n*(n-1)),eff/(n*(n-1)),hh/max(hc,1)])
    return nxt,stats,a

# 노드 큐의 첫 패킷을 제거하고 뒤 원소를 당깁니다. 고정64개 배열 큐를 위한 연산입니다.
@njit(cache=True)
def pop_front(q,lengths,node):
    for i in range(1,lengths[node]):q[node,i-1]=q[node,i]
    lengths[node]-=1

# 5ms TDMA에서 한 슬롯에 하나의 드론만 송신합니다. 총30packet/s를 수색 방향 쌍에 순환 배정합니다.
# 패킷은64개 셀ID를 담고 실제 목적지 수신 때만 그 드론의 지식에 반영됩니다.
# TTL2초·홉당4회·큐64개를 처리하고 생성=전달+수명초과+재시도한도+큐초과를 검사할 값을 반환합니다.
# disable_relays는 같은 궤적·슬롯에서 전용 중계 무선만 끄는 진단이며 총대수를 줄이는 실험은 아닙니다.
@njit(cache=True)
def packet_run(p,node_ids,visible,ns,dt,seed,disable_relays=False,
               attempt_log=None,success_log=None,last_ack_log=None,queue_log=None):
    np.random.seed(seed)
    nt,n,_=p.shape;nc=visible.shape[1];slots=int(round(dt/.005))
    # 송신자가 직접 알 수 있는 시도·ACK 기록을 별도 버퍼로 수집합니다.
    # 지정하지 않으면 기존 실행과 같은 계산·난수 순서·반환 형식을 유지합니다.
    if attempt_log is not None:attempt_log.fill(0)
    if success_log is not None:success_log.fill(0)
    if last_ack_log is not None:last_ack_log.fill(-1.)
    if queue_log is not None:queue_log.fill(0)
    duration=nt*dt;capacity=int(math.ceil(duration*30))+32
    # observed=각수색드론이직접본셀, known=직접관측+실제목적지로수신한셀입니다.
    known=np.zeros((ns,nc),np.bool_);observed=np.zeros((ns,nc),np.bool_)
    payload=np.full((capacity,64),-1,np.int32)
    source=np.zeros(capacity,np.int32);dest=np.zeros(capacity,np.int32);born=np.zeros(capacity)
    status=np.zeros(capacity,np.int8);tries=np.zeros(capacity,np.int32)
    hops=np.zeros(capacity,np.int32);relay_hops=np.zeros(capacity,np.int32)
    delays=np.zeros(capacity);last_received=np.zeros((ns,ns));cursor=np.zeros((ns,ns),np.int32)
    queue=np.full((n,64),-1,np.int32);lengths=np.zeros(n,np.int32)
    frame=np.zeros((nt,18));stats=np.zeros((nt,10))
    generated=0;delivered=0;attempts=0;successful=0;retries=0
    expire=0;retrydrop=0;overflow=0;timely=0;relay_successes=0
    global_slot=0;next_birth=0.;aoisum=0.;outage=0.;outagemax=0.;outagelen=0.;outevents=0
    for t in range(nt):
        current=p[t].copy()
        if disable_relays:
            for i in range(ns,n):current[i,:]=0;current[:,i]=0
        routing,stat,a=graph_statistics(current,ns);stats[t]=stat
        if stat[0]<.5:
            outage+=dt;outagelen+=dt
            if outagelen==dt:outevents+=1
            outagemax=max(outagemax,outagelen)
        else:outagelen=0
        # Source sensing at 3s boundaries even when radio is recalculated at .5s.
        now=t*dt
        if abs(now/3-round(now/3))<1e-7:
            for i in range(ns):
                for c in range(nc):
                    if visible[node_ids[t,i],c]:observed[i,c]=True;known[i,c]=True
        for slot in range(slots):
            now=(global_slot+slot)*.005
            while next_birth<=now+1e-10 and next_birth<duration-1e-10:
                pid=generated;s=pid%ns;d=(s+1+(pid//ns)%(ns-1))%ns
                source[pid]=s;dest[pid]=d;born[pid]=next_birth
                if lengths[s]>=64:status[pid]=4;overflow+=1
                else:
                    pos=cursor[s,d];col=0;checked=0
                    while checked<nc and col<64:
                        if known[s,pos]:payload[pid,col]=pos;col+=1
                        pos=(pos+1)%nc;checked+=1
                    cursor[s,d]=pos
                    queue[s,lengths[s]]=pid;lengths[s]+=1
                generated+=1;next_birth=generated/30.
            # 이상적동기TDMA로한슬롯에한드론만송신합니다. 군집내동시송신간섭은0이며외부간섭만있습니다.
            tx=(global_slot+slot)%n
            while lengths[tx]>0 and now-born[queue[tx,0]]>=2.-1e-10:
                pid=queue[tx,0];status[pid]=2;expire+=1;pop_front(queue,lengths,tx)
            if lengths[tx]==0:continue
            pid=queue[tx,0];rx=routing[tx,dest[pid]]
            if rx<0:continue
            attempts+=1
            if attempt_log is not None:attempt_log[t,tx,rx]+=1
            if tries[pid]>0:retries+=1
            tries[pid]+=1
            if np.random.random()<current[tx,rx]:
                successful+=1;hops[pid]+=1
                if success_log is not None:success_log[t,tx,rx]+=1
                if last_ack_log is not None:last_ack_log[t,tx,rx]=now+.005
                if tx>=ns:relay_hops[pid]+=1;relay_successes+=1
                pop_front(queue,lengths,tx);tries[pid]=0
                if rx==dest[pid]:
                    status[pid]=1;delivered+=1;delays[pid]=now+.005-born[pid]
                    if delays[pid]<=.25:timely+=1
                    # 최신수신패킷의생성시각으로AoI를만듭니다. 셀영상의나이나단순전송지연과는다릅니다.
                    last_received[source[pid],dest[pid]]=born[pid]
                    for c in payload[pid]:
                        if c>=0:known[rx,c]=True
                elif lengths[rx]>=64:status[pid]=4;overflow+=1
                else:
                    # EDF by generation ID after merging incoming traffic.
                    pos=lengths[rx]
                    while pos>0 and queue[rx,pos-1]>pid:
                        queue[rx,pos]=queue[rx,pos-1];pos-=1
                    queue[rx,pos]=pid;lengths[rx]+=1
            elif tries[pid]>=4:
                status[pid]=3;retrydrop+=1;pop_front(queue,lengths,tx)
        global_slot+=slots
        if queue_log is not None:queue_log[t,:]=lengths
        union=0;intersection=0
        for c in range(nc):
            u=False;allk=True
            for i in range(ns):
                if observed[i,c]:u=True
                if not known[i,c]:allk=False
            if u:union+=1
            if allk:intersection+=1
        aoi=0.
        for i in range(ns):
            for j in range(ns):
                if i!=j:aoi+=(t+1)*dt-last_received[i,j]
        # AoI는프레임끝의수색방향쌍평균을시간평균합니다. 초기최신timestamp는0으로둡니다.
        aoi/=ns*(ns-1);aoisum+=aoi
        frame[t,:]=np.array([generated,delivered,delivered/max(generated,1),union/nc,intersection/nc,
          float(np.sum(lengths)),stat[0],stat[1],aoi,attempts,successful,expire,retrydrop,overflow,
          float(union),float(intersection),timely/max(generated,1),relay_successes])
    # No new generation after mission end; queued packets declared expired at TTL.
    # Last packets may still have time left, therefore finish an actual 2s drain.
    routing,_,_=graph_statistics(current,ns)
    # 임무종료후생성은중지하고2초(400슬롯)동안잔여큐를정산합니다. 마지막채널확률을고정합니다.
    for slot in range(400):
        now=duration+slot*.005;tx=(global_slot+slot)%n
        while lengths[tx]>0 and now-born[queue[tx,0]]>=2.-1e-10:
            pid=queue[tx,0];status[pid]=2;expire+=1;pop_front(queue,lengths,tx)
        if lengths[tx]==0:continue
        pid=queue[tx,0];rx=routing[tx,dest[pid]]
        if rx<0:continue
        attempts+=1
        if tries[pid]>0:retries+=1
        tries[pid]+=1
        if np.random.random()<current[tx,rx]:
            successful+=1;hops[pid]+=1
            if tx>=ns:relay_hops[pid]+=1;relay_successes+=1
            pop_front(queue,lengths,tx);tries[pid]=0
            if rx==dest[pid]:
                status[pid]=1;delivered+=1;delays[pid]=now+.005-born[pid]
                if delays[pid]<=.25:timely+=1
                for c in payload[pid]:
                    if c>=0:known[rx,c]=True
            elif lengths[rx]>=64:status[pid]=4;overflow+=1
            else:
                pos=lengths[rx]
                while pos>0 and queue[rx,pos-1]>pid:queue[rx,pos]=queue[rx,pos-1];pos-=1
                queue[rx,pos]=pid;lengths[rx]+=1
        elif tries[pid]>=4:status[pid]=3;retrydrop+=1;pop_front(queue,lengths,tx)
    for node in range(n):
        for q in range(lengths[node]):status[queue[node,q]]=2;expire+=1
    # 반환 values순서:생성,전달,시도,성공홉,재시도,TTL유실,재시도한도유실,큐유실,적시전달,중계성공홉,AoI,누적단절,최장단절,단절사건수.
    # frame의18열·stats의10열은full_run에서한국어지표로연결됩니다.
    values=np.array([generated,delivered,attempts,successful,retries,expire,retrydrop,overflow,
      timely,relay_successes,aoisum/max(nt,1),outage,outagemax,outevents])
    return values,frame,stats,status[:generated],delays[:generated],hops[:generated],relay_hops[:generated],observed,known,source[:generated],dest[:generated]
