"""상태 비콘의 슬롯 비용을 반영한 기존 TDMA 패킷 엔진의 분리 버전.

데이터 생성/TTL/재시도/경로 계산은 기존 구현과 같으며 임무 중 매초
앞 8개 5ms 슬롯을 비콘용으로 예약합니다. 임무 종료 후에는 비콘 없이
2초 잔여 큐 정산을 수행합니다.
"""
import math
import numpy as np
from numba import njit
from swarm_sim.fast_network import graph_statistics, pop_front

@njit(cache=True)
def packet_run(p,node_ids,visible,ns,dt,seed,disable_relays=False,
               attempt_log=None,success_log=None,last_ack_log=None,queue_log=None,reserved_beacon_slots=8):
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
            # 매 1초의 앞 N개 슬롯은 상태 비콘 전용입니다. 앱 생성/TTL은 계속 처리합니다.
            if (global_slot+slot)%200 < reserved_beacon_slots:continue
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
