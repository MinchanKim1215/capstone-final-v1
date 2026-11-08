"""18B local packet-state summary in unused128Bslots, <=10nodes only."""
import numpy as np
VERSION=16
SLOTS=np.r_[np.arange(86,98),np.arange(108,114)]
RAW_DIM=27
DECODED_DIM=28
NAMES=['oldest_queue_age_s','mean_queue_age_s','overdue_count','deadline_soon_count','retry_backlog_count','no_route_opportunities_1s','route_switches_1s','retry_attempts_1s']+[f'destination_queue_{i}' for i in range(6)]+[f'last_used_next_hop_{i}' for i in range(6)]+[f'recent_route_known_{i}' for i in range(6)]+['acks_1s','report_age_s']

def encode(state):
    s=np.asarray(state,float)
    if s.shape!=(RAW_DIM,):raise ValueError('Expected27local packet measurements')
    b=np.zeros(18,np.uint8);b[:2]=np.clip(np.rint(s[:2]*100),0,255).astype(np.uint8);b[2:14]=np.clip(np.rint(s[2:14]),0,255).astype(np.uint8)
    hop=np.where((s[14:20]>=0)&(s[20:26]<=1.),s[14:20],15).astype(np.uint8)
    if not np.isin(hop,np.r_[np.arange(10),15]).all():raise ValueError('Invalid locally observed next-hop')
    b[14:17]=hop[::2]|(hop[1::2]<<4);b[17]=int(np.clip(round(s[26]),0,255));return b

def decode(payload,report_age):
    p=np.asarray(payload,np.uint8)
    if p.shape!=(18,) or not 0<=report_age<=2.+1e-9:raise ValueError('Unsupported packet-state report or stale age')
    hop=np.empty(6,np.uint8);hop[::2]=p[14:17]&15;hop[1::2]=p[14:17]>>4;known=hop<10
    return np.r_[p[:2].astype(float)/100,p[2:14].astype(float),np.where(known,hop,-1),known.astype(float),float(p[17]),report_age].astype(np.float32)

def extend(packet,payload):
    p=np.asarray(packet,np.uint8).copy()
    if p.shape!=(128,) or p[2]>10:raise ValueError('Packet-state extension onlysupports n<=10')
    if np.any(p[SLOTS]) or p[127]!=0:raise ValueError('Reserved slots are not unused; refusing overwrite')
    p[SLOTS]=payload;p[127]=VERSION;return p

def original_packet(packet):
    p=np.asarray(packet,np.uint8).copy()
    if p.shape!=(128,) or p[127]!=VERSION:raise ValueError('Not a packet-state extension')
    p[SLOTS]=0;p[127]=0;return p

def extend_trace(base,observed):
    base=np.asarray(base,np.uint8);result=np.array(base,copy=True)
    empty=np.zeros((base.shape[1],RAW_DIM));empty[:,14:20]=-1;empty[:,20:26]=np.inf
    for k,packets in enumerate(base):
        state=empty if k==0 else observed[k*2-1]
        for sender,packet in enumerate(packets):result[k,sender]=extend(packet,encode(state[sender]))
    return result

class Received:
    def __init__(self,n,ns):self.n=n;self.ns=ns;self.latest=[None]*n
    def observe(self,now,packets,own=None):
        import struct
        for packet in list(packets)+([] if own is None else [own]):
            p=np.asarray(packet,np.uint8)
            if p.shape!=(128,) or p[127]!=VERSION or int(p[2])!=self.n or int(p[3])!=self.ns:raise ValueError('Packet-state header mismatch')
            stamp=struct.unpack_from('<I',p.tobytes(),4)[0]/1000;sender=int(p[1])
            if stamp>now+1e-9:raise ValueError('Future packet-state report')
            old=self.latest[sender]
            if old is None or stamp>=old[0]:self.latest[sender]=(stamp,p[SLOTS].copy())
    def snapshot(self,now,own_state):
        out=np.full((self.n,DECODED_DIM),np.nan,np.float32);known=np.zeros(self.n,bool)
        for j,item in enumerate(self.latest):
            if item is not None and 0<=now-item[0]<=2.:out[j]=decode(item[1],now-item[0]);known[j]=True
        if own_state is not None:out[self.ns]=decode(encode(own_state),0.);known[self.ns]=True
        return out,known
