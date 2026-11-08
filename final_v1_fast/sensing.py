"""Delayed successfully received RF reports and physical command delivery."""
from dataclasses import dataclass
import numpy as np
from adaptive_joint_expanded.features import THRESHOLD
from .protocol import CONFIG


@dataclass(frozen=True)
class Estimate:
    power:np.ndarray
    noise:np.ndarray
    measured:np.ndarray
    local_rf:bool
    drop_db:float
    latest_stamp_s:float


class Sensors:
    def __init__(self,n,observer,seed,period,hybrid=False):
        self.n=n;self.observer=observer;self.period=period
        self.hybrid=hybrid;self.fast_nodes=(observer,0)
        self.fast_mask=np.zeros(n,bool);self.fast_mask[[observer,0]]=True
        self.incoming=np.full((n,n),np.nan);self.sinr=np.full((n,n),np.nan)
        self.stamp=np.full((n,n),-1e9)
        self.remote_power=np.full((n,n),np.nan);self.remote_sinr=np.full((n,n),np.nan)
        self.remote_stamp=np.full((n,n),-1e9)
        self.pending=[];self.history=[]
        self.random=np.random.default_rng(seed).random((241,n,n))
        self.beacons=0;self.delivered=0

    def deliver(self,now):
        keep=[]
        for due,receiver,power,sinr,stamps,direct,direct_sinr,direct_stamp in self.pending:
            if due>now+1e-8:
                keep.append((due,receiver,power,sinr,stamps,direct,direct_sinr,direct_stamp));continue
            # Direct inbound coordinator reception, plus sender's earlier reports.
            self.remote_power[:,receiver]=power
            self.remote_sinr[:,receiver]=sinr
            self.remote_stamp[:,receiver]=stamps
            self.remote_power[receiver,self.observer]=direct
            self.remote_sinr[receiver,self.observer]=direct_sinr
            self.remote_stamp[receiver,self.observer]=direct_stamp
        self.pending=keep
        local=self.remote_sinr[:,self.observer].copy()
        stamps=self.remote_stamp[:,self.observer].copy()
        self.history.append((now,local,stamps))
        self.history=self.history[-8:]

    def transmit(self,frame,now,p,received,sinr):
        if frame % int(round(self.period*10)):
            return 0
        old_power=self.incoming.copy();old_sinr=self.sinr.copy();old_stamps=self.stamp.copy()
        success=self.random[frame]<p
        np.fill_diagonal(success,False)
        senders=np.ones(self.n,bool)
        if self.hybrid and frame%10:
            senders=self.fast_mask.copy()
        success&=senders[:,None]
        self.beacons+=int(senders.sum())
        self.delivered+=int(success.sum())
        for sender in range(self.n):
            if sender!=self.observer and success[sender,self.observer]:
                self.pending.append((now+CONFIG['sensor_processing_s'],sender,
                    old_power[:,sender].copy(),old_sinr[:,sender].copy(),old_stamps[:,sender].copy(),
                    received[sender,self.observer],sinr[sender,self.observer],now))
        self.incoming[success]=received[success]
        self.sinr[success]=sinr[success]
        self.stamp[success]=now
        return int(senders.sum())

    def estimate(self,now):
        fresh=(self.remote_stamp<=now+1e-8) & (now-self.remote_stamp<=CONFIG['fresh_report_s']) & np.isfinite(self.remote_power)
        power=np.where(fresh,self.remote_power,np.nan)
        noise=np.full(self.n,-107.)
        for j in range(self.n):
            good=fresh[:,j]&np.isfinite(self.remote_sinr[:,j])
            if good.any():
                noise[j]=np.clip(np.median(self.remote_power[good,j]-self.remote_sinr[good,j]),-107.,-50.)
        local=fresh[:,self.observer]
        local[self.observer]=False
        drop=0.
        for time,old,old_stamps in self.history:
            if .19<=now-time<=.21:
                valid=local & (self.remote_stamp[:,self.observer]>old_stamps+1e-8) & np.isfinite(old)
                if valid.any():
                    drop=max(0.,float(np.max(old[valid]-self.remote_sinr[valid,self.observer])))
        return Estimate(power,noise,fresh.any(axis=0),bool(local.any()),drop,
                        float(self.remote_stamp[:,self.observer][local].max()) if local.any() else -1e9)


class Commands:
    def __init__(self,n,observer,initial,seed,latency):
        self.n=n;self.observer=observer
        self.applied=initial.copy();self.requested=initial.copy();self.latency=latency
        self.last_stamp=np.full(n-observer,-1e9);self.pending=[]
        self.fast_stamp=np.full(n,-1e9)
        self.fast_mask=np.zeros(n,bool);self.fast_mask[[observer,0]]=True
        self.requested_fast=self.fast_mask.copy()
        self.random=np.random.default_rng(seed).random((241,3,n,n))
        self.sent=0;self.received=0
        self.delivery_delays=[]

    def deliver(self,now):
        keep=[]
        for due,stamp,goals,reach,fast in self.pending:
            if due>now+1e-8:
                keep.append((due,stamp,goals,reach,fast));continue
            update=reach & (stamp>=self.fast_stamp)
            self.fast_mask[update]=fast[update]
            self.fast_stamp[update]=stamp
            for k in range(len(goals)):
                if reach[self.observer+k] and stamp>=self.last_stamp[k]:
                    self.applied[k]=goals[k]
                    self.last_stamp[k]=stamp
                    self.received+=1
                    self.delivery_delays.append(now-stamp)
        self.pending=keep

    def transmit(self,frame,now,p):
        reached=np.zeros(self.n,bool);reached[self.observer]=True
        broadcast=np.zeros(self.n,bool)
        for hop in range(3):
            edge=self.random[frame,hop]<p
            np.fill_diagonal(edge,False)
            eligible=np.flatnonzero(reached & ~broadcast)
            if not len(eligible):
                break
            sender=self.observer if hop==0 else int(eligible[(frame+hop)%len(eligible)])
            broadcast[sender]=True
            reached|=edge[sender]
        self.sent+=len(self.requested)
        self.pending.append((now+self.latency,now,self.requested.copy(),reached.copy(),self.requested_fast.copy()))
        return 3
