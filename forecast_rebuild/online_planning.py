"""Stateful one-frame equivalent of the offline public-plan features."""
import numpy as np
from scipy.special import ndtr
from .features import HORIZONS
from .planning import infer_phase,nominal_positions,DIMENSION


class PlanContext:
    def __init__(self,city,nominal,observer,n_search):
        self.city=city;self.nominal=np.asarray(nominal);self.observer=observer;self.n_search=n_search
        self.previous=None

    def compute(self,x,m,c,clock_s):
        n=len(x);out=np.full((n,DIMENSION),np.nan,np.float32)
        inferred=infer_phase(self.nominal,x,m,self.n_search,self.previous,clock_s)
        if inferred is not None:self.previous=inferred[0]
        quality=([inferred[1]/3,inferred[2]/20,inferred[3]/max(1,self.n_search),1.] if inferred is not None else [0,0,0,0])
        good=m[:,[0,1,2,12,13,30,34,35,36,37]].all(-1)&(x[:,30]<=2)
        good[self.observer]=False;j=np.flatnonzero(good)
        if not len(j):return out
        a=np.c_[x[j,12:14],np.full(len(j),50.)]
        b=a+np.c_[x[j,34:36],np.zeros(len(j))]
        va=np.c_[x[j,:2],np.zeros(len(j))];vb=va+np.c_[x[j,36:38],np.zeros(len(j))]
        rem=np.clip(c[j,5],0,3);prem=np.where(c[j,8]>.5,np.clip(3-c[j,4],0,3),0.)
        scenario=self.city._forecast_scenario
        current=self.city.radio.propagation(a,b,scenario)[0]
        ca=np.clip(a[:,None]+HORIZONS[None,:,None]*va[:,None],[-195,-195,50],[195,195,50])
        cb=np.clip(b[:,None]+HORIZONS[None,:,None]*vb[:,None],[-195,-195,50],[195,195,50])
        loss,los,_=self.city.radio.propagation(ca.reshape(-1,3),cb.reshape(-1,3),scenario)
        out[j,56:69]=(loss.reshape(-1,13)-current[:,None])/30;out[j,69:82]=los.reshape(-1,13)
        out[j,52:56]=quality
        if inferred is not None:
            phase=np.full((len(j),1),inferred[0]);own=np.full((len(j),1),self.observer);peer=j[:,None]
            early=nominal_positions(self.nominal,phase+np.minimum(HORIZONS[None],rem[:,None])/3,own)
            future=nominal_positions(self.nominal,phase+HORIZONS[None]/3,own)
            pa=a[:,None]+np.minimum(HORIZONS[None],rem[:,None])[...,None]*va[:,None]
            pa[...,:2]+=future-early
            early=nominal_positions(self.nominal,phase+np.minimum(HORIZONS[None],prem[:,None])/3,peer)
            future=nominal_positions(self.nominal,phase+HORIZONS[None]/3,peer)
            pb=b[:,None]+np.minimum(HORIZONS[None],prem[:,None])[...,None]*vb[:,None]
            pb[...,:2]+=future-early
            pa[...,:2]=np.clip(pa[...,:2],-195,195);pb[...,:2]=np.clip(pb[...,:2],-195,195)
            loss,los,_=self.city.radio.propagation(pa.reshape(-1,3),pb.reshape(-1,3),scenario)
            delta=loss.reshape(-1,13)-current[:,None];margin=x[j,41]
            sigma=np.sqrt(1.5**2*(1-np.exp(-2*HORIZONS/4))+.5**2)
            out[j,:13]=delta/30;out[j,13:26]=los.reshape(-1,13)
            out[j,26:39]=(margin[:,None]-delta)/30;out[j,39:52]=ndtr((delta-margin[:,None])/sigma)
        return out
