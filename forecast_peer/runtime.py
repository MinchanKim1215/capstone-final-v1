"""Actual received-only per-peer research API, frozen V13 base unchanged."""
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,sha
from .features import steps
from .inference import predict
from forecast_recall.runtime import Stream as ParentStream
from forecast_rebuild.calibration import transform

def bundle():
    from .evaluate import verify_freeze
    f=verify_freeze();b=read(ARTIFACTS/'research_candidate.json')
    if b['freeze_sha256']!=sha(ARTIFACTS/'freeze.json') or b['runtime_sha256']!=sha(ROOT/'runtime.py'):raise ValueError('Peer export changed')
    if b['candidate']!=f['decision']['candidate'] or sha(b['policy'])!=b['policy_sha256']:raise ValueError('Peer choice changed')
    for path,value in b['dependencies'].items():
        if sha(path)!=value:raise ValueError('Peer dependency changed')
    return b

class Stream:
    def __init__(self,city,n,ns,nominal=None,device='cpu'):
        b=bundle();self.candidate=b['candidate'];self.policy=read(b['policy']);self.n=n;self.ns=ns;self.device=device
        self.parent=ParentStream(city,n,ns,nominal,device);self.peer_history=[];self.baseline_raw=None;self.last_inputs=None;self.active=False
        original=self.parent.raw
        def capture(candidate,inputs):
            p=original(candidate,inputs)
            if candidate==self.parent.candidate:self.baseline_raw=np.array(p,copy=True)
            return p
        self.parent.raw=capture

    @property
    def local(self):return self.parent.local

    def raw(self,candidate,inputs):
        if candidate['family']=='baseline':return self.baseline_raw
        if candidate['family']=='blend':return sum(self.raw(c,inputs)*.5 for c in candidate['members'])
        return predict(candidate['model'],inputs,device=self.device)

    def observe(self,time_end_s,values,masks,intents,energy=None,received_packets=(),own_packet=None):
        old=self.parent.observe(time_end_s,values,masks,intents,energy,received_packets,own_packet);f=self.local;end=f.frames
        if end%2==0:self.peer_history.append(steps(f._values[end-1:end,0],f._masks[end-1:end,0],f._intents[end-1:end,0],f._energy[end-1:end,0],self.n,self.ns)[0])
        if old['status']!='ready':return dict(status=old['status'],trusted_alarm_enabled=False)
        inputs=dict(self.parent.last_inputs,peer_history=np.asarray(self.peer_history),n=np.array([self.n]),ns=np.array([self.ns]));self.last_inputs=inputs
        p=transform(self.raw(self.candidate,inputs),self.policy['parameters']);x=f.values[-1,0];m=f.masks[-1,0];support=m[:,41]&m[:,46]&(x[:,46]>.5);support[self.ns]=False;eligible=bool(support.any())
        score=float(p[0,5]);threshold=self.policy['threshold'];active=bool(eligible and np.isfinite(score) and score>=(threshold*.5 if self.active else threshold))
        if time_end_s<21:active=False
        first=active and not self.active;self.active=active
        return dict(status='ready',risk_5s=float(p[0,2]) if eligible else None,risk_3to7s=score if eligible else None,alarm_data_eligible=eligible,qualified_first_alert=bool(first),
            trusted_alarm=False,trusted_alarm_enabled=False,hardware_validated=False,production_ready=False)

def load_service(city,n_drones,n_search,nominal=None,device='cpu'):return Stream(city,n_drones,n_search,nominal,device)
