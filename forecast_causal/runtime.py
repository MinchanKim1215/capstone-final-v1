"""Actual observable packet-state+physics research API; no simulator truth."""
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,sha
from .wire import Received,VERSION,original_packet
from .features import one as summary_one,tree_features
from .physics import Geometry
from .inference import predict
from forecast_recall.runtime import Stream as ParentStream
from forecast_graph.features import LocalHistory
from forecast_service.features import build as old_tree
from forecast_rebuild.calibration import transform

def bundle():
    from .evaluate import verify_freeze
    f=verify_freeze();b=read(ARTIFACTS/'research_candidate.json')
    if b['freeze_sha256']!=sha(ARTIFACTS/'freeze.json') or b['runtime_sha256']!=sha(ROOT/'runtime.py'):raise ValueError('Causalruntime export changed')
    if b['candidate']!=f['decision']['candidate'] or sha(b['policy'])!=b['policy_sha256']:raise ValueError('Causalruntime choice changed')
    for p,v in b['dependencies'].items():
        if sha(p)!=v:raise ValueError('Causalruntime dependency changed')
    return b

def needs_network(candidate):
    if candidate['family']=='baseline':return False
    if candidate['family']=='blend':return any(w>0 and needs_network(c) for c,w in zip(candidate['members'],candidate['weights']))
    return read(candidate['model'])['variant'] in ('tree_network','tree_both','gru_network')

class Stream:
    def __init__(self,city,n,ns,nominal=None,device='cpu'):
        b=bundle();self.candidate=b['candidate'];self.policy=read(b['policy']);self.n=n;self.ns=ns;self.city=city;self.device=device
        self.parent=ParentStream(city,n,ns,nominal,device);self.received=Received(n,ns);self.geometry=Geometry(city);self.summaries=[];self.active=False;self.last_inputs=None;self.baseline_raw=None;self.requires_network=needs_network(self.candidate)
        original=self.parent.raw
        def capture(candidate,inputs):
            value=original(candidate,inputs)
            if candidate==self.parent.candidate:self.baseline_raw=np.array(value,copy=True)
            return value
        self.parent.raw=capture

    @property
    def local(self):return self.parent.local

    def raw(self,candidate,inputs):
        if candidate['family']=='baseline':return self.baseline_raw
        if candidate['family']=='blend':return sum(self.raw(c,inputs)*w for c,w in zip(candidate['members'],candidate['weights']))
        return predict(candidate['model'],inputs,device=self.device)

    def observe(self,time_end_s,values,masks,intents,energy=None,received_packets=(),own_packet=None,local_network=None):
        packets=list(received_packets)
        self.received.observe(time_end_s,[p for p in packets if np.asarray(p)[127]==VERSION],own_packet if own_packet is not None and np.asarray(own_packet)[127]==VERSION else None)
        restore=lambda p:original_packet(p) if np.asarray(p)[127]==VERSION else p
        old=self.parent.observe(time_end_s,values,masks,intents,energy,[restore(p) for p in packets],None if own_packet is None else restore(own_packet));end=self.local.frames
        network,known=self.received.snapshot(time_end_s,local_network);self.summaries.append(summary_one(network,known,self.n,self.ns))
        if old['status']!='ready':return dict(status=old['status'],trusted_alarm_enabled=False)
        if self.requires_network and local_network is None:
            self.active=False;return dict(status='missing_local_packet_state',risk_5s=None,risk_3to7s=None,alarm_data_eligible=False,qualified_first_alert=False,trusted_alarm_enabled=False,hardware_validated=False)
        f=self.local;local=LocalHistory(f.values,f.masks,f.intents,self.n,self.ns);summary=np.asarray(self.summaries,np.float32)
        inputs=dict(self.parent.last_inputs,base=np.c_[old_tree(f,[[0,0]],self.ns,self.city),self.baseline_raw[:,[2,5]]].astype(np.float32),
            network=tree_features(summary,[end]),physics=self.geometry.build(local,self.parent.parent.mission,[end]),network_history=summary.astype(np.float16),
            n=np.array([self.n]),ns=np.array([self.ns]),baseline=self.baseline_raw)
        self.last_inputs=inputs;raw=self.raw(self.candidate,inputs);p=transform(raw,self.policy['parameters']);x=f.values[-1,0];m=f.masks[-1,0]
        support=m[:,41]&m[:,46]&(x[:,46]>.5);support[self.ns]=False;eligible=bool(support.any());score=float(p[0,5]);threshold=self.policy['threshold']
        active=bool(eligible and np.isfinite(score) and score>=(threshold*.5 if self.active else threshold))
        if time_end_s<21:active=False
        first=active and not self.active;self.active=active
        return dict(status='ready',risk_5s=float(p[0,2]) if eligible else None,risk_3to7s=score if eligible else None,alarm_data_eligible=eligible,qualified_first_alert=bool(first),
            network_observation_available=local_network is not None,trusted_alarm=False,trusted_alarm_enabled=False,hardware_validated=False,production_ready=False)

def load_service(city,n_drones,n_search,nominal=None,device='cpu'):return Stream(city,n_drones,n_search,nominal,device)
