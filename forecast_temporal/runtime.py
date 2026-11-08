"""Actual incremental, received-only20snapshot research service API."""
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,sha
from .inference import predict
from .features import MAX_N
from forecast_recall.runtime import Stream as ParentStream
from forecast_graph.features import frame,LocalHistory
from forecast_rebuild.calibration import transform

def bundle():
    from .evaluate import verify_freeze
    f=verify_freeze();b=read(ARTIFACTS/'research_candidate.json')
    if b['freeze_sha256']!=sha(ARTIFACTS/'freeze.json') or b['runtime_sha256']!=sha(ROOT/'runtime.py'):raise ValueError('Temporal bundle changed')
    if b['candidate']!=f['decision']['candidate'] or sha(b['policy'])!=b['policy_sha256']:raise ValueError('Choice changed')
    for path,value in b['dependencies'].items():
        if sha(path)!=value:raise ValueError('Dependency changed')
    return b

class Stream:
    def __init__(self,city,n,ns,nominal=None,device='cpu'):
        b=bundle();self.candidate=b['candidate'];self.policy=read(b['policy']);self.n=n;self.ns=ns;self.city=city;self.device=device
        self.parent=ParentStream(city,n,ns,nominal,device);self.nodes=[];self.edges=[];self.active=False;self.baseline_raw=None;self.last_inputs=None
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
        if candidate['family']=='blend':return sum(self.raw(c,inputs)*.5 for c in candidate['members'])
        return predict(candidate['model'],inputs,device=self.device)

    def observe(self,time_end_s,values,masks,intents,energy=None,received_packets=(),own_packet=None):
        old=self.parent.observe(time_end_s,values,masks,intents,energy,received_packets,own_packet);end=self.local.frames
        if end%2==0:
            f=self.local;local=LocalHistory(f._values[:end],f._masks[:end],f._intents[:end],self.n,self.ns)
            node,edge,_=frame(local,self.parent.parent.mission,end,0,True)
            self.nodes.append(node[:MAX_N].astype(np.float16));self.edges.append(edge[:MAX_N,:MAX_N].astype(np.float16))
        if old['status']!='ready':return dict(status=old['status'],trusted_alarm_enabled=False)
        current=self.parent.parent.last_inputs['mission']
        inputs=dict(node=np.asarray(self.nodes[-20:]),edge=np.asarray(self.edges[-20:]),index=np.arange(20)[None],physics=current['physics'][:,:,:MAX_N,:MAX_N],
            global_features=current['global'],n=np.array([self.n]),ns=np.array([self.ns]),observer=np.array([self.ns]))
        self.last_inputs=inputs;raw=self.raw(self.candidate,inputs);p=transform(raw,self.policy['parameters'])
        f=self.local;x=f.values[-1,0];m=f.masks[-1,0];support=m[:,41]&m[:,46]&(x[:,46]>.5);support[self.ns]=False;eligible=bool(support.any())
        score=float(p[0,5]);threshold=self.policy['threshold'];active=bool(eligible and np.isfinite(score) and score>=(threshold*.5 if self.active else threshold))
        if time_end_s<21:active=False
        first=active and not self.active;self.active=active
        return dict(status='ready',risk_5s=float(p[0,2]) if eligible else None,risk_3to7s=score if eligible else None,alarm_data_eligible=eligible,
            qualified_first_alert=bool(first),trusted_alarm=False,trusted_alarm_enabled=False,production_ready=False,hardware_validated=False)

def load_service(city,n_drones,n_search,nominal=None,device='cpu'):return Stream(city,n_drones,n_search,nominal,device)
