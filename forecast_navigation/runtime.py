"""One received packet supports both plan views; previous code remains frozen."""
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,sha,PARENT
from .wire import original_packet
from forecast_graph.runtime import Stream as ParentStream,ReceivedCache
from forecast_graph.features import LocalHistory,build
from forecast_graph.inference import predict
from forecast_rebuild.calibration import transform


def bundle():
    from .evaluate import verify_freeze
    frozen=verify_freeze();b=read(ARTIFACTS/'research_candidates.json')
    if b['freeze_sha256']!=sha(ARTIFACTS/'freeze.json') or b['runtime_sha256']!=sha(ROOT/'runtime.py'):raise ValueError('V12 bundle changed')
    for path,expected in b['dependencies'].items():
        if sha(path)!=expected:raise ValueError('V12 runtime dependency changed')
    for task,entry in b['tasks'].items():
        if entry['candidate']!=frozen['decisions'][task]['candidate'] or sha(entry['policy'])!=entry['policy_sha256']:raise ValueError('Unfrozen export')
    return b


class Stream:
    def __init__(self,city,n,ns,task,nominal,operating_point='advisory',device='cpu'):
        entry=bundle()['tasks'][task];self.candidate=entry['candidate'];self.policy=read(entry['policy']);self.task=task;self.n=n;self.ns=ns;self.city=city;self.device=device
        self.engine=ParentStream(city,n,ns,task,nominal,device=device);self.mission=ReceivedCache(n,ns)
        self.active=np.zeros(n if task=='link' else 1,bool);self.last_inputs=None
        if operating_point=='advisory':self.threshold=float(self.policy['threshold'])
        elif self.policy.get(operating_point):self.threshold=float(self.policy[operating_point]['threshold'])
        else:raise ValueError('No calibrated operating point: '+operating_point)

    @property
    def local(self):return self.engine.local

    def raw(self,candidate,views,old_output):
        if candidate['family']=='blend':return sum(self.raw(m,views,old_output)*.5 for m in candidate['members'])
        if candidate['family']=='parent':return self.engine.raw(candidate['parent_candidate'],views['original'],old_output)
        return predict(candidate['model'],views[candidate['profile']],device=self.device)[self.task]

    def observe(self,time_end_s,values,masks,intents,energy=None,received_packets=(),own_packet=None):
        old_output=self.local.observe(time_end_s,values,masks,intents,energy);end=self.local.frames
        original=[original_packet(p) for p in received_packets];own_original=original_packet(own_packet) if own_packet is not None else None
        self.engine.received.observe(end,original,own_original);self.mission.observe(end,received_packets,own_packet)
        if end<40 or end%2:return dict(status='warming_up' if end<40 else 'recording',links=[],trusted_alarm_enabled=False)
        local=LocalHistory(self.local.values,self.local.masks,self.local.intents,self.n,self.ns)
        views={}
        for profile,cache in [('original',self.engine.received),('mission',self.mission)]:
            a=build(local,cache,[[end,0]],self.city);a.update(n=np.array([self.n]),ns=np.array([self.ns]),observer=np.array([self.ns]),query=np.array([[0,0]]));views[profile]=a
        self.last_inputs=views;raw=self.raw(self.candidate,views,old_output);p=transform(raw,self.policy['parameters'])
        x=self.local.values[-1,0];m=self.local.masks[-1,0];support=m[:,41]&m[:,46]&(x[:,46]>.5);support[self.ns]=False
        if self.task=='link':
            e=self.local.energy[-1,0];eligible=support&(e[:,3]>.5)&(e[:,4]>.5)&np.isfinite(e[:,0])&np.isfinite(e[:,1])&np.isfinite(e[:,5])
            eligible &= (e[:,2]>=1)&(e[:,2]<=2)&(e[:,5]>=.5)&(e[:,5]<=1);score=np.asarray(p[0,:self.n,5],np.float64)
        else:eligible=np.array([support.any()]);score=np.asarray(p[:,5],np.float64)
        new=eligible&np.isfinite(score)&(score>=np.where(self.active,self.threshold*.5,self.threshold))
        if time_end_s<21:new[:]=False
        first=new&~self.active;self.active=new
        if self.task=='link':
            return dict(status='ready',links=[dict(peer=j,risk_5s=float(p[0,j,2]) if support[j] else None,risk_3to7s=float(p[0,j,5]) if support[j] else None,
                alarm_data_eligible=bool(eligible[j]),qualified_first_alert=bool(first[j]),trusted_alarm=False) for j in range(self.n) if j!=self.ns],trusted_alarm_enabled=False,hardware_validated=False)
        return dict(status='ready',risk_5s=float(p[0,2]) if eligible[0] else None,risk_3to7s=float(p[0,5]) if eligible[0] else None,
            qualified_first_alert=bool(first[0]),alarm_data_eligible=bool(eligible[0]),trusted_alarm=False,trusted_alarm_enabled=False,hardware_validated=False)


def load_link(city,n_drones,n_search,nominal=None,operating_point='advisory',device='cpu'):return Stream(city,n_drones,n_search,'link',nominal,operating_point,device)
def load_service(city,n_drones,n_search,nominal=None,operating_point='advisory',device='cpu'):return Stream(city,n_drones,n_search,'service',nominal,operating_point,device)
