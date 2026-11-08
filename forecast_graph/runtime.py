"""First-relay live research API: local telemetry and actually received packets."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,sha,V10
from .codec import unpack,at
from .features import H,LocalHistory,build
from .inference import predict
from forecast_rebuild.calibration import transform


class ReceivedCache:
    def __init__(self,n,ns):
        self.n=n;self.ns=ns;self.messages=[None]*n;self.snapshots={}

    def observe(self,end,packets,own_packet):
        now=end*.5
        for blob in list(packets)+([] if own_packet is None else [own_packet]):
            message=unpack(blob)
            if message['n']!=self.n or message['ns']!=self.ns:raise ValueError('Received packet fleet mismatch')
            if message['stamp']>now+1e-9 or message['issued']>message['stamp']+1e-9:
                raise ValueError('Future information in received packet')
            j=message['sender'];old=self.messages[j]
            if old is None or message['stamp']>=old['stamp']:self.messages[j]=message
        n=self.n
        state=dict(known=np.zeros(n,bool),age=np.full(n,now),position=np.zeros((n,2)),future=np.zeros((n,9,2)),
            velocity=np.zeros((n,2)),queue=np.zeros(n),attempts=np.zeros(n),successes=np.zeros(n),commit=np.zeros(n),
            issued_age=np.zeros(n),desired=np.zeros((n,2)),sinr=np.full((n,n),np.nan),report_age=np.full((n,n),np.inf))
        for j,m in enumerate(self.messages):
            if m is None:continue
            age=now-m['stamp'];state['known'][j]=age<=10;state['age'][j]=age
            future=at(m,now+H);state['position'][j]=future[0];state['future'][j]=future
            for key in ('velocity','queue','attempts','successes','desired'):state[key][j]=m[key]
            state['commit'][j]=max(0,m['commit_until']-now);state['issued_age'][j]=max(0,now-m['issued'])
            state['sinr'][:,j]=m['sinr'];state['report_age'][:,j]=m['report_age']+age
        self.snapshots[end]=state
        for key in list(self.snapshots):
            if key<end-7:del self.snapshots[key]

    def snapshot(self,end,observer):
        if observer!=self.ns:raise ValueError('This API was confirmed for the first relay')
        if end not in self.snapshots:raise ValueError('No received history at requested time')
        return self.snapshots[end]


def bundle():
    from .evaluate import verify_freeze
    frozen=verify_freeze();value=read(ARTIFACTS/'research_candidates.json')
    if value['freeze_sha256']!=sha(ARTIFACTS/'freeze.json') or value['runtime_sha256']!=sha(ROOT/'runtime.py'):
        raise ValueError('Runtime bundle differs from the frozen experiment')
    for path,expected in value['runtime_dependencies'].items():
        if sha(path)!=expected:raise ValueError('Runtime dependency changed')
    for task,item in value['tasks'].items():
        if item['candidate']!=frozen['decisions'][task]['candidate'] or sha(item['policy'])!=item['policy_sha256']:
            raise ValueError('Exported candidate or policy changed')
    return value


class Stream:
    def __init__(self,city,n,ns,task,nominal,operating_point='advisory',device='cpu'):
        import torch
        from forecast_rebuild.runtime_stable import StableLocalForecast
        value=bundle();entry=value['tasks'][task]
        if not 1<=ns<n<=10:raise ValueError('Unsupported modeled fleet')
        self.n=n;self.ns=ns;self.task=task;self.city=city;self.device=device
        self.candidate=entry['candidate'];self.policy=read(entry['policy']);self.received=ReceivedCache(n,ns)
        old=read(V10/'research_candidates.json')['tasks']['link']['model']
        #The frozen predecessor stream supplies identical20s raw history,
        #geometry, public plan phase and correct public fleet-budget columns.
        self.local=StableLocalForecast(old,city,ns,n,ns,policy=None,device=device,nominal=nominal)
        self.active=np.zeros(n if task=='link' else 1,bool)
        if operating_point=='advisory':self.threshold=float(self.policy['threshold'])
        else:
            point=self.policy.get(operating_point)
            if point is None:raise ValueError('No calibrated operating point: '+operating_point)
            self.threshold=float(point['threshold'])
        torch.set_num_threads(2)

    def raw(self,candidate,arrays,old_output):
        if candidate['family']=='blend':return .5*self.raw(candidate['members'][0],arrays,old_output)+.5*self.raw(candidate['members'][1],arrays,old_output)
        if candidate['family']=='previous':
            if self.task=='link':
                out=np.full((1,16,6),np.nan,np.float32)
                for row in old_output['links']:
                    if row['risk_5s'] is not None:out[0,row['peer'],2]=row['risk_5s'];out[0,row['peer'],5]=row['risk_3to7s']
                return out
            from forecast_service.features import build as old_features
            from forecast_service.inference import tree_predict
            model=read(V10/'research_candidates.json')['tasks']['service']['model']
            return tree_predict(model,old_features(self.local,np.array([[0,0]]),self.ns,self.city))
        return predict(candidate['model'],arrays,ep=self.local,record=dict(n_search=self.ns),city=self.city,device=self.device)[self.task]

    def observe(self,time_end_s,values,masks,intents,energy=None,received_packets=(),own_packet=None):
        result=self.local.observe(time_end_s,values,masks,intents,energy)
        end=self.local.frames;self.received.observe(end,received_packets,own_packet)
        if end<40 or end%2:return dict(status='warming_up' if end<40 else 'recording',links=[],trusted_alarm_enabled=False)
        local=LocalHistory(self.local.values,self.local.masks,self.local.intents,self.n,self.ns)
        arrays=build(local,self.received,[[end,0]],self.city)
        arrays.update(n=np.array([self.n]),ns=np.array([self.ns]),observer=np.array([self.ns]),query=np.array([[0,0]]))
        raw=self.raw(self.candidate,arrays,result)
        calibrated=transform(raw,self.policy['parameters'])
        x=self.local.values[-1,0];m=self.local.masks[-1,0];support=m[:,41]&m[:,46]&(x[:,46]>.5);support[self.ns]=False
        if self.task=='link':
            e=self.local.energy[-1,0]
            eligible=support&(e[:,3]>.5)&(e[:,4]>.5)&np.isfinite(e[:,0])&np.isfinite(e[:,1])&np.isfinite(e[:,5])
            eligible &= (e[:,2]>=1)&(e[:,2]<=2)&(e[:,5]>=.5)&(e[:,5]<=1)
            p=np.asarray(calibrated[0,:self.n,5],np.float64)
        else:
            eligible=np.array([support.any()]);p=np.asarray(calibrated[:,5],np.float64)
        new=eligible&np.isfinite(p)&(p>=np.where(self.active,self.threshold*.5,self.threshold))
        if time_end_s<21:new[:]=False
        first=new&~self.active;self.active=new
        if self.task=='link':
            rows=[dict(peer=j,risk_5s=float(calibrated[0,j,2]) if support[j] else None,
                risk_3to7s=float(calibrated[0,j,5]) if support[j] else None,
                alarm_data_eligible=bool(eligible[j]),qualified_first_alert=bool(first[j]),trusted_alarm=False) for j in range(self.n) if j!=self.ns]
            return dict(status='ready',links=rows,trusted_alarm_enabled=False,production_ready=False,hardware_validated=False)
        return dict(status='ready',risk_5s=float(calibrated[0,2]) if eligible[0] else None,
            risk_3to7s=float(calibrated[0,5]) if eligible[0] else None,qualified_first_alert=bool(first[0]),
            alarm_data_eligible=bool(eligible[0]),trusted_alarm=False,trusted_alarm_enabled=False,
            physics_disconnection_proxy=arrays['global'][0,:9].astype(float).tolist(),production_ready=False,hardware_validated=False)


def load_link(city,n_drones,n_search,nominal=None,operating_point='advisory',device='cpu'):
    return Stream(city,n_drones,n_search,'link',nominal,operating_point,device)


def load_service(city,n_drones,n_search,nominal=None,operating_point='advisory',device='cpu'):
    return Stream(city,n_drones,n_search,'service',nominal,operating_point,device)
