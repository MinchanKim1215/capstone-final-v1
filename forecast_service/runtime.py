"""Two separate, hash-bound research APIs; no automatic PPO replacement."""
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from . import ARTIFACTS
from .contracts import read,sha
from .evaluate import verify_freeze
from .inference import pin,tree_predict
from .features import build
from forecast_rebuild import ROOT as PARENT_ROOT
from forecast_rebuild.runtime_stable import StableLocalForecast
from forecast_rebuild.calibration import transform


def dependencies():
    value={str(PARENT_ROOT/'runtime_stable.py'):'ca216d3fd4ef3f595a5a0dc8e3d60ac3e9367fd57cefaaff626ebac80b06cd3a'}
    for path,expected in value.items():
        if sha(path)!=expected:raise ValueError('Audited stable numerical adapter changed')
    return value


def bundle(task):
    frozen=verify_freeze();data=read(ARTIFACTS/'research_candidates.json');item=data['tasks'][task]
    if data['freeze_sha256']!=sha(ARTIFACTS/'freeze.json') or data['runtime_sha256']!=sha(__file__) or data['runtime_dependencies']!=dependencies():
        raise ValueError('Exported runtime differs from its audited implementation')
    if item['model']!=frozen['decisions'][task]['model'] or pin(item['model'])!=item['pin'] or sha(item['policy'])!=item['policy_sha256']:
        raise ValueError('Exported candidate or operating policy differs')
    return item


def operating_policy(path,operating_point):
    value=read(path)
    if operating_point not in ('advisory','common_FA_budget1','precision90_FA1_partial'):raise ValueError('Unknown operating point')
    if operating_point!='advisory':
        if not value.get(operating_point):raise ValueError('This operating point did not exist on calibration data')
        value=dict(value,threshold=value[operating_point]['threshold'],enabled=False)
    # A strong float64 scalar preserves the frozen offline comparison at
    # empirical-quantile boundaries even inside the inherited float32 stream.
    value['threshold']=np.float64(value['threshold'])
    return value


def load_link(city,observer,n_drones,n_search,nominal,device='cpu',operating_point='advisory'):
    item=bundle('link');result=StableLocalForecast(item['model'],city,observer,n_drones,n_search,nominal=nominal,
        policy=item['policy'],device=device,confirmation_goal_met=False,hardware_validated=False)
    result.policy=operating_policy(item['policy'],operating_point);return result


class ServiceForecast:
    """New service-impairment onset risk, not a claim of current connectivity."""
    def __init__(self,model,policy,city,observer,n_drones,n_search,nominal=None,device='cpu',operating_point='advisory',max_frames=14400):
        if observer!=n_search or not 1<=n_search<n_drones<=10:raise ValueError('Service confirmation covered the designated first relay only')
        scenario=getattr(city,'_forecast_scenario',None)
        profile=dict(frequency_hz=2.4e9,altitude_m=50.,tx_power_dbm=10.,bandwidth_hz=1e6,bit_rate_bps=1e6,shadow_sigma_db=1.5,shadow_correlation_s=4.)
        if scenario is None or any(not np.isclose(getattr(scenario,k,float('nan')),v) for k,v in profile.items()) or scenario.mac!='tdma' or scenario.radio_model!='bullington':
            raise ValueError('Unvalidated radio/flight profile')
        self.model=model;self.meta=read(model);self.policy=operating_policy(policy,operating_point);self.city=city;self.n=n_drones;self.ns=n_search
        self.values=np.zeros((max_frames,1,n_drones,48),np.float16);self.masks=np.zeros_like(self.values,bool)
        self.intents=np.zeros((max_frames,1,n_drones,9),np.float16);self.energy=np.full((max_frames,1,n_drones,6),np.nan,np.float16);self.energy[...,3:5]=0
        self.frames=0;self.active=False;self.max_frames=max_frames;self.proxy=None
        if self.meta['family']=='service_proxy':
            self.proxy=StableLocalForecast(self.meta['link_model'],city,observer,n_drones,n_search,nominal=nominal,device=device,max_frames=max_frames)

    def observe(self,time_end_s,values,masks,intents,energy=None):
        x=np.asarray(values,np.float32);m=np.asarray(masks,bool);c=np.asarray(intents,np.float32)
        if x.shape!=(self.n,48) or m.shape!=x.shape or c.shape!=(self.n,9):raise ValueError('Local telemetry shape mismatch')
        if not np.isclose(time_end_s,(self.frames+1)*.5) or not np.isfinite(time_end_s):raise ValueError('Supply ordered half-second frames')
        if self.frames>=self.max_frames:raise ValueError('Reset before maximum mission duration')
        if not np.isfinite(x[m]).all() or not np.isfinite(c).all():raise ValueError('Nonfinite observed data')
        if np.any((c[:,5]<0)|(c[:,5]>3)) or np.any(c[:,4]<0) or np.any((c[:,8]!=0)&(c[:,8]!=1)):raise ValueError('Invalid issued-command ages or support')
        observed=np.where(m,x,0).astype(np.float16);encoded=c.astype(np.float16)
        if not np.isfinite(observed).all() or not np.isfinite(encoded).all():raise ValueError('Numeric range overflow')
        self.values[self.frames,0]=observed;self.masks[self.frames,0]=m;self.intents[self.frames,0]=encoded
        if energy is not None:
            e=np.asarray(energy,np.float32).copy()
            if e.shape!=(self.n,6) or np.any((e[:,3:5]!=0)&(e[:,3:5]!=1)):raise ValueError('Invalid optional energy sample')
            if np.any(e[:,[2,5]][np.isfinite(e[:,[2,5]])]<0):raise ValueError('Future-dated measurement')
            if np.any((e[:,3]>.5)&(~np.isfinite(e[:,0])|~np.isfinite(e[:,5]))) or np.any((e[:,4]>.5)&(~np.isfinite(e[:,1])|~np.isfinite(e[:,2]))):
                raise ValueError('Valid sensor flag needs finite value and age')
            e[e[:,3]<.5,0]=np.nan;e[e[:,3]<.5,5]=np.nan;e[e[:,4]<.5,1:3]=np.nan;self.energy[self.frames,0]=e.astype(np.float16)
        proxy=self.proxy.observe(time_end_s,x,m,c,energy) if self.proxy else None
        self.frames+=1
        if self.frames<40 or self.frames%2:return dict(status='warming_up' if self.frames<40 else 'recording',service_risk_5s=None,service_risk_3to7s=None,first_alert=False,trusted_alarm=False)
        started=time.perf_counter();latest=self.values[self.frames-1,0].astype(np.float32);known=self.masks[self.frames-1,0]
        support=known[:,41]&known[:,46]&(latest[:,46]>.5);support[self.ns]=False
        eligible=bool(support.any());p=np.full((1,6),np.nan,np.float32)
        if eligible:
            if self.proxy:
                for head,key in [(2,'risk_5s'),(5,'risk_3to7s')]:
                    values=[row[key] for row in proxy['links'] if row[key] is not None]
                    if values:p[0,head]=max(values)
            else:
                view=SimpleNamespace(values=self.values[:self.frames],masks=self.masks[:self.frames],intents=self.intents[:self.frames],
                    energy=self.energy[:self.frames],ends=np.array([self.frames]))
                features=build(view,np.array([[0,0]]),self.ns,self.city);p=tree_predict(self.model,features)
            p=transform(p,self.policy['parameters'])
        eligible=bool(eligible and np.isfinite(p[0,[2,5]]).all())
        active=bool(eligible and np.isfinite(p[0,5]) and float(p[0,5])>=(self.policy['threshold']*.5 if self.active else self.policy['threshold']))
        first=active and not self.active;self.active=active
        return dict(status='ready' if eligible else 'insufficient_observation',service_risk_5s=float(p[0,2]) if eligible else None,
            service_risk_3to7s=float(p[0,5]) if eligible else None,alarm_data_eligible=eligible,first_alert=bool(first),trusted_alarm=False,
            inference_ms=(time.perf_counter()-started)*1000,scope='New service-onset risk from designated relay local view; no current-service-state or hardware assurance')


def load_service(city,observer,n_drones,n_search,nominal=None,device='cpu',operating_point='advisory'):
    item=bundle('service');return ServiceForecast(item['model'],item['policy'],city,observer,n_drones,n_search,nominal,device,operating_point)
