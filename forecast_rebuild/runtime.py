"""Local stream interface. Unknown/stale information never becomes zero risk."""
import time
from pathlib import Path
import numpy as np
from .inference import Predictor
from .features import build,sequence
from .calibration import transform
from .contracts import read


class LocalForecast:
    def __init__(self,model,city,observer,n_drones,n_search,policy=None,device='cpu',nominal=None,max_frames=14400,
                 confirmation_goal_met=False,hardware_validated=False):
        if not 1<=n_search<n_drones<=10 or not n_search<=observer<n_drones:raise ValueError('A relay observer and known fleet roles are required')
        self.predictor=Predictor(model,device);self.city=city;self.observer=observer;self.n=n_drones;self.ns=n_search
        scenario=getattr(city,'_forecast_scenario',None)
        if scenario is None:raise ValueError('Supply the explicit modeled radio/flight profile through city._forecast_scenario')
        profile=dict(frequency_hz=2.4e9,altitude_m=50.,tx_power_dbm=10.,gain_dbi=0.,bandwidth_hz=1e6,
                     noise_figure_db=7.,bit_rate_bps=1e6,link_success_threshold=.9,shadow_sigma_db=1.5,
                     shadow_correlation_s=4.,slot_s=.005)
        if any(not np.isclose(getattr(scenario,key,float('nan')),expected) for key,expected in profile.items()) or scenario.radio_model!='bullington' or scenario.mac!='tdma':
            raise ValueError('This model has not been validated for the supplied radio/flight profile')
        self.nr=n_drones-n_search;self.max_frames=max_frames
        self.policy=read(policy) if policy else None
        if self.policy and self.policy['model_tag']!=self.predictor.meta['tag']:raise ValueError('Calibration belongs to a different predictor')
        self.confirmation_goal_met=confirmation_goal_met;self.hardware_validated=hardware_validated
        self.nominal=nominal;self.reset()

    def reset(self):
        self._values=np.zeros((self.max_frames,1,self.n,48),np.float16)
        self._masks=np.zeros_like(self._values,bool)
        self._intents=np.zeros((self.max_frames,1,self.n,9),np.float16)
        self._energy=np.full((self.max_frames,1,self.n,6),np.nan,np.float16);self._energy[...,3:5]=0
        self.frames=0;self.last_time=None;self.active=np.zeros(self.n,bool);self.qualified=np.zeros(self.n,bool);self.plan=None
        if self.predictor.meta.get('use_plan',False):
            if self.nominal is None:raise ValueError('This candidate requires its public preflight nominal plan')
            from .online_planning import PlanContext
            self.plan=PlanContext(self.city,self.nominal,self.observer,self.ns)

    def features(self,rows,use_energy=True,planning=False):
        x=build(self.values,self.masks,self.intents,self.ends,self.geometry,rows,self.energy,use_energy)
        # A live instance has one local observer. Fleet budgets are public
        # deployment properties, not the number of observers in its buffer.
        x[:,644]=self.nr/5;x[:,645]=self.n/10
        if planning:
            _,_,j=np.asarray(rows,int).T
            x=np.concatenate([x,self._planning[j]],-1)
        return x

    def sequence(self,rows,use_energy=True):
        return sequence(self.values,self.masks,self.intents,self.ends,rows,self.energy,use_energy)

    def observe(self,time_end_s,values,masks,intents,energy=None):
        x=np.asarray(values,np.float32);m=np.asarray(masks,bool);c=np.asarray(intents,np.float32)
        if x.shape!=(self.n,48) or m.shape!=x.shape or c.shape!=(self.n,9):raise ValueError('Local telemetry shape mismatch')
        if not np.isfinite(time_end_s) or not np.isclose(time_end_s,(self.frames+1)*.5):raise ValueError('Supply each half-second frame in order, including explicit missing frames')
        if self.frames>=self.max_frames:raise ValueError('Reset before the maximum mission duration')
        if not np.isfinite(x[m]).all() or not np.isfinite(c).all():raise ValueError('Nonfinite observed telemetry')
        if np.any((c[:,5]<0)|(c[:,5]>3)) or np.any(c[:,4]<0) or np.any((c[:,8]!=0)&(c[:,8]!=1)):
            raise ValueError('Invalid issued-plan age/support')
        observed=np.where(m,x,0).astype(np.float16)
        if not np.isfinite(observed).all():raise ValueError('Telemetry exceeds supported numeric range')
        encoded=c.astype(np.float16)
        if not np.isfinite(encoded).all():raise ValueError('Issued intent exceeds supported numeric range')
        self._values[self.frames,0]=observed;self._masks[self.frames,0]=m;self._intents[self.frames,0]=encoded
        if energy is not None:
            e=np.asarray(energy,np.float32).copy()
            if e.shape!=(self.n,6):raise ValueError('Energy measurement shape mismatch')
            if np.any((e[:,3:5]!=0)&(e[:,3:5]!=1)):raise ValueError('Energy support flags must be binary')
            if np.any((e[:,3]>.5)&(~np.isfinite(e[:,0]))) or np.any((e[:,4]>.5)&(~np.isfinite(e[:,1]))):
                raise ValueError('A valid energy sample requires a finite measurement')
            if np.any(e[:,[2,5]][np.isfinite(e[:,[2,5]])]<0):raise ValueError('A sensor cannot report a future sample')
            e[e[:,3]<.5,0]=np.nan;e[e[:,3]<.5,5]=np.nan
            e[e[:,4]<.5,1:3]=np.nan
            if np.any((e[:,3]>.5)&~np.isfinite(e[:,5])) or np.any((e[:,4]>.5)&~np.isfinite(e[:,2])):
                raise ValueError('Supported sensor readings require a measurement age')
            self._energy[self.frames,0]=e.astype(np.float16)
        self.frames+=1;self.last_time=time_end_s
        if self.frames<40 or self.frames%2:return dict(status='warming_up' if self.frames<40 else 'recording',links=[])
        started=time.perf_counter()
        self.values=self._values[:self.frames];self.masks=self._masks[:self.frames]
        self.intents=self._intents[:self.frames];self.energy=self._energy[:self.frames]
        self.ends=np.array([self.frames])
        latest=self.values[-1,0].astype(np.float32);known=self.masks[-1,0];intent=self.intents[-1,0].astype(np.float32)
        from cooperative.fast_features import geometry_profile
        self.geometry=geometry_profile(self.city,latest,known,intent)[None,None]
        if self.plan:self._planning=self.plan.compute(latest,known,intent,time_end_s)
        rows=np.array([[0,0,j] for j in range(self.n) if j!=self.observer])
        raw=self.predictor.predict(self,rows)[0]
        if self.policy:raw=transform(raw,self.policy['parameters'])
        trusted=bool(self.policy and self.policy['enabled'] and self.confirmation_goal_met and self.hardware_validated)
        threshold=self.policy['threshold'] if self.policy else .5
        output=[]
        for index,(_,_,j) in enumerate(rows):
            supported=bool(known[j,41] and known[j,46] and latest[j,46]>.5)
            available=bool(known[j,30] and known[j,34] and known[j,35] and latest[j,30]<=10)
            new=bool(supported and np.isfinite(raw[index,5]) and raw[index,5]>=(threshold*.5 if self.active[j] else threshold))
            first=new and not self.active[j];self.active[j]=new
            measured=self.energy[-1,0,j]
            sensor_ok=bool(not (self.policy and self.policy.get('require_energy',False)) or
                           (measured[3]>.5 and measured[4]>.5 and np.isfinite(measured[0]) and np.isfinite(measured[1]) and
                            .5<=measured[5]<=1 and 1<=measured[2]<=2))
            eligible=supported and sensor_ok
            qualified=bool(eligible and np.isfinite(raw[index,5]) and raw[index,5]>=(threshold*.5 if self.qualified[j] else threshold))
            qualified_first=qualified and not self.qualified[j];self.qualified[j]=qualified
            status='supported' if supported else 'stale_or_incomplete' if available else 'insufficient_observation'
            output.append(dict(peer=int(j),status=status,risk_5s=float(raw[index,2]) if supported else None,
                risk_3to7s=float(raw[index,5]) if supported else None,
                advisory_risk_5s=float(raw[index,2]) if available else None,
                advisory_first_alert=bool(first),qualified_first_alert=bool(qualified_first),
                alarm_data_eligible=bool(eligible),trusted_alarm=bool(qualified_first and trusted)))
        return dict(status='ready',links=output,inference_ms=(time.perf_counter()-started)*1000,
                    trusted_alarm_enabled=trusted,hardware_validated=self.hardware_validated,
                    note='Research prediction; operational alarm requires calibrated and independently confirmed goals plus hardware validation')
