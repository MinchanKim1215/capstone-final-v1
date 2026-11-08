"""Optional first-alert monitor. Frozen validation rules, public predictions only."""
from pathlib import Path
import json
import numpy as np
from solo.contracts import ROOT,sha,sources,load_plan,object_hash


class AlarmMonitor:
    def __init__(self,checkpoint,selection):
        from solo.evaluate import analysis_sources
        policy=json.loads(Path(selection).read_text())
        if policy.get('format')!='standalone-transformer-selection-v5' or policy['validation_cases']!=len(load_plan()['validation']):
            raise ValueError('Alarm activation requires a full-validation selection; pilot rules remain diagnostic')
        if policy['selected_sha256']!=sha(checkpoint) or policy['source_sha256']!=sources() or policy['plan_sha256']!=object_hash(load_plan()):
            raise ValueError('Alarm rules do not match the checkpoint/data/code')
        if policy['analysis_sha256']!=analysis_sources():raise ValueError('Frozen alarm implementation changed')
        self.rules=policy['alarm_policy'];self.reset()
    def reset(self):self.active={name:np.zeros((5,8),bool) for name in self.rules};self.last_time=None
    def update(self,time_s,probability,supported):
        t=float(time_s);p=np.asarray(probability,float);support=np.asarray(supported,bool).copy()
        if not np.isfinite(t) or t<0 or abs(t/3-round(t/3))>1e-6:raise ValueError('Use 3-second decision timestamps')
        if self.last_time is not None and abs(t-self.last_time-3)>1e-6:raise ValueError('Consecutive decisions required; reset between missions')
        if p.shape!=(5,8,4) or support.shape!=(5,8):raise ValueError('Expected [5,8,4] probabilities and [5,8] support')
        support[np.arange(5),np.arange(3,8)]=False
        if not np.isfinite(p[support]).all() or np.any((p[support]<0)|(p[support]>1)):raise ValueError('Supported probabilities must be finite in [0,1]')
        out={}
        for name,r in self.rules.items():
            old=self.active[name]
            new=bool(r['enabled'])&(t>=21)&support&np.isfinite(p[...,r['head']])&(p[...,r['head']]>=np.where(old,r['exit_threshold'],r['threshold']))
            first=new&~old;self.active[name]=new
            out[name]=[dict(observer=int(i+3),peer=int(j),time_s=t,probability=float(p[i,j,r['head']])) for i,j in np.argwhere(first)]
        self.last_time=t
        return out
