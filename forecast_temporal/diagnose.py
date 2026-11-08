"""Selection-only post-fit diagnosis; never changes frozen models or policies."""
from collections import Counter
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,now
from .evaluate import records
from forecast_recall.diagnose import starts
from forecast_rebuild.targets import onsets
from forecast_rebuild.calibration import transform

def main():
    frozen=read(ARTIFACTS/'freeze.json');models={}
    for entry in frozen['candidates']:
        summary=read(entry['selection_summary']);policy=read(entry['policy']);causes=Counter();maximum=[];lead=[]
        by_kind={k:Counter() for k in ('clear','intermittent','continuous')}
        for case,(ep,p) in zip(summary['cases'],records(summary)):
            score=transform(p,policy['parameters'])[...,5];support=ep.legacy_support;alarms=starts(ep,score,support,policy['threshold']);aa=np.array([t for t,a in alarms if a[0,0]])
            times=ep.ends*.5;last=len(ep.links)*.5-8.5;events=np.flatnonzero(onsets(ep.links)[:,1,0])*.5;events=events[(events>=24)&(events<=last)]
            for when in events:
                pre=(times>when-7)&(times<=when-3)&(times>=21);known=pre&support[:,0,0]&np.isfinite(score[:,0,0]);hit=bool(((aa>when-7)&(aa<=when-3)).any());high=bool((score[known,0,0]>=policy['threshold']).any())
                cause='timely_first_alert' if hit else 'no_qualified_input' if not known.any() else 'risk_below_threshold' if not high else 'already_active_without_new_alert'
                causes[cause]+=1;by_kind[case['kind']][cause]+=1
                if known.any():maximum.append(float(score[known,0,0].max()))
                row=[]
                for offset in range(-12,1):
                    q=np.flatnonzero((times>when+offset-.51)&(times<=when+offset+.01)&support[:,0,0]&np.isfinite(score[:,0,0]));row.append(float(score[q[-1],0,0]) if len(q) else np.nan)
                lead.append(row)
        models[entry['candidate']['tag']]=dict(events=sum(causes.values()),causes=dict(causes),by_interference={k:dict(v) for k,v in by_kind.items()},threshold=policy['threshold'],policy_sha256=sha(entry['policy']),
            prewindow_max_risk_quantiles=np.quantile(maximum,[0,.1,.25,.5,.75,.9,1]).tolist(),lead_offsets_s=list(range(-12,1)),median_onset_aligned_risk=np.nanmedian(lead,axis=0).tolist())
    value=dict(created_at=now(),scope='Selection45 only, post-fit diagnostic; no confirmation prediction access or model/policy change',freeze_sha256=sha(ARTIFACTS/'freeze.json'),models=models)
    write(ARTIFACTS/'selection_diagnosis.json',value);print({k:v['causes'] for k,v in models.items()},flush=True)

if __name__=='__main__':main()
