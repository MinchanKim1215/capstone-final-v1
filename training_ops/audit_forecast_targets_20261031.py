"""Offline future-label reference alarms, not deployable prediction results."""
from pathlib import Path
import json
import math
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from adaptive_joint.metrics import alert_counts,summarize_alerts


def main():
    folder=ROOT/'adaptive_joint/artifacts/runs/adaptive_v6_20261030/validation/u0000144/rollouts'
    names=('next_currently_up','any_currently_up','next_all_states','any_all_states')
    counts={k:[] for k in names};events=compatible=observed=0
    for p in sorted(folder.glob('*.json')):
        row=json.loads(p.read_text(encoding='utf-8'));ns=row['search_drones']
        with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
            a={k:z[k] for k in ('links','aux_endpoints','aux_support','aux_labels','aux_valid')}
        up=a['links'];starts=np.zeros_like(up,bool);starts[1:-1]=up[:-2]&~up[1:-1]&~up[2:]
        next_score=(a['aux_labels']>=6)&(a['aux_labels']<14)
        any_score=np.stack([starts[e+6:e+14,ns:].any(0) if e+16<len(up) else np.zeros_like(next_score[0]) for e in a['aux_endpoints']])
        scores=dict(next_currently_up=next_score&a['aux_valid'],any_currently_up=any_score&a['aux_valid'],
                    next_all_states=next_score,any_all_states=any_score)
        for name,score in scores.items():
            probability=np.zeros((*score.shape,4));probability[...,2]=score
            counts[name].append(alert_counts(dict(a,aux_probabilities=probability),ns))
        times=a['aux_endpoints']*.5;end=len(up)*.5-8.5
        for i in range(ns,up.shape[1]):
            for j in range(up.shape[1]):
                if i==j:continue
                onsets=np.flatnonzero(starts[:,i,j])*.5
                for e in onsets[(onsets>=24)&(onsets<=end)]:
                    use=(times>=21)&(times<=end)&(times>e-7)&(times<=e-3)
                    events+=1
                    observed+=int(a['aux_support'][use,i-ns,j].any())
                    compatible+=int((a['aux_support'][use,i-ns,j]&scores['next_currently_up'][use,i-ns,j]).any())
    result=dict(scope='Offline future-truth reference ONLY, not actual accuracy and not a universal mathematical performance bound',
                cases=8,events=events,events_with_causal_support=observed,causal_support_fraction=observed/events,
                events_with_label_compatible_supported_pre_alert=compatible,label_compatible_fraction=compatible/events,
                reference_alarms={k:summarize_alerts(v) for k,v in counts.items()},
                interpretation='0/1 scores are the future labels themselves. Currently-up variants abstain where current next-event training target is invalid. All use unchanged first-alert hysteresis. Differences diagnose target/alarm semantics, not gains achieved by a learned model.')
    (ROOT/'training_ops/reports/environment_target_alignment_20261031.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
