"""Secondary relevance diagnostic; local past traffic, never future routing."""
import numpy as np
from .contracts import read,write,sha,now
from . import ARTIFACTS
from forecast_rebuild.dataset import Episode
from forecast_rebuild.metrics import binary


def prepare_plan():
    path=ARTIFACTS/'secondary_diagnostics_plan.json'
    value=dict(created_at=now(),diagnostic='Recent locally observed data traffic',
        active_definition='At prediction endpoint, any own send attempt(field5) or received data packet(field14) during the preceding6 half-second frames',
        cohorts=['recent_data','no_recent_data'],outputs=['Common currently-up5s AP/positive rate','Common all-state3to7 AP/positive rate'],
        model_selection_used=False,alarm_thresholds_changed=False,
        limitation='Observed traffic is not a verified routing-critical link or end-to-end outage. AP across cohorts depends on prevalence.')
    if not path.exists():write(path,value)


def diagnostic(models):
    prepare_plan();output={}
    for tag,model in models.items():
        rows={r['key']:r for r in read(model['raw']['dataset_path'])['rows']}
        pools={name:{key:[] for key in ('p5','p37','y5','y37')} for name in ('recent_data','no_recent_data')}
        for case in model['raw']['cases']:
            ep=Episode(rows[case['key']]);v=np.asarray(ep.values);m=np.asarray(ep.masks)
            counts=np.where(m[...,5],v[...,5],0).astype(np.float32)+np.where(m[...,14],v[...,14],0)
            cumulative=np.concatenate([np.zeros_like(counts[:1]),np.cumsum(counts,axis=0,dtype=np.float32)])
            ends=np.asarray(ep.ends);active=(cumulative[ends]-cumulative[np.maximum(0,ends-6)])>0
            with np.load(case['predictions_path'],allow_pickle=False) as raw:p=raw['prediction']
            for name,group in [('recent_data',active),('no_recent_data',~active)]:
                support=ep.valid&ep.legacy_support&group;live=support&ep.currently_up;pool=pools[name]
                pool['p5'].append(p[...,2][live]);pool['y5'].append(ep.targets[...,2][live])
                pool['p37'].append(p[...,5][support]);pool['y37'].append(ep.targets[...,5][support])
        output[tag]={}
        for name,pool in pools.items():
            a=binary(np.concatenate(pool['y5']),np.concatenate(pool['p5']));b=binary(np.concatenate(pool['y37']),np.concatenate(pool['p37']))
            output[tag][name]=dict(next5=a,any3to7=b,positive_fraction5=a['positives']/a['samples'] if a['samples'] else None,
                                  positive_fraction37=b['positives']/b['samples'] if b['samples'] else None)
    return dict(plan_sha256=sha(ARTIFACTS/'secondary_diagnostics_plan.json'),models=output,model_selection_used=False,interpretation=read(ARTIFACTS/'secondary_diagnostics_plan.json')['limitation'])

