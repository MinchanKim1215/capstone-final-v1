"""Paired validation summaries; resample whole spatial regions, never RF frames."""
import json
from pathlib import Path
import numpy as np


def paired_comparison(reference,candidate,replicates=10000,seed=20261029):
    def rows(result):
        root=Path(__file__).resolve().parents[1]
        entries=[json.loads((root/p).with_suffix('.json').read_text()) for p in result['raw_paths']]
        return {(e['region'],e['direction'],e['kind'],e['seed']):e for e in entries}
    a=rows(reference);b=rows(candidate)
    if a.keys()!=b.keys():raise ValueError('Paired evaluation needs identical scenario specifications')
    fields=['search_connected_mean','pdr','timely_pdr','outage_total_s','outage_max_s','duration_ratio']
    regions=sorted({key[0] for key in a});rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(regions),size=(replicates,len(regions)));out={}
    for field in fields:
        delta=np.array([np.mean([b[k][field]-a[k][field] for k in a if k[0]==r]) for r in regions])
        bootstrap=delta[samples].mean(1)
        out[field]=dict(mean_delta=float(delta.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                        per_region=dict(zip(regions,delta.tolist())))
    return dict(regions=len(regions),episodes=len(a),replicates=replicates,seed=seed,
                method='Paired spatial-region bootstrap; validation pilot, not independent final-test evidence',metrics=out)
