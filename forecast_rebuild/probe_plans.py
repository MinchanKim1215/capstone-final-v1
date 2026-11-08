"""Outage-conditioned diagnostic; future positions are used only for scoring."""
import json
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,now
from .dataset import Episode
from .planning import public_context,profile


def main():
    records=read(ARTIFACTS/'samples/engineering/manifest.json')['contract']['records']
    selected=[r for r in records if r['role']=='selection' and r['region'] in ('gyeonggi_20261030_bh3_01','gyeonggi_20261026_batch2_10')][-2:]
    results=[]
    for record in selected:
        ep=Episode(record)
        rows=np.argwhere(ep.valid&ep.currently_up&ep.legacy_support&((ep.next_bin==6)|(ep.next_bin==7)))
        if len(rows)>400:rows=rows[np.random.default_rng(17).choice(len(rows),400,replace=False)]
        city,nominal=public_context(record)
        predicted=profile(city,nominal,ep.values,ep.masks,ep.intents,ep.ends,record['n_search'],only_rows=rows)
        t,r,j=rows.T;g=predicted[t,r,j];old=np.asarray(ep.geometry[t,r,j]);support=(g[:,55]>.5)&(old[:,-1]>.5)
        t,r,j=t[support],r[support],j[support];g=g[support];old=old[support]
        onset=np.asarray(ep.ends)[t]+ep.next_bin[t,r,j]
        h=(ep.next_bin[t,r,j]-1).astype(int)
        with np.load(record['array_path'],allow_pickle=False) as archive:positions=archive['positions']
        step=onset//6;alpha=(onset%6)/6
        actual=positions[step]*(1-alpha[:,None,None])+positions[step+1]*alpha[:,None,None]
        a=np.c_[actual[np.arange(len(actual)),r+record['n_search']],np.full(len(actual),50.)]
        b=np.c_[actual[np.arange(len(actual)),j],np.full(len(actual),50.)]
        _,los,_=city.radio.propagation(a,b,city._forecast_scenario)
        old_los=old[np.arange(len(old)),13+h]>.5
        new_los=g[np.arange(len(g)),13+h]>.5
        row=dict(region=record['region'],points=len(los),actual_nlos=int((~los).sum()),
                 old_los_agreement=float((old_los==los).mean()),new_los_agreement=float((new_los==los).mean()),
                 old_nlos_recall=float((~old_los&~los).sum()/max(1,(~los).sum())),
                 new_nlos_recall=float((~new_los&~los).sum()/max(1,(~los).sum())))
        results.append(row);print(json.dumps(row),flush=True)
    write(ARTIFACTS/'planning_probe.json',dict(created_at=now(),scope='Selection-only outage-conditioned geometry diagnostic; not model accuracy',cases=results))


if __name__=='__main__':main()
