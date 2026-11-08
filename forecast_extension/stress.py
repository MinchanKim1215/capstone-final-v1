"""Fixed query diagnostic of B/C/winner under measurement perturbations."""
from copy import copy
from pathlib import Path
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,digest,now,maps,SEED
from .belief import create,noisy_energy
from forecast_rebuild.dataset import Episode
from forecast_rebuild.inference import Predictor
from forecast_rebuild.metrics import binary


def main():
    completed=read(ARTIFACTS/'completed_comparison.json');registry=maps();data=read(ARTIFACTS/'dataset.json')
    records=[r for r in data['rows'] if r['role']=='selection' and r['source_kind']=='fresh_v9'];regions=[]
    for added in (False,True):
        candidates=sorted({r['region'] for r in records if (registry[r['region']]['source_dataset']!='legacy_v6')==added})
        regions.append(candidates[0])
    records=[r for r in records if r['region'] in regions];queries={}
    for r in records:
        eligible=Episode(r).eligible(legacy=True);rng=np.random.default_rng(int(digest(['stress-query',SEED,r['spec']])[:16],16))
        queries[r['key']]=eligible[rng.choice(len(eligible),min(6000,len(eligible)),replace=False)]
    models=[str(ARTIFACTS/'models/same_data_20000/model.json'),str(ARTIFACTS/'models/new_data_20000/model.json'),completed['selected_model']]
    models=list(dict.fromkeys(models));outputs=[]
    for kind in ('clean','energy_missing','energy_noise','gps_sigma2m'):
        episodes={}
        for r in records:
            episode=Episode(r)
            if kind=='gps_sigma2m':
                key=digest(['stress-gps',r['key']])[:20]
                changed=dict(r,key=key,cache_path=str(ARTIFACTS/'stress_cache'/key),belief_profile=dict(gps_axis_sigma_m=2,energy_mode='clean'))
                create(r,changed,queries[r['key']]);episode=Episode(changed)
            elif kind.startswith('energy_'):
                mode='missing' if kind=='energy_missing' else 'noise'
                episode.energy=noisy_energy(episode.energy,r['n_search'],np.random.default_rng(int(digest(['stress-energy',r['spec']])[:16],16)),mode).astype(np.float16)
            episodes[r['key']]=episode
        for model in models:
            predictor=Predictor(model,'cuda');p5=[];p37=[];y5=[];y37=[]
            for record in records:
                episode=episodes[record['key']];query=queries[record['key']];t,r,j=query.T;parts=[]
                for start in range(0,len(query),768):parts.append(predictor.predict(episode,query[start:start+768])[0])
                p=np.concatenate(parts);up=episode.currently_up[t,r,j]
                p5.append(p[up,2]);y5.append(episode.targets[t,r,j,2][up]);p37.append(p[:,5]);y37.append(episode.targets[t,r,j,5])
            outputs.append(dict(profile=kind,tag=predictor.meta['tag'],ap5=binary(np.concatenate(y5),np.concatenate(p5))['ap'],
                                ap37=binary(np.concatenate(y37),np.concatenate(p37))['ap']))
            print(kind,predictor.meta['tag'],flush=True)
    result=dict(created_at=now(),profiles=outputs,regions=regions,flights=len(records),max_queries_per_flight=6000,
        note='Selection-only fixed query ranking diagnostic. Not full-trajectory first-alert recall, independent confirmation, or measured hardware performance.')
    write(ARTIFACTS/'stress_results.json',result);print(ARTIFACTS/'stress_results.json')


if __name__=='__main__':main()
