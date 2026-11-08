"""Selection-only ensemble weights; underlying predictions remain auditable."""
import argparse
from pathlib import Path
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,digest,now
from .evaluate import evaluate


def create(first_summary,second_summary,weight,tag):
    a=read(first_summary);b=read(second_summary)
    if a['scope']!='selection' or b['scope']!='selection':raise ValueError('Ensemble weights are selected without calibration/confirmation outcomes')
    if a['dataset_sha256']!=b['dataset_sha256']:raise ValueError('Ensemble observations differ')
    first=read(a['model_manifest']);second=read(b['model_manifest'])
    members=[dict(path=a['model_manifest'],model_sha256=sha(a['model_manifest']),weight=weight),
             dict(path=b['model_manifest'],model_sha256=sha(b['model_manifest']),weight=1-weight)]
    model=ARTIFACTS/'models'/tag/'model.json'
    write(model,dict(family='ensemble',tag=tag,members=members,use_energy=bool(first['use_energy'] or second['use_energy']),
                     use_plan=bool(first.get('use_plan',False) or second.get('use_plan',False)),created_at=now(),
                     calibration_used=False,confirmation_used=False))
    records=read(a['dataset_path'])['rows'];rows=[r for r in records if r['role']=='selection']
    identity=digest([(r['key'],r['array_sha256']) for r in rows])[:16]
    folder=ARTIFACTS/'evaluation'/tag/identity/'selection';folder.mkdir(parents=True,exist_ok=True)
    other={case['key']:case for case in b['cases']}
    for case in a['cases']:
        match=other[case['key']]
        with np.load(case['predictions_path'],allow_pickle=False) as z:x=z['prediction']
        with np.load(match['predictions_path'],allow_pickle=False) as z:y=z['prediction']
        total=np.where(np.isfinite(x),x,0)*weight+np.where(np.isfinite(y),y,0)*(1-weight)
        den=np.isfinite(x)*weight+np.isfinite(y)*(1-weight)
        prediction=np.divide(total,den,out=np.full_like(total,np.nan),where=den>0)
        np.savez_compressed(folder/(case['key']+'.npz'),prediction=prediction)
    return evaluate(str(model),'selection','cuda',a['dataset_path'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',required=True);p.add_argument('--second',required=True)
    p.add_argument('--weight',type=float,default=.5);p.add_argument('--tag',required=True);a=p.parse_args()
    print(create(a.first,a.second,a.weight,a.tag))
