"""Prepare the corrected cohort and compare frozen candidates before calibration."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
from . import ARTIFACTS,WORKSPACE
from .contracts import read,write,digest,now,sha
from .evaluate import evaluate


def command(module,args,tag):
    folder=ARTIFACTS/'benchmarks';folder.mkdir(parents=True,exist_ok=True)
    with (folder/(tag+'.stdout.log')).open('w',encoding='utf-8') as out,(folder/(tag+'.stderr.log')).open('w',encoding='utf-8') as err:
        result=subprocess.run([sys.executable,'-u','-B','-m',module,*args],cwd=WORKSPACE,
                              stdout=out,stderr=err,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    if result.returncode:raise RuntimeError(tag+' failed; inspect the preserved log')


def reuse(tag,dataset):
    rows=[r for r in read(dataset)['rows'] if r['role']=='selection']
    identity=digest([(r['key'],r['array_sha256']) for r in rows])[:16]
    folder=ARTIFACTS/'evaluation'/tag/identity/'selection';folder.mkdir(parents=True,exist_ok=True)
    candidates=list((ARTIFACTS/'evaluation'/tag).glob('*/selection/summary.json'))
    lookup={}
    for path in candidates:
        old=read(path)
        for case in old['cases']:lookup[case['key']]=case
    for record in rows:
        if record['key'] not in lookup:continue
        source=lookup[record['key']]
        if source['source_sha256']!=record['array_sha256']:raise ValueError('Cannot reuse predictions from another RF source')
        target=folder/(record['key']+'.npz')
        if not target.exists():shutil.copyfile(source['predictions_path'],target)


def main():
    folder=ARTIFACTS/'benchmarks';folder.mkdir(exist_ok=True)
    status=folder/'balanced_status.json'
    while True:
        state=read(ARTIFACTS/'balanced_collection_status.json')
        if state['state']=='completed':break
        if state['state']=='failed':raise RuntimeError('Balanced collection failed')
        write(status,dict(state='waiting_for_balanced_collection',complete=state['complete'],total=state['total'],updated_at=now()))
        time.sleep(5)
    command('forecast_rebuild.prepare_data',['--workers','3'],'prepare_balanced_data')
    command('forecast_rebuild.prepare_plans',['--workers','2'],'prepare_balanced_public_plans')
    data=read(ARTIFACTS/'dataset.json')
    rows=[r for r in data['rows'] if r['role']=='train' or r['source_kind']=='fresh_v7']
    groups={}
    for row in rows:
        if row['role']=='train':continue
        groups.setdefault((row['role'],row['region']),[]).append(row['spec']['kind'])
    if any(sorted(k)!=['clear','continuous','intermittent'] for k in groups.values()):raise ValueError('Balanced cohort is incomplete')
    dataset=folder/'balanced_development_dataset.json'
    write(dataset,dict(rows=rows,cohort='One fresh flight per map/direction/interference condition',
        counts={k:sum(r['role']==k for r in rows) for k in ('train','selection','calibration')},
        confirmation_opened=False,created_at=now()))
    tags=['expanded_lgbm_base','expanded_lgbm_energy','expanded_residual_base','expanded_residual_energy',
          'expanded_lgbm_planned_energy','expanded_residual_planned_energy','expanded_dense_residual_energy']
    models=[('frozen_v7_64','baseline')]+[(tag,str(ARTIFACTS/'models'/tag/'model.json')) for tag in tags]
    results=[]
    for tag,model in models:
        if model!='baseline' and not Path(model).exists():raise ValueError('A declared candidate is missing: '+tag)
        reuse(tag,dataset)
        path=evaluate(model,'selection','cuda',dataset)
        from .calibration import audit_cohort
        audit_cohort(read(path));results.append(str(path))
        write(status,dict(state='evaluating',complete=len(results),total=len(models),results=results,updated_at=now()))
    write(folder/'balanced_selection_results.json',dict(results=results,dataset_path=str(dataset),dataset_sha256=sha(dataset),
        scope='Balanced selection only. Calibration and confirmation outcomes unused.',created_at=now()))
    write(status,dict(state='completed',complete=len(results),total=len(models),results=results,updated_at=now()))
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
