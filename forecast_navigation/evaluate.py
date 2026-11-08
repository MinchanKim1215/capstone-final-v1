"""Fast exact first-relay metrics on compact, unchanged ground-truth cohorts."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,digest,now,verify,status,PARENT
from .data import arrays
from .episodes import cohort
from .inference import predict,pin
from forecast_rebuild.calibration import alarm_support
from forecast_extension.metrics import binary,alert_counts,combine_alerts


def verify_freeze():
    verify();f=read(ARTIFACTS/'freeze.json')
    for name,expected in f['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen V12 source changed: '+name)
    for item in f['candidates']:
        if item.get('pin'):
            for path,expected in item['pin']['files'].items():
                if sha(path)!=expected:raise ValueError('V12 model changed')
        for key in ('selection_summary','calibration_summary','policy'):
            if sha(item[key])!=item[key+'_sha256']:raise ValueError('Frozen evidence changed')
    for name in ('data_manifest','training_plan','comparison_spec'):
        if sha(ARTIFACTS/(name+'.json'))!=f[name+'_sha256']:raise ValueError('Frozen study design changed')
    return f


def data_records(scope):
    path=ARTIFACTS/('confirmation_data_manifest.json' if scope=='confirmation' else 'data_manifest.json')
    return [i for i in read(path)['records'] if i['role']==scope],path


def prediction_path(candidate,key):return ARTIFACTS/'predictions'/candidate['task']/candidate['tag']/(key+'.npz')


def case_prediction(candidate,item,scope):
    path=prediction_path(candidate,item['key']);meta=path.with_suffix('.json')
    contract=dict(base_sha256=item['base_sha256'],mission_sha256=item['mission_sha256'],cohort_sha256=item['cohort_sha256'],candidate=candidate)
    if meta.exists():
        old=read(meta)
        if old['contract']!=contract or sha(path)!=old['sha256']:raise ValueError('Prediction cache changed')
        with np.load(path,allow_pickle=False) as a:return a['prediction'],path
    task=candidate['task'];family=candidate['family']
    if family=='blend':
        a,_=case_prediction(candidate['members'][0],item,scope);b,_=case_prediction(candidate['members'][1],item,scope);out=(a+b)*.5
    elif family=='parent':
        if scope=='confirmation':
            with np.load(item['reference_path'],allow_pickle=False) as a:value=a[task+'_'+candidate['parent_kind']]
        else:
            source=PARENT/'predictions'/task/candidate['parent_candidate']['tag']/(item['key']+'.npz')
            with np.load(item['base_path'],allow_pickle=False) as a:t=a['query'][:,0]
            with np.load(source,allow_pickle=False) as a:p=a['prediction']
            value=p[t,0] if task=='link' else p[t,0,0]
        out=value[:,None] if task=='link' else value[:,None,None]
    else:
        joint=ARTIFACTS/'joint_predictions'/candidate['tag']/(item['key']+'.npz');jm=joint.with_suffix('.json')
        jc=dict(base_sha256=item['base_sha256'],mission_sha256=item['mission_sha256'],pin=pin(candidate['model']),profile=candidate['profile'])
        if jm.exists():
            old=read(jm)
            if old['contract']!=jc or sha(joint)!=old['sha256']:raise ValueError('Joint model prediction changed')
            with np.load(joint,allow_pickle=False) as a:result={task:a[task] for task in ('link','service')}
        else:
            result=predict(candidate['model'],arrays(item,candidate['profile']),device='cuda');joint.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(joint,**result);write(jm,dict(contract=jc,sha256=sha(joint)))
        target=cohort(item,task)
        out=result['link'][:,:target.valid.shape[-1]][:,None] if task=='link' else result['service'][:,None,None]
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=out);write(meta,dict(contract=contract,sha256=sha(path)))
    return out,path


def evaluate(candidate,scope):
    verify()
    if scope=='confirmation' and not any(i['candidate']==candidate for i in verify_freeze()['candidates']):raise ValueError('Unfrozen V12 candidate')
    items,source=data_records(scope);folder=ARTIFACTS/'evaluation'/candidate['task']/candidate['tag']/scope;folder.mkdir(parents=True,exist_ok=True);path=folder/'summary.json'
    if path.exists():
        old=read(path)
        if old['candidate']!=candidate or old['dataset_sha256']!=sha(source):raise ValueError('Evaluation changed')
        return path
    pools={k:[] for k in ('p5','p37','y5','y37')};cases=[];counts=[]
    for index,item in enumerate(items):
        ep=cohort(item,candidate['task']);p,pp=case_prediction(candidate,item,scope);common=ep.valid&ep.legacy_support;live=common&ep.currently_up
        for key,value in [('p5',p[...,2][live]),('y5',ep.targets[...,2][live]),('p37',p[...,5][common]),('y37',ep.targets[...,5][common])]:pools[key].append(value)
        a=alert_counts(ep.links,ep.ends,p[...,5],alarm_support(ep,candidate['task']=='link'),ep.record['n_search']);counts.append(a)
        cases.append(dict(key=item['key'],region=item['region'],kind=item['source_record']['spec']['kind'],source_sha256=item['source_record']['array_sha256'],
            predictions_path=str(pp),prediction_sha256=sha(pp),alerts=a,next5=binary(ep.targets[...,2][live],p[...,2][live]),any3to7=binary(ep.targets[...,5][common],p[...,5][common])))
        write(ARTIFACTS/'evaluation_status.json',dict(scope=scope,task=candidate['task'],tag=candidate['tag'],complete=index+1,total=len(items),updated_at=now()))
        if index%10==0:print(scope,candidate['task'],candidate['tag'],index+1,'/',len(items),flush=True)
    a=binary(np.concatenate(pools['y5']),np.concatenate(pools['p5']));b=binary(np.concatenate(pools['y37']),np.concatenate(pools['p37']))
    write(path,dict(task=candidate['task'],scope=scope,model_tag=candidate['tag'],candidate=candidate,dataset_path=str(source),dataset_sha256=sha(source),cases=cases,
        next5_on_common_currently_up_links=a,any3to7_on_common_support=b,selection_score=(a['ap']+b['ap'])/2,strict_alarms_at_half=combine_alerts(counts),
        created_at=now(),production_ready=False))
    return path


def records(summary):
    if sha(summary['dataset_path'])!=summary['dataset_sha256']:raise ValueError('Evaluated dataset changed')
    lookup={i['key']:i for i in read(summary['dataset_path'])['records']};result=[]
    for case in summary['cases']:
        if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Cached forecast changed')
        with np.load(case['predictions_path'],allow_pickle=False) as a:p=a['prediction']
        result.append((cohort(lookup[case['key']],summary['task']),p))
    return result
