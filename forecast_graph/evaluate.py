"""Matched, unsampled first-relay evaluation; old confirmation never optimized."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status,V10
from .inference import predict,pin
from .episodes import physical,cohort
from forecast_rebuild.calibration import alarm_support
from forecast_extension.metrics import alert_counts,combine_alerts,binary


def verify_freeze():
    verify();f=read(ARTIFACTS/'freeze.json')
    for name,expected in f['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen V11 implementation changed: '+name)
    for item in f['candidates']:
        if item.get('pin'):
            for path,expected in item['pin']['files'].items():
                if sha(path)!=expected:raise ValueError('Frozen new model changed')
        for key in ('selection_summary','calibration_summary','policy'):
            if sha(item[key])!=item[key+'_sha256']:raise ValueError('Frozen evaluation or policy changed')
    if sha(ARTIFACTS/'data_manifest.json')!=f['data_manifest_sha256']:raise ValueError('Training data changed')
    for name in ('data_plan','training_plan','comparison_plan','tree_control_plan'):
        if sha(ARTIFACTS/(name+'.json'))!=f[name+'_sha256']:raise ValueError('Frozen study plan changed: '+name)
    return f


def data_records(scope):
    source=ARTIFACTS/('confirmation_data_manifest.json' if scope=='confirmation' else 'data_manifest.json')
    return [r for r in read(source)['records'] if r['role']==scope],source


@lru_cache(maxsize=4)
def old_cases(task,scope):
    bundle=read(V10/'research_candidates.json');tag=bundle['tasks'][task]['tag']
    path=V10/'evaluation'/task/tag/scope/'summary.json'
    if scope=='confirmation' or not path.exists():return {}
    return {r['key']:r for r in read(path)['cases']}


@lru_cache(maxsize=2)
def old_link_predictor(model):
    from forecast_rebuild.inference import Predictor
    return Predictor(model,'cuda')


def previous(task,item,arrays,ep,scope):
    record=item['source_record'];bundle=read(V10/'research_candidates.json');entry=bundle['tasks'][task]
    cached=old_cases(task,scope).get(record['key'])
    if cached:
        if cached['source_sha256']!=record['array_sha256']:raise ValueError('Prior prediction belongs to a different flight')
        path=cached['predictions_path']
        if sha(path)!=cached['prediction_sha256']:raise ValueError('Prior cached prediction changed')
        with np.load(path,allow_pickle=False) as a:p=a['prediction']
        q=arrays['query'];t,r=q.T
        return p[t,r] if task=='link' else p[t,0,0]
    if task=='service':
        from forecast_service.features import build
        from forecast_service.inference import tree_predict
        from forecast_rebuild.planning import public_context
        city,_=public_context(record);parts=[];q=arrays['query']
        for start in range(0,len(q),256):parts.append(tree_predict(entry['model'],build(ep,q[start:start+256],record['n_search'],city)))
        return np.concatenate(parts)
    q=arrays['query'];n=ep.values.shape[2];t,r=q.T
    rows=np.stack([np.repeat(t,n),np.repeat(r,n),np.tile(np.arange(n),len(q))],-1)
    model=old_link_predictor(entry['model']);parts=[]
    for start in range(0,len(rows),512):parts.append(model.predict(ep,rows[start:start+512])[0])
    return np.concatenate(parts).reshape(len(q),n,6)


def prediction_path(candidate,key):return ARTIFACTS/'predictions'/candidate['task']/candidate['tag']/(key+'.npz')


def case_prediction(candidate,item,ep,scope):
    record=item['source_record'];path=prediction_path(candidate,record['key']);meta=path.with_suffix('.json')
    contract=dict(data_sha256=item['sha256'],candidate=candidate)
    if meta.exists():
        old=read(meta)
        if old['contract']!=contract or sha(path)!=old['sha256']:raise ValueError('Existing predictions differ')
        with np.load(path,allow_pickle=False) as a:return a['prediction'],path
    with np.load(item['path'],allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    task=candidate['task'];family=candidate['family']
    if family=='previous':qvalues=previous(task,item,arrays,ep,scope)
    elif family=='blend':
        a,_=case_prediction(candidate['members'][0],item,ep,scope);b,_=case_prediction(candidate['members'][1],item,ep,scope)
        out=a*.5+b*.5;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=out)
        write(meta,dict(contract=contract,sha256=sha(path)));return out,path
    else:
        joint=ARTIFACTS/'joint_predictions'/Path(candidate['model']).parent.name/(record['key']+'.npz')
        joint_meta=joint.with_suffix('.json');joint_contract=dict(data_sha256=item['sha256'],pin=pin(candidate['model']))
        if joint_meta.exists():
            old=read(joint_meta)
            if old['contract']!=joint_contract or sha(joint)!=old['sha256']:raise ValueError('Joint prediction cache differs')
            with np.load(joint,allow_pickle=False) as a:result={name:a[name] for name in ('link','service')}
        else:
            result=predict(candidate['model'],arrays,record,ep,device='cuda')
            joint.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(joint,**result)
            write(joint_meta,dict(contract=joint_contract,sha256=sha(joint)))
        qvalues=result[task]
    target=cohort(record,task,ep);out=np.full((*target.valid.shape,6),np.nan,np.float32)
    q=arrays['query'];t,r=q.T
    if np.any(r!=0):raise ValueError('Primary evaluation is the predeclared first-relay cohort')
    if task=='link':out[t,0]=qvalues[:,:ep.values.shape[2]]
    else:out[t,0,0]=qvalues
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=out);write(meta,dict(contract=contract,sha256=sha(path)))
    return out,path


def evaluate(candidate,scope):
    verify()
    if scope=='confirmation':
        frozen=verify_freeze()
        if not any(i['candidate']==candidate for i in frozen['candidates']):raise ValueError('Unfrozen candidate')
    items,source=data_records(scope);task=candidate['task'];folder=ARTIFACTS/'evaluation'/task/candidate['tag']/scope;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'summary.json'
    if path.exists():
        old=read(path)
        if old['dataset_sha256']!=sha(source) or old['candidate']!=candidate:raise ValueError('Evaluation changed')
        return path
    pools={key:[] for key in ('p5','y5','p37','y37')};cases=[];counts=[]
    for i,item in enumerate(items):
        r=item['source_record'];ep=physical(r);target=cohort(r,task,ep);p,pp=case_prediction(candidate,item,ep,scope)
        common=target.valid&target.legacy_support;live=common&target.currently_up
        for key,value in [('p5',p[...,2][live]),('y5',target.targets[...,2][live]),('p37',p[...,5][common]),('y37',target.targets[...,5][common])]:pools[key].append(value)
        a=alert_counts(target.links,target.ends,p[...,5],alarm_support(target,task=='link'),target.record['n_search']);counts.append(a)
        cases.append(dict(key=r['key'],region=r['region'],kind=r['spec']['kind'],source_sha256=r['array_sha256'],
            predictions_path=str(pp),prediction_sha256=sha(pp),alerts=a,
            next5=binary(target.targets[...,2][live],p[...,2][live]),any3to7=binary(target.targets[...,5][common],p[...,5][common])))
        write(ARTIFACTS/'evaluation_status.json',dict(scope=scope,task=task,tag=candidate['tag'],complete=i+1,total=len(items),updated_at=now()))
        if i%10==0:print(scope,task,candidate['tag'],i+1,'/',len(items),flush=True)
    a=binary(np.concatenate(pools['y5']),np.concatenate(pools['p5']));b=binary(np.concatenate(pools['y37']),np.concatenate(pools['p37']))
    write(path,dict(task=task,scope=scope,model_tag=candidate['tag'],candidate=candidate,cases=cases,dataset_path=str(source),dataset_sha256=sha(source),
        next5_on_common_currently_up_links=a,any3to7_on_common_support=b,selection_score=(a['ap']+b['ap'])/2,
        strict_alarms_at_half=combine_alerts(counts),physical_primary_observer='first relay',created_at=now(),production_ready=False))
    return path


def records(summary):
    if sha(summary['dataset_path'])!=summary['dataset_sha256']:raise ValueError('Dataset changed')
    lookup={i['key']:i for i in read(summary['dataset_path'])['records']};result=[]
    for case in summary['cases']:
        item=lookup[case['key']];ep=physical(item['source_record']);target=cohort(item['source_record'],summary['task'],ep)
        if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Stored prediction changed')
        with np.load(case['predictions_path'],allow_pickle=False) as a:p=a['prediction']
        result.append((target,p))
    return result
