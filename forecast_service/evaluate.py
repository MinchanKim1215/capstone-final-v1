"""Fixed-observation evaluations; service and physical-link tasks stay distinct."""
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,digest,now,verify,V9,status
from .inference import pin,tree_only,tree_predict
from .feature_cache import FeatureEpisode
from .data import ServiceEpisode,query_rows
from .features import build
from forecast_rebuild.dataset import Episode
from forecast_rebuild.planning import public_context
from forecast_rebuild.inference import Predictor
from forecast_extension.metrics import binary,alert_counts,combine_alerts


def verify_freeze():
    verify();value=read(ARTIFACTS/'freeze.json')
    if sha(ARTIFACTS/'experiment_plan.json')!=value['experiment_plan_sha256']:raise ValueError('Experiment plan changed after freeze')
    for name,expected in value['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen implementation changed: '+name)
    for item in value['models']:
        for path,expected in item['pin']['files'].items():
            if sha(path)!=expected:raise ValueError('Frozen model changed')
        for name in ('selection_summary','calibration_summary','policy'):
            if sha(item[name])!=item[name+'_sha256']:raise ValueError('Frozen evidence changed: '+name)
    return value


def dataset(task,scope):
    path=ARTIFACTS/'datasets'/(task+'_'+scope+'.json')
    if path.exists():return path
    source=read(ARTIFACTS/'confirmation_dataset.json' if scope=='confirmation' else V9/'dataset.json')
    rows=[r for r in source['rows'] if r['role']==scope and (task=='link' or r['source_kind'] in ('fresh_v9','fresh_v10'))]
    write(path,dict(rows=rows,task=task,scope=scope,created_at=now(),original_final_test_opened=False));return path


@lru_cache(maxsize=12)
def parent_predictions(model,scope):
    frozen=read(V9/'freeze.json');item=next((i for i in frozen['models'] if i['pin']['path']==str(Path(model).resolve())),None)
    if item is None or scope=='confirmation':return {}
    source=read(item['selection_summary'] if scope=='selection' else item['calibration_summary'])
    return {case['key']:case for case in source['cases']}


def prediction_folder(model):return ARTIFACTS/'predictions'/(read(model)['tag']+'_'+digest(pin(model))[:12])


def link_prediction(model,record,scope):
    folder=prediction_folder(model);folder.mkdir(parents=True,exist_ok=True);path=folder/(record['key']+'.npz');header=path.with_suffix('.json')
    contract=dict(model_pin=pin(model),raw_source_sha256=record['array_sha256'],RF_manifest_sha256=sha(Path(record['cache_path'])/'manifest.json'),
        implementation_sha256=sha(__file__),cache_implementation_sha256=sha(ROOT/'feature_cache.py'))
    if header.exists():
        prior=read(header)
        if prior['contract']!=contract or sha(path)!=prior['sha256']:raise ValueError('Prediction cache differs')
        with np.load(path,allow_pickle=False) as a:return a['prediction'],path
    parent=parent_predictions(str(Path(model).resolve()),scope).get(record['key']);prediction=None
    if parent:
        if parent['source_sha256']!=record['array_sha256'] or sha(parent['predictions_path'])!=parent['prediction_sha256']:raise ValueError('Prior prediction input differs')
        with np.load(parent['predictions_path'],allow_pickle=False) as a:prediction=a['prediction']
    if prediction is None:
        ep=FeatureEpisode(record);q=np.argwhere(ep.valid);prediction=np.full((*ep.valid.shape,6),np.nan,np.float32)
        if tree_only(model):
            x=ep.features(q,read(model).get('use_energy',True),True);t,r,j=q.T;prediction[t,r,j]=tree_predict(model,x)
        else:
            predictor=Predictor(model,'cuda')
            for start in range(0,len(q),768):
                batch=q[start:start+768];t,r,j=batch.T;prediction[t,r,j]=predictor.predict(ep,batch)[0]
    temporary=path.with_suffix('.working.npz');np.savez_compressed(temporary,prediction=prediction);temporary.replace(path)
    write(header,dict(contract=contract,sha256=sha(path)));return prediction,path


def service_features(record):
    manifest=read(ARTIFACTS/'samples/manifest.json');lookup={r['key']:i for i,r in enumerate(manifest['records'])}
    if record['key'] in lookup and record['role'] in ('selection','calibration'):
        split=record['role'];index=np.load(ARTIFACTS/'samples'/(split+'_index.npy'),mmap_mode='r');take=np.flatnonzero(index[:,0]==lookup[record['key']])
        x=np.load(ARTIFACTS/'samples'/(split+'_x.npy'),mmap_mode='r');return index[take,1:],np.asarray(x[take])
    folder=ARTIFACTS/'service_features';folder.mkdir(exist_ok=True);path=folder/(record['key']+'.npz');header=path.with_suffix('.json')
    contract=dict(raw_sha256=record['array_sha256'],features_sha256=sha(ROOT/'features.py'),RF_manifest_sha256=sha(Path(record['cache_path'])/'manifest.json'))
    if header.exists():
        prior=read(header)
        if prior['contract']!=contract or sha(path)!=prior['sha256']:raise ValueError('Service input cache differs')
        with np.load(path,allow_pickle=False) as a:return a['query'],a['x']
    q=query_rows(record);ep=Episode(record);city,_=public_context(record);parts=[]
    for start in range(0,len(q),256):parts.append(build(ep,q[start:start+256],record['n_search'],city))
    x=np.concatenate(parts);np.savez_compressed(path,query=q,x=x);write(header,dict(contract=contract,sha256=sha(path)));return q,x


def service_prediction(model,record,scope):
    meta=read(model);ep=ServiceEpisode(record);out=np.full((*ep.valid.shape,6),np.nan,np.float32)
    if meta['family']=='service_proxy':
        raw,_=link_prediction(meta['link_model'],record,scope);physical=ep.physical;known=np.asarray(physical.legacy_support[:,0])
        for head in (2,5):
            values=np.where(known,raw[:,0,:,head],-np.inf).max(1);values[~np.isfinite(values)]=np.nan;out[:,0,0,head]=values
    else:
        query,x=service_features(record);p=tree_predict(model,x);out[query[:,0],0,0]=p
    folder=prediction_folder(model);folder.mkdir(parents=True,exist_ok=True);path=folder/(record['key']+'.npz')
    np.savez_compressed(path,prediction=out);return out,path


def evaluate(task,model,scope='selection'):
    verify()
    if scope=='confirmation':
        f=verify_freeze()
        if not any(i['task']==task and i['pin']['path']==str(Path(model).resolve()) for i in f['models']):raise ValueError('Unfrozen confirmation model')
    source=dataset(task,scope);rows=read(source)['rows'];tag=read(model)['tag'];folder=ARTIFACTS/'evaluation'/task/tag/scope;folder.mkdir(parents=True,exist_ok=True)
    summary_path=folder/'summary.json';model_pin=pin(model)
    if summary_path.exists():
        previous=read(summary_path)
        if previous['model_pin']!=model_pin or previous['dataset_sha256']!=sha(source):raise ValueError('Existing evaluation changed')
        return summary_path
    pools={key:[] for key in ('p5','p37','y5','y37')};cases=[];alerts=[];started=time.monotonic()
    for i,record in enumerate(rows):
        ep=Episode(record) if task=='link' else ServiceEpisode(record)
        p,path=link_prediction(model,record,scope) if task=='link' else service_prediction(model,record,scope)
        common=ep.valid&ep.legacy_support;live=common&ep.currently_up
        a=p[...,2][live];ya=ep.targets[...,2][live];b=p[...,5][common];yb=ep.targets[...,5][common]
        for key,value in [('p5',a),('y5',ya),('p37',b),('y37',yb)]:pools[key].append(value)
        counts=alert_counts(ep.links,ep.ends,p[...,5],ep.legacy_support,ep.record['n_search']);alerts.append(counts)
        cases.append(dict(key=record['key'],region=record['region'],kind=record['spec']['kind'],source_sha256=record['array_sha256'],
            next5=binary(ya,a),any3to7=binary(yb,b),alerts=counts,predictions_path=str(path),prediction_sha256=sha(path)))
        write(ARTIFACTS/'evaluation_status.json',dict(task=task,model=tag,scope=scope,complete=i+1,total=len(rows),updated_at=now()))
    next5=binary(np.concatenate(pools['y5']),np.concatenate(pools['p5']));any37=binary(np.concatenate(pools['y37']),np.concatenate(pools['p37']))
    write(summary_path,dict(task=task,scope=scope,model_tag=tag,model_manifest=str(Path(model).resolve()),model_pin=model_pin,
        dataset_path=str(source),dataset_sha256=sha(source),cases=cases,next5_on_common_currently_up_links=next5,any3to7_on_common_support=any37,
        selection_score=(next5['ap']+any37['ap'])/2,strict_alarms_at_half=combine_alerts(alerts),created_at=now(),wall_seconds=time.monotonic()-started,
        interpretation='Physical-link task unchanged; service task is one designated relay view of global packet deadline degradation. Task scores are not interchangeable.',production_ready=False))
    return summary_path
