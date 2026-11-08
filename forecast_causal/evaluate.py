"""All eligible timepoints and original event denominators, service only."""
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify
from forecast_recall.episodes import cohort
from .inference import baseline,predict,pin
from .metrics import evaluate_rows
from forecast_extension.metrics import binary,alert_counts,combine_alerts


def verify_freeze():
    verify();f=read(ARTIFACTS/'freeze.json')
    for name,value in f['sources'].items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen V13 source changed: '+name)
    for i in f['candidates']:
        if i.get('pin'):
            for p,value in i['pin']['files'].items():
                if sha(p)!=value:raise ValueError('Frozen adapter changed')
        for key in ('selection_summary','calibration_summary','policy'):
            if sha(i[key])!=i[key+'_sha256']:raise ValueError('Frozen evidence changed')
    for key in ('data_manifest','training_plan','comparison_spec','protocol','experiment_plan','path_uncertainty','cause_rulebook','adoption_rule'):
        if sha(ARTIFACTS/(key+'.json'))!=f[key+'_sha256']:raise ValueError('Study design changed')
    return f


def data_records(scope):
    path=ARTIFACTS/('confirmation_data_manifest.json' if scope=='confirmation' else 'data_manifest.json')
    return [i for i in read(path)['records'] if i['role']==scope],path


def case_prediction(candidate,item,scope):
    path=ARTIFACTS/'predictions'/candidate['tag']/(item['key']+'.npz');meta=path.with_suffix('.json')
    contract=dict(data_sha256=item['sha256'],candidate=candidate)
    if meta.exists():
        old=read(meta)
        if old['contract']!=contract or sha(path)!=old['sha256']:raise ValueError('Cached risk changed')
        with np.load(path,allow_pickle=False) as a:return a['prediction'],path
    if candidate['family']=='baseline':p=baseline(item,scope)
    elif candidate['family']=='blend':p=sum(case_prediction(c,item,scope)[0]*w for c,w in zip(candidate['members'],candidate['weights']))
    else:
        from .inputs import load as load_inputs
        inputs=load_inputs(item)
        p=predict(candidate['model'],inputs)[:,None,None]
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=p);write(meta,dict(contract=contract,sha256=sha(path)));return p,path


def evaluate(candidate,scope):
    verify()
    if scope=='confirmation' and not any(i['candidate']==candidate for i in verify_freeze()['candidates']):raise ValueError('Unfrozen candidate')
    items,dataset=data_records(scope);folder=ARTIFACTS/'evaluation'/candidate['tag']/scope;folder.mkdir(parents=True,exist_ok=True);path=folder/'summary.json'
    if path.exists():
        old=read(path)
        if old['candidate']!=candidate or old['dataset_sha256']!=sha(dataset):raise ValueError('Summary binding changed')
        return path
    pools={k:[] for k in ('y5','y37','p5','p37')};cases=[];counts=[];data=[]
    for index,item in enumerate(items):
        ep=cohort(item);p,pp=case_prediction(candidate,item,scope);common=ep.valid&ep.legacy_support;up=common&ep.currently_up;data.append((ep,p))
        for key,value in [('p5',p[...,2][up]),('y5',ep.targets[...,2][up]),('p37',p[...,5][common]),('y37',ep.targets[...,5][common])]:pools[key].append(value)
        a=alert_counts(ep.links,ep.ends,p[...,5],ep.legacy_support,1);counts.append(a)
        cases.append(dict(key=item['key'],region=item['region'],kind=item['source_record']['spec']['kind'],source_sha256=item['source_record']['array_sha256'],
            predictions_path=str(pp),prediction_sha256=sha(pp),alerts=a,next5=binary(ep.targets[...,2][up],p[...,2][up]),any3to7=binary(ep.targets[...,5][common],p[...,5][common])))
        write(ARTIFACTS/'evaluation_status.json',dict(scope=scope,tag=candidate['tag'],complete=index+1,total=len(items),updated_at=now()))
        if index%10==0:print(scope,candidate['tag'],index+1,'/',len(items),flush=True)
    a=binary(np.concatenate(pools['y5']),np.concatenate(pools['p5']));b=binary(np.concatenate(pools['y37']),np.concatenate(pools['p37']))
    ranking=evaluate_rows(data,full=True) if scope=='selection' else None
    write(path,dict(task='service',scope=scope,model_tag=candidate['tag'],candidate=candidate,cases=cases,dataset_path=str(dataset),dataset_sha256=sha(dataset),
        next5_on_common_currently_up_links=a,any3to7_on_common_support=b,strict_alarms_at_half=combine_alerts(counts),
        selection_event_score=ranking,selection_score=None if ranking is None else ranking['score'],created_at=now(),production_ready=False))
    return path


def records(summary):
    if sha(summary['dataset_path'])!=summary['dataset_sha256']:raise ValueError('Dataset changed')
    lookup={i['key']:i for i in read(summary['dataset_path'])['records']};out=[]
    for case in summary['cases']:
        if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Risk changed')
        with np.load(case['predictions_path'],allow_pickle=False) as a:p=a['prediction']
        out.append((cohort(lookup[case['key']]),p))
    return out
