"""Identical-observation evaluation with model/input/source-aware caches."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,digest,now,verify_protocol,BASELINE
from .feature_cache import CachedEpisode as Episode
from forecast_rebuild.inference import Predictor
from .metrics import binary,alert_counts,combine_alerts


def model_pin(model):
    if model=='baseline':return dict(path='baseline',sha256=sha(BASELINE),files={str(BASELINE):sha(BASELINE)})
    path=Path(model).resolve();meta=read(path);files={str(path):sha(path)}
    if meta['family']=='lightgbm':
        for item in meta['heads'].values():
            if sha(item['path'])!=item['sha256']:raise ValueError('Tree weights changed')
            files[item['path']]=item['sha256']
    elif meta['family']=='ensemble':
        for member in meta['members']:
            if sha(member['path'])!=member['model_sha256']:raise ValueError('Ensemble member changed')
            files.update(model_pin(member['path'])['files'])
    else:
        if sha(meta['checkpoint'])!=meta['sha256']:raise ValueError('Neural weights changed')
        files[meta['checkpoint']]=meta['sha256']
    return dict(path=str(path),sha256=sha(path),files=files)


def verify_freeze():
    verify_protocol();value=read(ARTIFACTS/'freeze.json')
    if sha(ARTIFACTS/'comparison_plan.json')!=value['comparison_plan_sha256']:raise ValueError('Frozen comparison plan changed')
    for name,expected in value['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen V9 implementation changed: '+name)
    for item in value['models']:
        for path,expected in item['pin']['files'].items():
            if sha(path)!=expected:raise ValueError('A frozen model changed')
        if sha(item['policy'])!=item['policy_sha256']:raise ValueError('A frozen alarm policy changed')
        if sha(item['selection_summary'])!=item['selection_sha256']:raise ValueError('Selection evidence changed')
        if sha(item['calibration_summary'])!=item['calibration_sha256']:raise ValueError('Calibration evidence changed')
    return value


def freeze(items,winner):
    value=dict(created_at=now(),selected_model=winner,models=items,
        sources={name:sha(ROOT/name) for name in ('__init__.py','contracts.py','samples.py','train.py','collect.py','data.py','belief.py','build_samples.py','train_tree.py','evaluate.py','calibration.py','metrics.py','selection.py','feature_cache.py')},
        comparison_plan_sha256=sha(ARTIFACTS/'comparison_plan.json'),
        confirmation_type='Fresh seed and opposite direction on known confirmation maps; not novel geography',
        original_final_test_opened=False,production_ready=False,model_selection_after_confirmation_forbidden=True)
    path=ARTIFACTS/'freeze.json'
    if path.exists():
        existing=read(path)
        if existing['selected_model']!=winner or existing['models']!=items:raise ValueError('A frozen comparison cannot change')
        verify_freeze();return path
    write(path,value);return path


def baseline_archive(record,episode):
    metadata=read(record['metadata_path'])
    if metadata.get('request',{}).get('checkpoint_sha256')!=sha(BASELINE):return None
    with np.load(record['array_path'],allow_pickle=False) as raw:old=raw['aux_probabilities'];ends=raw['aux_endpoints']
    lookup={int(end):i for i,end in enumerate(ends)};ix=[lookup[int(end)] for end in episode.ends]
    out=np.full((*episode.valid.shape,6),np.nan,np.float32)
    out[...,2]=old[ix,...,0];out[...,4]=old[ix,...,1];out[...,5]=old[ix,...,2]
    return out


def evaluate(model,scope='selection',dataset=None):
    verify_protocol();dataset=Path(dataset) if dataset else ARTIFACTS/'dataset.json'
    corpus=read(dataset);rows=[r for r in corpus['rows'] if r['role']==scope]
    if scope=='confirmation':
        frozen=verify_freeze();allowed={item['pin']['path'] for item in frozen['models']}
        normalized='baseline' if model=='baseline' else str(Path(model).resolve())
        if normalized not in allowed:raise ValueError('This predictor is not frozen for confirmation')
    if not rows:raise ValueError('No cases for '+scope)
    predictor=Predictor(model,'cuda');tag=predictor.meta['tag'];pinned=model_pin(model)
    identity=digest(dict(model=pinned,input=[(r['key'],r['array_sha256'],sha(Path(r['cache_path'])/'manifest.json')) for r in rows],
                         implementation=dict(evaluate=sha(Path(__file__)),feature_cache=sha(ROOT/'feature_cache.py'),metrics=sha(ROOT/'metrics.py'))))[:16]
    folder=ARTIFACTS/'evaluation'/tag/identity/scope;folder.mkdir(parents=True,exist_ok=True)
    summary_path=folder/'summary.json'
    if summary_path.exists():
        existing=read(summary_path)
        if existing['dataset_sha256']==sha(dataset) and existing['model_pin']==pinned:return summary_path
    p5=[];p37=[];y5=[];y37=[];cases=[];alarms=[];expanded=[];started=time.monotonic()
    for record in rows:
        episode=Episode(record);path=folder/(record['key']+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as raw:prediction=raw['prediction']
        else:
            prediction=baseline_archive(record,episode) if model=='baseline' else None
            if predictor.meta['family']=='ensemble':
                components=[]
                for member in predictor.meta['members']:
                    tag_member=read(member['path'])['tag'];matched=None
                    for summary_file in (ARTIFACTS/'evaluation'/tag_member).glob('*/'+scope+'/summary.json'):
                        summary=read(summary_file)
                        if summary['dataset_sha256']!=sha(dataset) or summary['model_pin']!=model_pin(member['path']):continue
                        case=next((c for c in summary['cases'] if c['key']==record['key']),None)
                        if case and sha(case['predictions_path'])==case['prediction_sha256']:
                            with np.load(case['predictions_path'],allow_pickle=False) as raw:matched=raw['prediction']
                            break
                    if matched is None:components=[];break
                    components.append((matched,member['weight']))
                if components:
                    total=sum(np.where(np.isfinite(p),p,0)*weight for p,weight in components)
                    weights=sum(np.isfinite(p)*weight for p,weight in components)
                    prediction=np.divide(total,weights,out=np.full_like(total,np.nan),where=weights>0)
            if prediction is None:
                prediction=np.full((*episode.valid.shape,6),np.nan,np.float32);eligible=np.argwhere(episode.valid)
                for start in range(0,len(eligible),768):
                    batch=eligible[start:start+768];t,r,j=batch.T;prediction[t,r,j]=predictor.predict(episode,batch)[0]
            np.savez_compressed(path,prediction=prediction)
        support=episode.valid&episode.legacy_support;live=support&episode.currently_up
        a=prediction[...,2][live];b=prediction[...,5][support];ya=episode.targets[...,2][live];yb=episode.targets[...,5][support]
        p5.append(a);p37.append(b);y5.append(ya);y37.append(yb)
        alert=alert_counts(episode.links,episode.ends,prediction[...,5],episode.legacy_support,record['n_search'])
        alarms.append(alert);expanded.append(alert_counts(episode.links,episode.ends,prediction[...,5],episode.available,record['n_search']))
        cases.append(dict(key=record['key'],region=record['region'],source_kind=record['source_kind'],kind=record['spec']['kind'],direction=record['spec']['direction'],
            source_sha256=record['array_sha256'],next5=binary(ya,a),any3to7=binary(yb,b),alerts=alert,
            predictions_path=str(path),prediction_sha256=sha(path)))
        write(ARTIFACTS/'evaluation_status.json',dict(state='running',model=tag,scope=scope,complete=len(cases),total=len(rows),updated_at=now()))
        print(json.dumps(dict(model=tag,scope=scope,complete=len(cases),total=len(rows))),flush=True)
    a=binary(np.concatenate(y5),np.concatenate(p5));b=binary(np.concatenate(y37),np.concatenate(p37))
    value=dict(model_tag=tag,model_manifest=pinned['path'],model_pin=pinned,scope=scope,dataset_path=str(dataset.resolve()),dataset_sha256=sha(dataset),
        cases=cases,regions=len({r['region'] for r in rows}),next5_on_common_currently_up_links=a,any3to7_on_common_support=b,
        strict_alarms_at_half=combine_alerts(alarms),expanded_support_alarms_at_half=combine_alerts(expanded),selection_score=(a['ap']+b['ap'])/2,
        created_at=now(),wall_seconds=time.monotonic()-started,original_final_test_opened=False,
        interpretation='Prediction-only on identical fixed-v7-policy observations; confirmation uses new flights on known maps')
    write(summary_path,value);return summary_path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='baseline');p.add_argument('--scope',default='selection');p.add_argument('--dataset')
    a=p.parse_args();print(evaluate(a.model,a.scope,a.dataset))

