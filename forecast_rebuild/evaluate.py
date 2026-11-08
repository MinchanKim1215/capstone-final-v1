"""Full trajectory prediction evaluation with common support and common truth."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,digest,sha,now
from .dataset import Episode
from .inference import Predictor
from .metrics import binary,alert_counts,combine_alerts


def evaluate(model,scope='selection',device='cuda',dataset=None):
    dataset=Path(dataset) if dataset else ARTIFACTS/'dataset.json'
    corpus=read(dataset);rows=[r for r in corpus['rows'] if r['role']==scope]
    if not rows:raise ValueError('No prepared cases for '+scope)
    if scope=='confirmation' and not (ARTIFACTS/'model_alarm_freeze.json').exists():
        raise ValueError('Confirmation cannot be used for model selection')
    if scope=='confirmation':
        from .calibration import verify_freeze
        frozen=verify_freeze()
        if model not in ('baseline',frozen['selected_model']):raise ValueError('This model was not included in the frozen confirmation comparison')
    predictor=Predictor(model,device);tag=predictor.meta['tag']
    identity=digest([(r['key'],r['array_sha256']) for r in rows])[:16]
    folder=ARTIFACTS/'evaluation'/tag/identity/scope;folder.mkdir(parents=True,exist_ok=True)
    p5=[];p37=[];y5=[];y37=[];alerts=[];expanded=[];cases=[];reproduced=[]
    started=time.monotonic()
    for record in rows:
        episode=Episode(record);path=folder/(record['key']+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as archive:prediction=archive['prediction']
        else:
            nr,n=episode.values.shape[1:3]
            prediction=np.full((len(episode.ends),nr,n,6),np.nan,np.float32)
            eligible=np.argwhere(episode.valid)
            for start in range(0,len(eligible),768):
                batch=eligible[start:start+768];t,r,j=batch.T
                prediction[t,r,j]=predictor.predict(episode,batch)[0]
            temporary=path.with_suffix('.working.npz');np.savez_compressed(temporary,prediction=prediction);temporary.replace(path)
        legacy=episode.valid&episode.legacy_support
        live=legacy&episode.currently_up
        py5=prediction[...,2][live];py37=prediction[...,5][legacy]
        yy5=episode.targets[...,2][live];yy37=episode.targets[...,5][legacy]
        p5.append(py5);y5.append(yy5);p37.append(py37);y37.append(yy37)
        strict=alert_counts(episode.links,episode.ends,prediction[...,5],episode.legacy_support,record['n_search'])
        relaxed=alert_counts(episode.links,episode.ends,prediction[...,5],episode.available,record['n_search'])
        alerts.append(strict);expanded.append(relaxed)
        case=dict(region=record['region'],role=scope,key=record['key'],source_kind=record['source_kind'],
            next5=binary(yy5,py5),any3to7=binary(yy37,py37),alerts=strict,
            predictions_path=str(path),prediction_sha256=sha(path),source_sha256=record['array_sha256'])
        if predictor.meta['family']=='baseline':
            meta=read(record['metadata_path']);request=meta.get('request',{})
            if request.get('checkpoint_sha256')==predictor.meta['sha256']:
                with np.load(record['array_path'],allow_pickle=False) as raw:
                    stored=raw['aux_probabilities'];rawends=raw['aux_endpoints']
                valid_indices={int(e):i for i,e in enumerate(rawends)}
                ix=np.array([valid_indices[int(e)] for e in episode.ends])
                use=np.asarray(episode.valid)
                delta=max(float(np.max(np.abs(prediction[...,2][use]-stored[ix,...,0][use]))),
                          float(np.max(np.abs(prediction[...,5][use]-stored[ix,...,2][use]))))
                if delta>2e-5:raise AssertionError('Stored baseline inference was not reproduced')
                reproduced.append(delta)
        cases.append(case)
        print(json.dumps(dict(stage='evaluate',model=tag,scope=scope,complete=len(cases),total=len(rows),region=record['region'])),flush=True)
    m5=binary(np.concatenate(y5),np.concatenate(p5));m37=binary(np.concatenate(y37),np.concatenate(p37))
    result=dict(model_tag=tag,model_manifest=model,dataset_path=str(dataset.resolve()),dataset_sha256=sha(dataset),scope=scope,cases=cases,regions=len(set(r['region'] for r in rows)),
        next5_on_common_currently_up_links=m5,any3to7_on_common_support=m37,
        strict_alarms_at_half=combine_alerts(alerts),expanded_support_alarms_at_half=combine_alerts(expanded),
        selection_score=(m5['ap']+m37['ap'])/2,baseline_reproduction_max_abs=max(reproduced) if reproduced else None,
        wall_seconds=time.monotonic()-started,created_at=now(),independent_final_test=False,
        target_note='Common 5s truth equals old next-event truth on currently-up links. Any 3-7s truth includes repeated events; old scores are re-evaluated on exactly that truth.',
        prediction_source_policy='Frozen v7 step64 or historical fixed archive policy; the new predictor has no influence on trajectories')
    write(folder/'summary.json',result)
    return folder/'summary.json'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='baseline');p.add_argument('--scope',default='selection',choices=['selection','calibration','confirmation'])
    p.add_argument('--device',default='cuda');p.add_argument('--dataset');a=p.parse_args()
    print(evaluate(a.model,a.scope,a.device,a.dataset))
