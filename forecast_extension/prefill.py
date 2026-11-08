"""Compute fixed-model prediction caches independently and publish atomically.

The comparison controller alone computes statistics and chooses a model. This
worker only avoids serial GPU idle periods, using identical query batches.
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,digest,now,verify_protocol
from .evaluate import model_pin,verify_freeze
from .feature_cache import CachedEpisode
from forecast_rebuild.inference import Predictor


def main(model,scope='selection'):
    if scope not in ('selection','calibration','confirmation'):raise ValueError('Unknown comparison scope')
    worker_sha=sha(__file__)
    verify_protocol()
    if scope=='confirmation':
        frozen=verify_freeze();normalized='baseline' if model=='baseline' else str(Path(model).resolve())
        if normalized not in {item['pin']['path'] for item in frozen['models']}:raise ValueError('Confirmation model is not frozen')
    dataset=ARTIFACTS/('dataset_with_confirmation.json' if scope=='confirmation' else 'dataset.json');corpus=read(dataset)
    rows=[r for r in corpus['rows'] if r['role']==scope]
    pinned=model_pin(model);predictor=Predictor(model,'cuda');tag=predictor.meta['tag']
    implementation=dict(evaluate=sha(ROOT/'evaluate.py'),feature_cache=sha(ROOT/'feature_cache.py'),metrics=sha(ROOT/'metrics.py'))
    identity=digest(dict(model=pinned,input=[(r['key'],r['array_sha256'],sha(Path(r['cache_path'])/'manifest.json')) for r in rows],
                         implementation=implementation))[:16]
    folder=ARTIFACTS/'evaluation'/tag/identity/scope;folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();computed=skipped=0
    for record in rows:
        path=folder/(record['key']+'.npz')
        if path.exists():skipped+=1;continue
        episode=CachedEpisode(record)
        # The prior tree has committed these caches before parallel workers.
        # Refuse to race a static-feature producer.
        if not (ARTIFACTS/'static_features'/record['key']/'manifest.json').exists():
            raise ValueError('Static feature cache has not been committed')
        prediction=np.full((*episode.valid.shape,6),np.nan,np.float32);eligible=np.argwhere(episode.valid)
        for start in range(0,len(eligible),768):
            batch=eligible[start:start+768];t,r,j=batch.T
            prediction[t,r,j]=predictor.predict(episode,batch)[0]
        temporary=folder/(record['key']+'.prefill.'+str(os.getpid())+'.npz')
        np.savez_compressed(temporary,prediction=prediction)
        # The main controller can reach a case while this worker calculates.
        # Never read or overwrite its in-progress file; both use the same pin.
        if path.exists():temporary.unlink();skipped+=1
        else:
            try:os.replace(temporary,path);computed+=1
            except PermissionError:
                if not path.exists():raise
                temporary.unlink();skipped+=1
        write(ARTIFACTS/('prefill_'+tag+'_'+scope+'_status.json'),dict(state='running',model=tag,scope=scope,computed=computed,skipped=skipped,total=len(rows),updated_at=now()))
    verify_protocol()
    if scope=='confirmation':verify_freeze()
    if model_pin(model)!=pinned:raise ValueError('Model pin changed during cache preparation')
    if sha(__file__)!=worker_sha:raise ValueError('Worker implementation changed during cache preparation')
    write(ARTIFACTS/('prefill_'+tag+'_'+scope+'_status.json'),dict(state='completed',model=tag,scope=scope,model_pin=pinned,computed=computed,skipped=skipped,total=len(rows),
        folder=str(folder),dataset_sha256=sha(dataset),implementation=implementation,worker_sha256=worker_sha,
        updated_at=now(),wall_seconds=time.monotonic()-started,statistics_or_selection_performed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,nargs='+')
    parser.add_argument('--scope',choices=['selection','calibration','confirmation'],default='selection');args=parser.parse_args()
    for model in args.model:main(model,args.scope)
