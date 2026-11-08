"""Probability-preserving binary training; no class-weight probability inflation."""
from pathlib import Path
import argparse
import ctypes
import json
import time
import numpy as np
import lightgbm as lgb
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,SEED
from .dataset import Samples,without_energy
from .features import feature_names
from .metrics import binary


def train(samples,tag,use_energy=True,rounds=800):
    folder=ARTIFACTS/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.json').exists():return read(folder/'model.json')
    data=Samples(samples,'train');valid=Samples(samples,'selection')
    use_plan=bool(data.manifest['contract'].get('planning',False))
    names=feature_names()
    if use_plan:
        from .planning import names as planning_names
        names=names+planning_names()
    feature_pins={name:sha(ROOT/name) for name in ('features.py','targets.py','dataset.py','train_lgbm.py')}
    x=data.x if use_energy else without_energy(data.x)
    vx=valid.x if use_energy else without_energy(valid.x)
    params=dict(objective='binary',metric='average_precision',learning_rate=.04,num_leaves=31,
                max_bin=127,min_data_in_leaf=100,lambda_l2=5.,feature_fraction=.9,
                bagging_fraction=.85,bagging_freq=1,num_threads=4,verbosity=-1,
                deterministic=True,force_col_wise=True,seed=SEED,is_unbalance=False)
    started=time.monotonic();models={};scores={}
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        for head in (2,5):
            mask=valid.up&valid.legacy if head==2 else np.ones(len(valid.y),bool)
            trainset=lgb.Dataset(x,label=np.asarray(data.y[:,head]),feature_name=names,free_raw_data=True)
            validation=lgb.Dataset(vx[mask],label=np.asarray(valid.y[mask,head]),reference=trainset,free_raw_data=True)
            write(folder/'status.json',dict(state='training',family='lightgbm',head=head,started_at=now(),samples=len(data.y),
                                           use_simulated_energy=use_energy,system_sleep_inhibited=held))
            booster=lgb.train(params,trainset,num_boost_round=rounds,valid_sets=[validation],valid_names=['selection'],
                callbacks=[lgb.early_stopping(60,first_metric_only=True,verbose=False),lgb.log_evaluation(100)])
            # Native Windows fopen may not support a Korean OneDrive path.
            # Python's Unicode filesystem API preserves the exact same model.
            path=folder/f'head{head}.txt';path.write_text(booster.model_to_string(),encoding='utf-8')
            prediction=booster.predict(vx,num_iteration=booster.best_iteration,num_threads=4)
            scores[str(head)]=binary(valid.y[mask,head],prediction[mask])
            gain=booster.feature_importance(importance_type='gain')
            order=np.argsort(gain)[::-1][:25]
            models[str(head)]=dict(path=str(path),sha256=sha(path),iterations=booster.best_iteration,
                top_features=[dict(name=names[i],gain=float(gain[i])) for i in order])
            print(json.dumps(dict(stage='trained',head=head,selection=scores[str(head)],iterations=booster.best_iteration)),flush=True)
            del booster,trainset,validation
        for name,expected in feature_pins.items():
            if sha(ROOT/name)!=expected:raise ValueError('Training source changed during model fitting')
        result=dict(family='lightgbm',tag=tag,use_energy=use_energy,use_plan=use_plan,heads=models,selection=scores,
                    samples=str(Path(samples).resolve()),sample_manifest_sha256=sha(Path(samples)/'manifest.json'),
                    source_sha256=feature_pins,params=params,wall_seconds=time.monotonic()-started,created_at=now(),
                    selection_score=float(np.mean([scores[str(h)]['ap'] for h in (2,5)])),
                    calibration_used=False,confirmation_used=False,final_test_used=False)
        write(folder/'model.json',result);write(folder/'status.json',dict(state='completed',wall_seconds=result['wall_seconds']))
        return result
    except BaseException as error:
        write(folder/'status.json',dict(state='failed',error=repr(error),updated_at=now()));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--no-energy',action='store_true');p.add_argument('--rounds',type=int,default=800);a=p.parse_args()
    result=train(a.samples,a.tag,not a.no_energy,a.rounds);print(json.dumps(result['selection'],ensure_ascii=False))
