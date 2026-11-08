"""Same tree settings on expanded experience; energy/no-energy ablation."""
from pathlib import Path
import argparse
import ctypes
import time
import numpy as np
import lightgbm as lgb
from . import ARTIFACTS
from .contracts import read,write,sha,now,pin,SEED,verify_protocol
from forecast_rebuild.dataset import Samples,without_energy
from forecast_rebuild.features import feature_names
from forecast_rebuild.planning import names as planning_names
from forecast_rebuild.metrics import binary


def train(tag,use_energy=True):
    verify_protocol();folder=ARTIFACTS/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.json').exists():return folder/'model.json'
    sample_folder=ARTIFACTS/'samples/merged_trees';data=Samples(sample_folder,'train');validation=Samples(sample_folder,'selection')
    x=data.x if use_energy else without_energy(data.x);vx=validation.x if use_energy else without_energy(validation.x)
    params=dict(objective='binary',metric='average_precision',learning_rate=.04,num_leaves=31,max_bin=127,min_data_in_leaf=100,
        lambda_l2=5.,feature_fraction=.9,bagging_fraction=.85,bagging_freq=1,num_threads=4,verbosity=-1,
        deterministic=True,force_col_wise=True,seed=SEED,is_unbalance=False)
    names=feature_names()+planning_names();heads={};scores={};clock=time.monotonic()
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        for head in (2,5):
            mask=validation.up&validation.legacy if head==2 else validation.legacy
            trainset=lgb.Dataset(x,label=np.asarray(data.y[:,head]),feature_name=names,free_raw_data=True)
            valset=lgb.Dataset(vx[mask],label=np.asarray(validation.y[mask,head]),reference=trainset,free_raw_data=True)
            model=lgb.train(params,trainset,num_boost_round=800,valid_sets=[valset],callbacks=[lgb.early_stopping(60,first_metric_only=True,verbose=False)])
            path=folder/f'head{head}.txt';path.write_text(model.model_to_string(),encoding='utf-8')
            p=model.predict(vx,num_threads=4);scores[str(head)]=binary(validation.y[mask,head],p[mask])
            heads[str(head)]=dict(path=str(path),sha256=sha(path),iterations=model.best_iteration)
            write(folder/'status.json',dict(state='training',head=head,updated_at=now()))
            del model,trainset,valset
        value=dict(family='lightgbm',tag=tag,use_energy=use_energy,use_plan=True,heads=heads,selection=scores,
            sample_manifest_sha256=sha(sample_folder/'manifest.json'),source_sha256=pin(['train_tree.py']),params=params,
            created_at=now(),wall_seconds=time.monotonic()-clock,calibration_used=False,confirmation_used=False)
        write(folder/'model.json',value);write(folder/'status.json',dict(state='completed',wall_seconds=value['wall_seconds']))
        return folder/'model.json'
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',default='new_data_tree');p.add_argument('--no-energy',action='store_true');a=p.parse_args();print(train(a.tag,not a.no_energy))
