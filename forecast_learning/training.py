"""Fit early/short outage heads and lock alert operating points on development data."""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import ARTIFACTS, ROOT
from .features import COLUMNS, HELD_COLUMNS
from .protocol import CAP, V20_THRESHOLD, now, read, sha, status, write
from forecast_shield.evaluate import summarize


ITERATIONS = (160,420)
BUDGETS = (3.,3.5,4.)
PARAMS = dict(objective='binary',metric='binary_logloss',learning_rate=.04,num_leaves=19,
              min_data_in_leaf=64,max_bin=127,feature_fraction=.85,bagging_fraction=.9,
              bagging_freq=1,lambda_l2=2.,num_threads=5,deterministic=True,force_col_wise=True,
              seed=2026110521,verbosity=-1)


@dataclass
class Flight:
    row: dict
    x: np.ndarray
    raw: np.ndarray
    early: np.ndarray
    short: np.ndarray
    usable: np.ndarray
    support: np.ndarray


def load(record):
    if sha(record['path'])!=record['sha256']:
        raise ValueError('Flight metadata changed')
    row = read(record['path'])
    path = row['archive_path']
    if sha(path)!=row['archive_sha256']:
        raise ValueError('Flight evidence changed')
    with np.load(path,allow_pickle=False) as a:
        x, raw, network = a['x'], a['scores'], a['network']
    times = raw[:,0]
    onsets = np.asarray([e['onset_s'] for e in row['events']])
    up = network[(times*2).astype(int)-1]
    early = up & (np.searchsorted(onsets,times+7.)>np.searchsorted(onsets,times+3.))
    short = up & (np.searchsorted(onsets,times+3.)>np.searchsorted(onsets,times+.5))
    support = np.isfinite(raw[:,1])
    usable = support & (times+7.<=row['duration_s'])
    if x.shape!=(len(raw),len(COLUMNS)) or np.isinf(x).any():
        raise ValueError('Unexpected predictor input shape')
    return Flight(row,x,raw,early,short,usable,support)


def metrics(flights, scores, threshold, detail=False):
    return summarize([(f.row,dict(time_s=f.raw[:,0],shield=s)) for f,s in zip(flights,scores)],
                     'shield_3s',threshold,detail)


def ranking(flights, scores, label='early'):
    y = np.concatenate([getattr(f,label)[f.usable] for f in flights])
    p = np.concatenate([s[f.usable] for f,s in zip(flights,scores)])
    return dict(ap=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)),
                rows=len(y),positives=int(y.sum()))


def select_threshold(flights, scores, cap):
    finite = np.concatenate([v[np.isfinite(v)] for v in scores])
    points = np.unique(np.r_[1.01,np.quantile(finite,np.linspace(.25,1.,101))])
    feasible = []
    for threshold in points:
        stat = metrics(flights,scores,float(threshold))
        if stat['false_per_hour']<=cap+1e-9:
            feasible.append((float(threshold),stat))
    return max(feasible,key=lambda p:(p[1]['recall'],p[1]['precision'],-p[1]['false_per_hour'],p[0]))


def features(kind):
    return HELD_COLUMNS if kind=='held_early' else COLUMNS


def train_model(kind, x, y):
    cols = features(kind)
    data = lgb.Dataset(x[:,:len(cols)],label=y,feature_name=list(cols))
    model = lgb.train(PARAMS,data,num_boost_round=ITERATIONS[-1])
    path = ARTIFACTS/'models'/(kind+'.txt')
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_suffix('.pending.txt')
    # Python handles the Korean workspace path consistently on Windows.
    pending.write_text(model.model_to_string(),encoding='utf-8')
    pending.replace(path)
    return model,dict(path=str(path),sha256=sha(path),dimension=len(cols),rounds=ITERATIONS[-1],positives=int(y.sum()))


def restore_model(metadata):
    from pathlib import Path
    return lgb.Booster(model_str=Path(metadata['path']).read_text(encoding='utf-8'))


def make_score(model, profile, x, v20):
    weights = profile['learned_weight']
    if not weights:
        return v20
    dim = profile['dimension']
    p = model.predict(x[...,:dim],num_iteration=profile['rounds'],num_threads=1)
    return weights*p + (1.-weights)*v20


def scores_for(model, profile, flights):
    scores = []
    for f in flights:
        values = np.full(len(f.raw),np.nan)
        values[f.support] = make_score(model,profile,f.x[f.support],f.raw[f.support,1])
        scores.append(values)
    return scores


def verify_freeze(freeze):
    if freeze['protocol_sha256']!=sha(ARTIFACTS/'protocol.json'):
        raise ValueError('Fitted model uses another data protocol')
    for name,expected in freeze['sources'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('Locked predictor/control source changed: '+name)
    for name,expected in freeze['manifests'].items():
        if sha(ARTIFACTS/(name+'_manifest.json'))!=expected:
            raise ValueError('Locked learning cohort changed: '+name)
    for metadata in freeze['models'].values():
        if sha(metadata['path'])!=metadata['sha256']:
            raise ValueError('Locked model file changed')


def fit_and_lock(development):
    path = ARTIFACTS/'freeze.json'
    if path.exists():
        value = read(path)
        verify_freeze(value)
        return value
    from .control import GUARDS
    status('loading_training_examples')
    train = [load(row) for row in development['train']]
    x = np.concatenate([f.x[f.usable] for f in train])
    early = np.concatenate([f.early[f.usable] for f in train])
    short = np.concatenate([f.short[f.usable] for f in train])
    calibration = [load(row) for row in development['calibration']]
    selection = [load(row) for row in development['selection']]
    stats = dict(maps=len(train),rows=len(x),early_positive=int(early.sum()),short_positive=int(short.sum()),
                 persistent_onsets=sum(len(f.row['events']) for f in train),
                 hours=sum(f.row['duration_s'] for f in train)/3600.,held_dimension=len(HELD_COLUMNS),dual_dimension=len(COLUMNS))
    del train
    models = {}
    trained = {}
    for i,(name,y) in enumerate([('held_early',early),('dual_early',early),('dual_short',short)]):
        status('training_lightgbm',current=name,complete=i,total=3,iterations=ITERATIONS[-1],**stats)
        trained[name],models[name] = train_model(name,x,y)
        print('Trained',name,ITERATIONS[-1],flush=True)
    baseline = dict(calibration=metrics(calibration,[f.raw[:,1] for f in calibration],V20_THRESHOLD),
                    selection=metrics(selection,[f.raw[:,1] for f in selection],V20_THRESHOLD),
                    selection_ranking=ranking(selection,[f.raw[:,1] for f in selection]))
    status('calibrating_alert_budget')
    choices = []
    for name in ('held_early','dual_early'):
        for iteration in ITERATIONS:
            for weight in (1.,.7) if name=='dual_early' else (1.,):
                profile = dict(model=name,dimension=models[name]['dimension'],rounds=iteration,learned_weight=weight)
                p = scores_for(trained[name],profile,calibration)
                q = scores_for(trained[name],profile,selection)
                rank = ranking(selection,q)
                for cap in BUDGETS:
                    threshold, cal = select_threshold(calibration,p,cap)
                    dev = metrics(selection,q,threshold)
                    choices.append(dict(profile,threshold=threshold,calibration_cap=cap,calibration=cal,selection=dev,ranking=rank))
    feasible = [v for v in choices if v['selection']['false_per_hour']<=CAP and v['selection']['alerts']>0]
    ranked = feasible or choices
    picked = max(ranked,key=lambda v:(v['selection']['recall'],v['selection']['precision'],-v['selection']['false_per_hour']))
    old = baseline['selection']
    new = picked['selection']
    eligible_for_control = bool(feasible and new['recall']>=old['recall'] and new['precision']>=old['precision']-.05)
    short_profile = dict(dimension=len(COLUMNS),rounds=ITERATIONS[-1],learned_weight=1.)
    timing = ranking(selection,scores_for(trained['dual_short'],short_profile,selection),'short')
    value = dict(chosen=picked,control_forecast='learned' if eligible_for_control else 'v20',
                 control_guards=GUARDS,choices=choices,models=models,training=stats,short_head_selection=timing,
                 baseline=baseline,parameters=PARAMS,protocol_sha256=sha(ARTIFACTS/'protocol.json'),
                 sources={k:sha(ROOT/k) for k in ('training.py','control.py')},
                 manifests={k:sha(ARTIFACTS/(k+'_manifest.json')) for k in development},created_at=now())
    write(path,value)
    verify_freeze(value)
    return value
