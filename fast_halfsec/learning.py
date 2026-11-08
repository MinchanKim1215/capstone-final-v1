"""Train causal forecasting heads and freeze event-level operating points."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from forecast_rebuild.contracts import now, read, sha, write
from . import ARTIFACTS
from .features import FEATURE_NAMES, N_FEATURES
from .protocol import ALERT_CAP, PLAN, SEED

HEADS = dict(early_current=(3.,7.,N_FEATURES-8), early_history=(3.,7.,N_FEATURES),
             short_history=(.5,3.,N_FEATURES))
COARSE_PROFILE = ARTIFACTS / 'models/forecast_profile.json'
PROFILE = ARTIFACTS / 'models/cutoff_v2/forecast_profile.json'
ROUNDS = 160


def flight(row, archive_path=None):
    path = archive_path or row['archive_path']
    if sha(path) != row['archive_sha256']:
        raise ValueError('Forecast inputs changed')
    with np.load(path, allow_pickle=False) as saved:
        times = saved['times'].astype(float)
        inputs = saved['features']
        physics = saved['physics']
        up = saved['network'][np.rint(times*2).astype(int)]
    onsets = np.asarray([a for a,b in row['cuts'] if a > 0.], dtype=float)
    if inputs.shape != (len(times),N_FEATURES) or not np.isfinite(inputs).all():
        raise ValueError('Bad forecast input shape')
    return dict(spec=row['spec'], duration_s=row['duration_s'], times=times, x=inputs,
                physics=physics, up=up, onsets=onsets, completed=row['mission_completed'])


def read_manifest(role):
    manifest = read(ARTIFACTS/(role+'_manifest.json'))
    result = []
    for record in manifest['records']:
        row = read(record['metadata'])
        if row['contract']['scope'] != role or row['archive_sha256'] != record['sha256']:
            raise ValueError('Wrong cohort in supervised manifest')
        result.append(flight(row, record['archive']))
    return result


def truth(case, low, high):
    times, onsets = case['times'], case['onsets']
    return case['up'] & (np.searchsorted(onsets,times+high) > np.searchsorted(onsets,times+low))


def evaluate(flights, scores, threshold, low, high, ranking=False):
    events = alerts = hits = 0
    hours = 0.
    leads, y, p = [], [], []
    for case, score in zip(flights,scores,strict=True):
        hours += case['duration_s']/3600.
        onsets = case['onsets']
        events += len(onsets)
        supported = case['times']+high <= case['duration_s'] + 1e-8
        active = supported & (score >= threshold)
        beginnings = active & (~np.r_[False,active[:-1]] | np.r_[True,np.diff(case['times'])>.50001])
        chosen = np.flatnonzero(beginnings)
        alerts += len(chosen)
        matched = np.zeros(len(onsets),bool)
        for index in chosen:
            if not case['up'][index]:
                continue
            t = case['times'][index]
            start, end = np.searchsorted(onsets,[t+low,t+high])
            for k in range(start,end):
                if not matched[k]:
                    matched[k] = True
                    leads.append(float(onsets[k]-t))
                    break
        hits += int(matched.sum())
        if ranking:
            y.append(truth(case,low,high)[supported])
            p.append(score[supported])
    row = dict(events=events, alerts=alerts, hits=hits, false_or_late=alerts-hits, hours=hours,
               recall=hits/max(1,events), precision=hits/max(1,alerts),
               false_or_late_per_fleet_h=(alerts-hits)/max(1e-9,hours),
               median_lead_s=float(np.median(leads)) if leads else None)
    if ranking:
        labels, estimates = np.concatenate(y), np.concatenate(p)
        count = int(labels.sum())
        row.update(ranking_rows=len(labels), ranking_positive=count,
                   ap=float(average_precision_score(labels,estimates)) if count else None,
                   auroc=float(roc_auc_score(labels,estimates)) if 0<count<len(labels) else None)
        if len(estimates) and estimates.min() >= 0. and estimates.max() <= 1.:
            row['brier'] = float(brier_score_loss(labels,estimates))
    return row


def calibrate(flights, scores, low, high):
    all_scores = np.concatenate(scores)
    quantiles = np.unique(np.r_[np.linspace(0.,.98,99),np.linspace(.98,1.,401)])
    thresholds = np.unique(np.r_[np.quantile(all_scores,quantiles), np.nextafter(all_scores.max(),np.inf)])
    candidates = []
    for cutoff in thresholds:
        result = evaluate(flights,scores,float(cutoff),low,high)
        if result['false_or_late_per_fleet_h'] <= ALERT_CAP:
            candidates.append((float(cutoff),result))
    threshold, _ = max(candidates,key=lambda item:(item[1]['recall'],item[1]['precision'],
                                                   -item[1]['false_or_late_per_fleet_h'],item[0]))
    return dict(threshold=threshold, window=[low,high], tested_thresholds=len(thresholds),
                metrics=evaluate(flights,scores,threshold,low,high,ranking=True))


class Predictors:
    def __init__(self, path=PROFILE):
        self.profile = read(path)
        if self.profile['features'] != list(FEATURE_NAMES):
            raise ValueError('Wrong forecast feature contract')
        self.models = {}
        for name, metadata in self.profile['models'].items():
            if sha(metadata['path']) != metadata['sha256']:
                raise ValueError(f'Changed model: {name}')
            self.models[name] = lgb.Booster(model_str=Path(metadata['path']).read_text(encoding='utf-8'))
        self.threshold = self.profile['calibration']['early_history']['threshold']

    def scores(self, name, values):
        if not len(values):
            return np.empty(0)
        return self.models[name].predict(values[:,:HEADS[name][2]],num_threads=1)

    def predict(self, raw):
        x = raw.reshape(1,-1)
        return float(self.scores('early_history',x)[0]), float(self.scores('short_history',x)[0])


def fit():
    pins = dict(protocol=sha(PLAN), source=sha(__file__),
                train=sha(ARTIFACTS/'train_manifest.json'), calibration=sha(ARTIFACTS/'calibration_manifest.json'))
    if PROFILE.exists():
        predictors = Predictors()
        if predictors.profile['pins'] != pins:
            raise ValueError('Stored predictor uses other code or data')
        return predictors.profile
    train, validation = read_manifest('train'), read_manifest('calibration')
    PROFILE.parent.mkdir(parents=True,exist_ok=True)
    parameters = dict(objective='binary', metric='binary_logloss', learning_rate=.04,
                      num_leaves=19, min_data_in_leaf=96, feature_fraction=.85,
                      bagging_fraction=.9, bagging_freq=1, lambda_l2=2., deterministic=True,
                      force_col_wise=True, seed=SEED%2147483647, num_threads=6, verbosity=-1)
    models, operating = {}, {}
    previous = read(COARSE_PROFILE) if COARSE_PROFILE.exists() else None
    if previous and any(previous['pins'][key] != pins[key] for key in ('protocol','train','calibration')):
        raise ValueError('Original forecast models use other data')
    for name,(low,high,columns) in HEADS.items():
        if previous:
            metadata = previous['models'][name]
            path = Path(metadata['path'])
            if sha(path) != metadata['sha256'] or metadata['dimension'] != columns:
                raise ValueError('Original fitted model changed')
            model = lgb.Booster(model_str=path.read_text(encoding='utf-8'))
        else:
            masks = [case['times']+high <= case['duration_s']+1e-8 for case in train]
            x = np.concatenate([case['x'][mask,:columns] for case,mask in zip(train,masks)])
            y = np.concatenate([truth(case,low,high)[mask] for case,mask in zip(train,masks)])
            if not 0 < int(y.sum()) < len(y):
                raise ValueError(f'Need positive and negative events for {name}')
            model = lgb.train(parameters,lgb.Dataset(x,label=y,feature_name=list(FEATURE_NAMES[:columns])),
                              num_boost_round=ROUNDS)
            path = PROFILE.parent / (name+'.txt')
            temporary = path.with_suffix('.pending')
            temporary.write_text(model.model_to_string(),encoding='utf-8')
            temporary.replace(path)
            metadata = dict(path=str(path),sha256=sha(path),rows=len(y),positive=int(y.sum()),
                            dimension=columns,rounds=ROUNDS)
        scores = [model.predict(case['x'][:,:columns],num_threads=1) if len(case['x']) else np.empty(0)
                  for case in validation]
        operating[name] = calibrate(validation,scores,low,high)
        models[name] = metadata
        print(f"calibrated {name}: {metadata['rows']} rows, {metadata['positive']} positive, "
              f"calibration recall={operating[name]['metrics']['recall']:.3f}",flush=True)
    operating['physics'] = calibrate(validation,[case['physics'] for case in validation],3.,7.)
    profile = dict(created_at=now(),features=list(FEATURE_NAMES),pins=pins,models=models,calibration=operating,
                   parameters=parameters,train_flights=len(train),calibration_flights=len(validation),
                   train_completed=sum(case['completed'] for case in train),
                   calibration_completed=sum(case['completed'] for case in validation),
                   calibration_search='99 bulk quantiles and 401 quantiles from 98% through 100%; fixed before confirmation',
                   original_weight_profile_sha256=sha(COARSE_PROFILE) if previous else None,
                   labels='New persistent onset in the explicit window, with current actual connection used only in scoring and labels')
    write(PROFILE,profile)
    return profile
