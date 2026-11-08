"""Fit a separate time-of-onset head on the same development-only inputs."""
import lightgbm as lgb
import numpy as np

from . import ARTIFACTS
from .features import COLUMNS
from .protocol import CAP, now, read, sha, write
from .training import PARAMS, load, metrics, ranking, restore_model, select_threshold


EDGES = np.array([.5,3.,4.5,5.5,7.,9.])
ROUNDS = 180


def labels(flight):
    time = flight.raw[:,0]
    onsets = np.r_[[event['onset_s'] for event in flight.row['events']],np.inf]
    next_onset = onsets[np.searchsorted(onsets,time+.5)]
    lead = next_onset-time
    with np.load(flight.row['archive_path'],allow_pickle=False) as a:
        connected = a['network'][(time*2).astype(int)-1]
    target = np.digitize(lead,EDGES)
    target[(target==len(EDGES)) | ~connected] = 0
    use = flight.support & (time+EDGES[-1]<=flight.row['duration_s'])
    return target,use


def scores(model, flights):
    results = []
    for flight in flights:
        prediction = np.full(len(flight.raw),np.nan)
        p = model.predict(flight.x[flight.support],num_threads=1)
        # Lead bins [3,4.5), [4.5,5.5), [5.5,7).
        prediction[flight.support] = p[:,2:5].sum(axis=1)
        results.append(prediction)
    return results


def fit():
    path = ARTIFACTS/'timing_freeze.json'
    if path.exists() or (ARTIFACTS/'final_results.json').exists():
        raise ValueError('Time-of-onset settings are already locked or final results were opened')
    names = ('train','calibration','selection')
    manifests = {key:ARTIFACTS/(key+'_manifest.json') for key in names}
    cohorts = {key:[load(row) for row in read(meta)['records']] for key,meta in manifests.items()}
    inputs,targets = [],[]
    for f in cohorts['train']:
        y,use = labels(f)
        inputs.append(f.x[use])
        targets.append(y[use])
    x,y = np.concatenate(inputs),np.concatenate(targets)
    params = dict(PARAMS,objective='multiclass',metric='multi_logloss',num_class=len(EDGES),num_threads=2)
    data = lgb.Dataset(x,label=y,feature_name=list(COLUMNS))
    model = lgb.train(params,data,num_boost_round=ROUNDS)
    file = ARTIFACTS/'models'/'timing.txt'
    pending = file.with_suffix('.pending.txt')
    pending.write_text(model.model_to_string(),encoding='utf-8')
    pending.replace(file)
    ca = scores(model,cohorts['calibration'])
    sv = scores(model,cohorts['selection'])
    variants = []
    for weight in (1.,.7):
        cal = [weight*s+(1-weight)*f.raw[:,1] for f,s in zip(cohorts['calibration'],ca)]
        dev = [weight*s+(1-weight)*f.raw[:,1] for f,s in zip(cohorts['selection'],sv)]
        for cap in (3.,3.5,CAP):
            threshold, c = select_threshold(cohorts['calibration'],cal,cap)
            d = metrics(cohorts['selection'],dev,threshold)
            variants.append(dict(weight=weight,threshold=threshold,cap=cap,calibration=c,selection=d))
    eligible = [v for v in variants if v['selection']['false_per_hour']<=CAP and v['selection']['alerts']]
    selected = max(eligible or variants,key=lambda v:(v['selection']['recall'],v['selection']['precision'],-v['selection']['false_per_hour']))
    value = dict(selected=selected,eligible_on_selection=bool(eligible),variants=variants,classes=np.bincount(y,minlength=len(EDGES)).tolist(),
                 edges=EDGES.tolist(),rounds=ROUNDS,model=dict(path=str(file),sha256=sha(file)),
                 source_sha256=sha(__file__),manifests={k:sha(v) for k,v in manifests.items()},created_at=now())
    write(path,value)
    print('Locked onset bins',value['classes'],'selection',selected['selection'],flush=True)


def result(flights):
    path = ARTIFACTS/'timing_freeze.json'
    freeze = read(path)
    if freeze['source_sha256']!=sha(__file__) or freeze['model']['sha256']!=sha(freeze['model']['path']):
        raise ValueError('Timing model changed after selection')
    for key,expected in freeze['manifests'].items():
        if sha(ARTIFACTS/(key+'_manifest.json'))!=expected:
            raise ValueError('Timing model used different data')
    choice = freeze['selected']
    p = scores(restore_model(freeze['model']),flights)
    w = choice['weight']
    p = [w*s+(1-w)*f.raw[:,1] for f,s in zip(flights,p)]
    return dict(**metrics(flights,p,choice['threshold'],True),ranking=ranking(flights,p),
                threshold=choice['threshold'],weight=w,training_classes=freeze['classes'],
                freeze_sha256=sha(path),created_at=now())


if __name__=='__main__':
    fit()
