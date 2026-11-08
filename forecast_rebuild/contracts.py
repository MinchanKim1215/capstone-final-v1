"""Study registry, source pins and spatial data boundaries."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import tempfile
import time
from . import ROOT, WORKSPACE, ARTIFACTS, VERSION

PLAN = WORKSPACE / 'additional_maps_20261030/expanded_plan.json'
PARENT = WORKSPACE / 'adaptive_joint_expanded'
PARENT_RUN = PARENT / 'artifacts/runs/adaptive_v7_expanded100_20261031'
BASELINE = PARENT_RUN / 'checkpoints/s0000064.pt'
SEED = 2026110201
GOALS = dict(alert_precision=.90,event_recall_3s=.80,false_or_late_per_drone_hour=1.,
             minimum_alerts=50,minimum_lead_s=3.,maximum_alert_horizon_s=7.)


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):value.update(part)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(handle,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        for attempt in range(20):
            try:
                os.replace(temp,path);break
            except PermissionError:
                if attempt==19:raise
                # OneDrive/Windows readers may briefly deny delete-sharing.
                # Keep the old complete JSON while retrying atomic replacement.
                time.sleep(.05*(attempt+1))
    finally:
        if os.path.exists(temp):os.unlink(temp)


def maps():
    return {m['key']:m for m in read(PLAN)['maps']}


def folds():
    # Keep the previous geography boundaries; do not redefine them after results.
    from adaptive_joint_expanded.experiment import development_splits
    return development_splits()


def role(region):
    item=maps().get(region)
    if item is None:raise ValueError('Unregistered geography')
    if item['joint_split']=='train':return 'train'
    if item['joint_split']=='validation':
        return next(key for key,regions in folds().items() if region in regions)
    raise ValueError('Protected/test/historical/quarantined geography is forbidden: '+region)


def verify_prior():
    summaries=[]
    for name in ('adaptive_joint','adaptive_joint_expanded'):
        folder=WORKSPACE/name;p=read(folder/'artifacts/protocol.json')
        for relative,expected in p['source_sha256'].items():
            if sha(folder/relative)!=expected:raise ValueError('Frozen prior source changed: '+relative)
        if p['final_test_opened'] or (folder/'artifacts/test_access.json').exists():
            raise ValueError('Previously sealed final test has been opened')
        summaries.append(dict(package=name,protocol_sha256=digest(p),source_files=len(p['source_sha256'])))
    expected=read(PARENT_RUN/'best.json')
    if sha(BASELINE)!=expected['sha256']:raise ValueError('Selected old baseline is not intact')
    return summaries


def prepare():
    pins=verify_prior();registry=maps();partition=folds()
    value=dict(version=VERSION,created_at=now(),seed=SEED,plan_sha256=sha(PLAN),
        prior_pins=pins,baseline=dict(path=str(BASELINE),sha256=sha(BASELINE),step=64),
        train_regions=sorted(k for k,v in registry.items() if v['joint_split']=='train'),
        development_folds=partition,sealed_test_regions=[k for k,v in registry.items() if v['joint_split']=='test'],
        goals=GOALS,goals_status='Development targets, not professor specifications or certified real-world requirements',
        forecast='Separate prediction supervision; no PPO gradient into this predictor',
        targets='Any persistent onset in explicit future windows, including recovery/re-outage; next-event metrics retained for old comparison',
        input_contract='Only received RF/intent history, public GIS, optional simulated idle-channel energy and its reception timestamps. No true current/future link state, jammer coordinates/schedule/seed or actual future positions.',
        timing=dict(radio_s=.5,prediction_s=1,history_frames=40,maximum_horizon_s=8),
        simulator='Frozen v7 policy/physics for primary comparisons. New measurements are input-only; do not make the channel artificially easier.',
        hardware_selected=False,final_test_opened=False,
        selection='Model ranking from selection fold only; calibration and alarm thresholds from calibration only; confirmation once after freeze',
        limitations=['Original warm predictor was trained with eight historically used validation flights; old baseline is not an unseen-data estimator there.',
                     'Some older geography/outcomes have historical exposure. Sealed 32 maps remain unused.',
                     'Real hardware radio, position and antenna calibration remain required.'])
    path=ARTIFACTS/'protocol.json'
    if path.exists():
        old=read(path)
        for key in ('version','seed','plan_sha256','baseline','train_regions','development_folds','goals','input_contract'):
            if old[key]!=value[key]:raise ValueError('Existing study boundary changed: '+key)
        return old
    write(path,value)
    return value
