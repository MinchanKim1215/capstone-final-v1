from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,os,random,time,uuid
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='standalone-transformer-rf-v1'
HISTORY=40
HEADS=('risk_5s','onset_3to4s','onset_3to7s','onset_0to7s')
SOURCE_NAMES=['solo/contracts.py','solo/features.py','solo/model.py','solo/world.py','solo/rollout.py','solo/data.py','solo/training.py','solo/targets.py','transformer_run.py','solo/__init__.py','solo/deploy.py']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('w',encoding='utf-8') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def atomic_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def sources():
    old=json.loads((ROOT/'docs/previous_published_selection.json').read_text())['source_sha256']
    for name,value in old.items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen v3.1 source changed: '+name)
    names=set(old)|set(SOURCE_NAMES)
    for directory in ['cooperative','swarm_sim','usable','advance','upgrade']:
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*.py'))
    return {name:sha(ROOT/name) for name in sorted(names)}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'][:torch.cuda.device_count()])


def device_for(name='auto'):
    if name=='auto':name='cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable. Install a supported CUDA PyTorch build or use --device cpu.')
    if name not in ('cpu','cuda') and not name.startswith('cuda:'):raise ValueError('Use cpu, cuda, cuda:N or auto. MPS is not validated; use CPU on macOS.')
    return torch.device(name)


def deterministic(enabled=True):
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=enabled
    torch.use_deterministic_algorithms(enabled)


@contextmanager
def file_lock(path,timeout=0,stop=None):
    from cooperative.locking import TrainingLock
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();lock=None
    while lock is None:
        if stop is not None and stop.is_set():raise InterruptedError('Stop requested')
        try:lock=TrainingLock(path)
        except RuntimeError:
            if time.monotonic()-start>=timeout:raise RuntimeError('Another process owns '+str(path)) from None
            time.sleep(.1)
    try:yield
    finally:lock.close()


def load_plan():
    p=ROOT/'configs/transformer_plan.json'
    if not p.exists():raise ValueError('Run: python transformer_run.py doctor')
    data=json.loads(p.read_text(encoding='utf-8'))
    if data.get('schema')!=SCHEMA:raise ValueError('Dataset schema changed; use a new study')
    for name,value in data['baseline_files'].items():
        if sha(ROOT/name)!=value:raise ValueError('Baseline model/source changed: '+name)
    return data


def ensure_map(entry,stop=None):
    """One map lock covers shared visibility plus both direction caches on Windows and POSIX."""
    from swarm_sim.config import Scenario,Design
    from usable.collect import prepare_city_without_unused_radio_cache
    from usable.motion import full_mission
    cache=ROOT/'cache/usable';cache.mkdir(parents=True,exist_ok=True)
    path=ROOT/entry['path']
    if sha(path)!=entry['sha256']:raise ValueError('Map bytes changed: '+entry['key'])
    signature=object_hash(dict(map_sha=entry['sha256'],source={n:sha(ROOT/n) for n in
        ['swarm_sim/city.py','swarm_sim/config.py','swarm_sim/motion.py','usable/motion.py','usable/collect.py']}))
    marker=cache/(entry['key']+'_joint_cache.json')
    def valid():
        if not marker.exists():return False
        data=json.loads(marker.read_text())
        return data.get('signature')==signature and all((cache/n).exists() and sha(cache/n)==h for n,h in data.get('files',{}).items())
    if valid():return
    with file_lock(cache/(entry['key']+'.lock'),timeout=600,stop=stop):
        if valid():return
        files={}
        for direction in ['east_west','north_south']:
            if stop is not None and stop.is_set():raise InterruptedError('Stop requested')
            scenario=Scenario(duration_s=7200,motion_dt_s=3,direction=direction).validate()
            city=prepare_city_without_unused_radio_cache(path,scenario,cache)
            result=full_mission(city,Design(8,5,'v',40).validate(),scenario)
            if not result['mission_completed']:raise ValueError('Nominal full mission incomplete: '+entry['key']+' '+direction)
            output=cache/f'{entry["key"]}_{direction}_motion.npz'
            atomic_npz(output,**{k:result[k] for k in ['nodes','desired','phase']})
            atomic_json(output.with_suffix('.json'),{k:v for k,v in result.items() if k not in ['nodes','desired','phase']})
            files[output.name]=sha(output)
            files[output.with_suffix('.json').name]=sha(output.with_suffix('.json'))
        atomic_json(marker,dict(signature=signature,files=files))
