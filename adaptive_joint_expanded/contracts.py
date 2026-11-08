"""Experiment boundaries, public deployment constraints, atomic persistence."""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import random
import tempfile
import numpy as np
from . import ARTIFACTS, LEGACY, ROOT, VERSION


@dataclass(frozen=True)
class Deployment:
    key: str
    n: int
    ns: int
    formation: str
    spacing: float

    @property
    def nr(self):
        return self.n - self.ns

    def design(self):
        from swarm_sim.config import Design
        return Design(self.n, self.nr, self.formation, self.spacing).validate()


DEPLOYMENTS = (
    Deployment('v8_s3_r5_40', 8, 3, 'v', 40.),
    Deployment('cross6_s4_r2_20', 6, 4, 'line_cross', 20.),
    Deployment('cross10_s6_r4_20', 10, 6, 'line_cross', 20.),
    Deployment('v6_s4_r2_40', 6, 4, 'v', 40.),
)
DEPLOYMENT_BY_KEY = {d.key: d for d in DEPLOYMENTS}
HISTORY = 40
PUBLIC_CONTEXT_DIM = 140
MAX_SECONDS = 7200
PREDICTION_SECONDS = 1
CONTROL_SECONDS = 3
PILOT = ROOT.parent / 'prediction_improvement/runs/next_outage_1s_v1/best.pt'
PLAN = ROOT.parent / 'additional_maps_20261030/expanded_plan.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def lock(path):
    import msvcrt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError('This run is already locked by another process') from error
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def rng_state():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    import torch
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value['cuda'])


def deployment_mask(available=10, minimum_search=3):
    if not isinstance(available, int) or not isinstance(minimum_search, int):
        raise ValueError('Public fleet constraints must be integers')
    result = np.array([d.n <= available and d.ns >= minimum_search for d in DEPLOYMENTS], bool)
    if not result.any():
        raise ValueError('No deployment meets the available fleet and minimum search count')
    return result


def plan():
    value = read_json(PLAN)
    return value


def entry_for(region, allowed=('train', 'validation', 'historical_validation')):
    item = next((m for m in plan()['maps'] if m['key'] == region), None)
    permitted = item is not None and item['joint_split'] in allowed
    if item is not None and item['joint_split']=='test' and os.environ.get('ADAPTIVE_FINAL_TEST_FREEZE'):
        freeze=Path(os.environ['ADAPTIVE_FINAL_TEST_FREEZE'])
        marker=read_json(ARTIFACTS/'test_access.json')
        permitted=sha(freeze)==marker['freeze_sha256'] and read_json(freeze)['protocol_sha256']==marker['protocol_sha256']
    if not permitted:
        raise ValueError('Unknown or disallowed map split: ' + region)
    if sha(LEGACY / item['path']) != item['sha256']:
        raise ValueError('Map content differs from the original pinned split')
    return item


def public_context(entry, direction, available=10, minimum_search=3):
    """Static public map descriptors. No map ID, seed, split, jammer, future or outcomes."""
    if direction not in ('east_west', 'north_south'):
        raise ValueError('Unknown direction')
    from .map_features import descriptors
    values=descriptors(entry)
    n = max(1, values['buildings'])
    basic=[values['buildings']/1000, values['tall_buildings']/n,
                     values['no_fly_area_fraction'], values['nonimputed_heights']/n,
                     values['imputed_heights']/n, direction == 'east_west',
                     direction == 'north_south', available/10, minimum_search/6]
    context=np.r_[basic,values['height_statistics_scaled'],values['nofly_grid_8x8'],values['footprint_grid_8x8']].astype(np.float32)
    if context.shape!=(PUBLIC_CONTEXT_DIM,) or not np.isfinite(context).all():raise ValueError('Invalid public static-map context')
    return context


def legacy_hashes():
    result = {}
    for path in sorted(LEGACY.rglob('*.py')):
        relative = path.relative_to(LEGACY)
        if any(part in ('.venv', '__pycache__', 'cache', 'results') for part in relative.parts):
            continue
        result['legacy/' + relative.as_posix()] = sha(path)
    for path in sorted((LEGACY/'configs').glob('*.json')):
        result['legacy/' + path.relative_to(LEGACY).as_posix()] = sha(path)
    for path in (PILOT, LEGACY/'results/transformer/train/transformer_main/checkpoints/s0000176.pt',
                 LEGACY/'results/transformer/reference/physics.pt'):
        result[str(path.relative_to(ROOT.parent)).replace('\\', '/')] = sha(path)
    return result


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob('*.py'))
            if 'artifacts' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts}


def config_hash(config):
    # A request for additional updates is a budget extension, not a new experiment.
    return digest({k:v for k,v in config.items() if k not in ('updates',)})


DEFAULT = dict(seed=2026103001, updates=256, batch_episodes=8, epochs=4, minibatch=16,
               actor_lr=1e-4, encoder_lr=1e-5, timing_lr=5e-5, critic_lr=3e-4,
               selector_lr=5e-5, prediction_coef=.25, selector_coef=.2,
               entropy_coef=.003, anchor_coef=.003, gamma=.995, gae_lambda=.95,
               clip=.15, target_kl=.02, available_drones=10, minimum_search=3,
               fleet_cost_per_step=.002, validation_every=16, patience=8,
               warmup_updates=16, validation_cases=8, device='cuda', workers=4,
               mode='adaptive', keep_rollouts=2)
DEFAULT.update(minimum_pdr=.90, maximum_duration_ratio=1.15)
