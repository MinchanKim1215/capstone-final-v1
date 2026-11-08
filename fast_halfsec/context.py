"""Bind the new immutable map registry and original survey/PPO dependencies."""
from functools import lru_cache
import os

from . import ARTIFACTS, LEGACY

os.environ['NUMBA_CACHE_DIR'] = str(ARTIFACTS / 'numba_cache')

import numpy as np
import torch
from adaptive_joint_expanded import baseline, contracts, map_features, world
from adaptive_joint_expanded.baseline import ensure_nominal
from forecast_prevention import runner as legacy
from forecast_rebuild.contracts import read, sha
from swarm_sim.config import Scenario
from usable.collect import prepare_city_without_unused_radio_cache

from .import_maps import REGISTRY
from .navigation import make_survey

READY = False


@lru_cache(maxsize=1)
def study_plan():
    return read(REGISTRY)


@lru_cache(maxsize=1)
def registry():
    return {entry['key']: entry for entry in study_plan()['maps']}


def initialize():
    global READY
    if READY:
        return
    os.environ['NUMBA_CACHE_DIR'] = str(ARTIFACTS / 'numba_cache')
    torch.set_num_threads(1)
    contracts.plan = study_plan
    baseline.ARTIFACTS = ARTIFACTS
    map_features.ARTIFACTS = ARTIFACTS
    world.ARTIFACTS = ARTIFACTS
    legacy.initialize()
    legacy.MODEL.requires_grad_(False)
    READY = True


@lru_cache(maxsize=2)
def survey_for(region, direction):
    initialize()
    entry = registry()[region]
    if entry['study_role'] not in ('train', 'calibration', 'confirmation'):
        raise ValueError(f'Map is outside the active study roles: {region}')
    path = LEGACY / entry['path']
    if sha(path) != entry['sha256']:
        raise ValueError(f'Registered scene changed: {region}')
    scenario = Scenario(direction=direction, motion_dt_s=3.).validate()
    city = prepare_city_without_unused_radio_cache(path, scenario, ARTIFACTS / 'cache/city')
    city.radio.enable_cache()
    fleet = legacy.deployment_for(dict(region=region, direction=direction))
    plan = ensure_nominal(city, fleet, scenario)
    with np.load(plan, allow_pickle=False) as archive:
        survey = make_survey(city, fleet.ns, archive['nodes'])
    anchor = ensure_nominal(city, contracts.DEPLOYMENT_BY_KEY['v8_s3_r5_40'], scenario)
    with np.load(anchor, allow_pickle=False) as archive:
        anchor_s = (len(archive['nodes']) - 1) * 3.
    return city, scenario, fleet, survey, anchor_s


def interference(spec, city, anchor_s):
    rng = np.random.default_rng(spec['seed'])
    location = city.target_points[int(rng.integers(len(city.target_points)))]
    if spec['kind'] == 'clear':
        begin = end = power = 0.
    elif spec['kind'] == 'continuous':
        begin, end, power = 0., 1e9, float(rng.uniform(-3, 3))
    elif spec['kind'] == 'intermittent':
        begin = float(rng.uniform(.15, .45) * anchor_s)
        end = float(rng.uniform(.65, .9) * anchor_s)
        power = float(rng.uniform(-3, 3))
    else:
        raise ValueError('Unknown interference condition')
    return np.r_[location, 2.], begin, end, power
