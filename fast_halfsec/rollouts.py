"""Resumable, checksum-verified whole-flight collection."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import multiprocessing as mp
import os

import numpy as np

from forecast_rebuild.contracts import read, sha, write
from . import ARTIFACTS
from .context import initialize, registry
from .protocol import PLAN, status
from .runner import full_mission, load_actor


def job(scope, spec, mode, forecast=None, actor=None, collect=False, explore=False):
    return dict(scope=scope, spec=spec, mode=mode, collect=collect, explore=explore,
                protocol_sha256=sha(PLAN), forecast=forecast, actor=actor)


def native(spec):
    from final_v1.runner import run
    row, arrays = run(spec, 'v21')
    result = dict(spec=spec, mode='final_v1_3s', deployment=row['deployment'],
                  duration_s=row['duration_s'], mission_completed=row['mission_completed'],
                  onsets=row['persistent_onsets'], cuts=row['persistent_cut_intervals'],
                  cut_seconds=row['persistent_cut_seconds'], network_fraction=row['search_connected_mean'],
                  generated=row['generated_packets'], timely=int(round(row['timely_pdr']*row['generated_packets'])),
                  timely_pdr=row['timely_pdr'], delivered=row['delivered_packets'],
                  relay_travel_m=row['relay_travel_m'], search_travel_m=row['search_travel_m'],
                  safety_passed=bool(row['audited']['passed']), packet_conserved=bool(row['audited']['packet_conserved']),
                  coverage=row['observable_coverage'], min_separation_m=row['audited']['minimum_swept_separation_m'],
                  planner_p95_ms=row['simulator_step_p95_ms'], wall_s=row['wall_s'],
                  prevention_requests=sum(event.get('actual_pre_action_connected',False) for event in row['intervention']),
                  events=row['intervention'])
    return result, {key:arrays[key] for key in ('network','links','positions','commands','requested')}


@lru_cache(maxsize=2)
def forecast_at(path, expected):
    from .learning import Predictors
    if sha(path) != expected:
        raise ValueError('Forecast artifact changed')
    return Predictors(path)


@lru_cache(maxsize=2)
def actor_at(path, expected):
    if sha(path) != expected:
        raise ValueError('Actor checkpoint changed')
    return load_actor(path)


def record_paths(request):
    spec = request['spec']
    folder = ARTIFACTS / 'rollouts' / request['scope'] / request['mode']
    prefix = folder / f"{spec['region']}_{spec['kind']}_{spec['seed']}"
    return prefix.with_suffix('.json'), prefix.with_suffix('.npz')


def cached(request):
    meta, archive = record_paths(request)
    if not (meta.exists() and archive.exists()):
        return None
    row = read(meta)
    if row['contract'] != request or sha(archive) != row['archive_sha256']:
        raise ValueError(f'Rollout contract or archive changed: {meta.name}')
    return row


def one(request):
    previous = cached(request)
    if previous is not None:
        return previous
    scope, spec = request['scope'], request['spec']
    role = registry()[spec['region']]['study_role']
    expected = 'train' if scope.startswith('ppo_') else scope
    if role != expected or request['protocol_sha256'] != sha(PLAN):
        raise ValueError('Rollout geography or protocol mismatch')
    if scope == 'confirmation' and not (ARTIFACTS/'confirmation_open.json').exists():
        raise ValueError('Confirmation is still sealed')
    initialize()
    if request['mode'] == 'final_v1_3s':
        row, arrays = native(spec)
    else:
        f, a = request['forecast'], request['actor']
        forecast = forecast_at(f['path'],f['sha256']) if f else None
        policy = actor_at(a['path'],a['sha256']) if a else None
        row, arrays = full_mission(spec, request['mode'], forecast, policy, request['explore'], request['collect'])
    meta, archive = record_paths(request)
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix('.pending.npz')
    np.savez_compressed(pending, **arrays)
    os.replace(pending, archive)
    row.update(contract=request, archive_path=str(archive), archive_sha256=sha(archive))
    write(meta, row)
    print(f"{scope} {request['mode']} {spec['region']} {spec['kind']}: "
          f"{row['wall_s']:.1f}s wall, cut={row['onsets']}, complete={row['mission_completed']}", flush=True)
    return row


def run_region(requests):
    return [one(request) for request in requests]


def collect(requests, stage, workers=6):
    begun = perf_counter()
    rows, groups = [], defaultdict(list)
    for request in requests:
        result = cached(request)
        if result is not None:
            rows.append(result)
        else:
            groups[request['spec']['region']].append(request)
    total = len(requests)
    initial = len(rows)
    status(stage, complete=initial, total=total, workers=workers)
    if groups:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            tasks = [pool.submit(run_region, group) for group in groups.values()]
            for task in as_completed(tasks):
                rows.extend(task.result())
                elapsed = perf_counter()-begun
                status(stage, complete=len(rows), total=total, workers=workers,
                       wall_s=elapsed, eta_s=elapsed*(total-len(rows))/max(1,len(rows)-initial))
    rows.sort(key=lambda row: (row['spec']['region'], row['spec']['kind'], row['mode']))
    if len(rows) != total:
        raise ValueError('Incomplete rollout cohort')
    return rows
