"""Freeze geography, track progress, and cache verified simulation artifacts."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
import ctypes
import multiprocessing as mp
import msvcrt
import os

import numpy as np

from fast_halfsec.context import registry
from fast_halfsec.protocol import KINDS, choose, make_spec, verify as verify_prior
from forecast_rebuild.contracts import digest, now, read, sha, write

from . import ARTIFACTS, FORECAST, PRIOR, ROOT
from .choice import HORIZON, INPUTS, MAX_ALTERNATIVES
from .records import ANCHOR_CAP, replay

VERSION = 'Final V1 half-second guarded choice'
PROTOCOL = ARTIFACTS / 'protocol.json'
SOURCE_FILES = ('__init__.py', 'choice.py', 'records.py', 'experiment.py')


def status(stage, **values):
    write(ARTIFACTS / 'status.json', dict(version=VERSION, stage=stage,
                                         state=values.pop('state', 'running'), updated_at=now(), **values))


@contextmanager
def exclusive_run():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
        try:
            keep_awake(0x80000001)
            yield
        finally:
            keep_awake(0x80000000)
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def prepare():
    prior = read(PRIOR / 'protocol.json')
    verify_prior(prior)
    prior_freeze = read(PRIOR / 'freeze.json')
    pins = {name: sha(PRIOR / name) for name in
            ('protocol.json', 'freeze.json', 'train_manifest.json', 'calibration_manifest.json')}
    if (pins['protocol.json'] != prior_freeze['protocol_sha256']
            or pins['train_manifest.json'] != prior_freeze['train_manifest_sha256']
            or pins['calibration_manifest.json'] != prior_freeze['calibration_manifest_sha256']
            or sha(FORECAST) != prior_freeze['forecast']['sha256']):
        raise ValueError('Earlier model or cohorts have changed')
    sources = {name: sha(ROOT / name) for name in SOURCE_FILES}
    if PROTOCOL.exists():
        plan = read(PROTOCOL)
        if plan['prior'] != pins or plan['sources'] != sources:
            raise ValueError('Learning protocol was frozen with different inputs')
        return plan

    maps = registry()
    excluded = {maps[spec['region']]['study_group'] for spec in prior['confirmation']}
    available = [row for row in maps.values()
                 if row['study_role'] == 'confirmation' and row['study_group'] not in excluded]
    chosen = choose(available, 8, 'nov8_guarded_choice', lambda row: row['source_archive'])
    confirmation = [make_spec(row, 'nov8_guarded_choice', kind) for row in chosen for kind in KINDS[1:]]
    cohorts = {}
    for role in ('train', 'calibration'):
        rows = read(PRIOR / f'{role}_manifest.json')['records']
        by_key = {digest(read(record['metadata'])['spec']): record for record in rows}
        if len(rows) != len(by_key) or set(by_key) != {digest(spec) for spec in prior[role]}:
            raise ValueError(f'Incomplete previous {role} cohort')
        cohorts[role] = [dict(record, metadata_sha256=sha(record['metadata'])) for record in rows]
    groups = [{maps[spec['region']]['study_group'] for spec in cohort}
              for cohort in (prior['train'], prior['calibration'], confirmation)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Guarded choice cohorts overlap')
    plan = dict(version=VERSION, created_at=now(), prior=pins, sources=sources, records=cohorts,
                fresh_confirmation=confirmation, excluded_previous_confirmation_groups=sorted(excluded),
                confirmation_groups=sorted(groups[-1]), branch_horizon_s=HORIZON,
                max_candidates=MAX_ALTERNATIVES, max_anchors_per_flight=ANCHOR_CAP, inputs=list(INPUTS),
                decision='Select a guarded positive alternative at a half-second preventive planning tick.',
                label='Paired 12-second radio rollout with the same disturbances and 0.3s assumed delivery.',
                acceptance=prior['acceptance'])
    write(PROTOCOL, plan)
    return plan


def archive_paths(category, name):
    prefix = ARTIFACTS / category / name
    return prefix.with_suffix('.json'), prefix.with_suffix('.npz')


def cached(paths, contract):
    meta, archive = paths
    if not meta.exists() or not archive.exists():
        return None
    row = read(meta)
    if row['contract'] != contract or row['archive_sha256'] != sha(archive):
        raise ValueError(f'Cached simulation result was modified: {meta}')
    return row


def save(paths, contract, meta, arrays):
    meta_path, archive = paths
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix('.pending.npz')
    np.savez_compressed(pending, **arrays)
    os.replace(pending, archive)
    row = dict(meta, contract=contract, archive_path=str(archive), archive_sha256=sha(archive))
    write(meta_path, row)
    return row


def replay_job(job):
    role, record, plan_hash = job['role'], job['record'], job['protocol']
    paths = archive_paths(f'branches/{role}', Path(record['metadata']).stem)
    if sha(record['metadata']) != record['metadata_sha256']:
        raise ValueError('Training rollout metadata has changed')
    result = cached(paths, job)
    if result is not None:
        return result
    if sha(PROTOCOL) != plan_hash:
        raise ValueError('Protocol changed during collection')
    row, arrays = replay(record, role)
    return save(paths, job, row, arrays)


def gather(tasks, worker, stage, workers):
    total = sum(map(len, tasks))
    rows = []
    started = perf_counter()
    status(stage, complete=0, total=total, workers=workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(worker, jobs) for jobs in tasks]
        for task in as_completed(futures):
            results = task.result()
            rows.extend(results)
            elapsed = perf_counter() - started
            status(stage, complete=len(rows), total=total, workers=workers, wall_s=elapsed,
                   eta_s=elapsed * (total - len(rows)) / len(rows))
            for result in results:
                spec = result['spec']
                suffix = (f"pairs={result['comparisons']}" if 'comparisons' in result else
                          f"{result['mode']} starts={result['onsets']} complete={result['mission_completed']}")
                print(f"{stage} {len(rows)}/{total} {spec['region']} {spec['kind']} {suffix}", flush=True)
    return sorted(rows, key=lambda row: (row['spec']['region'], row['spec']['kind'], row.get('mode', '')))


def replay_jobs(jobs):
    return [replay_job(job) for job in jobs]


def build_data(plan, workers):
    path = ARTIFACTS / 'branch_manifest.json'
    if path.exists():
        manifest = read(path)
        if manifest['protocol_sha256'] != sha(PROTOCOL):
            raise ValueError('Stored branches have a different protocol')
        return manifest
    cohort = []
    for role, records in plan['records'].items():
        for record in records:
            cohort.append([dict(role=role, record=record, protocol=sha(PROTOCOL))])
    rows = gather(cohort, replay_jobs, 'replay_training_and_calibration', workers)
    counts = defaultdict(dict)
    for role in plan['records']:
        subset = [row for row in rows if row['role'] == role]
        for name in ('prevention_anchors', 'replayed', 'eligible', 'comparisons', 'beneficial',
                     'fewer_starts', 'worse'):
            counts[role][name] = sum(row[name] for row in subset)
        counts[role].update(maps=len(subset), eligible_maps=sum(row['eligible'] > 0 for row in subset))
    manifest = dict(protocol_sha256=sha(PROTOCOL), counts=dict(counts), records=rows)
    write(path, manifest)
    status('branch_data_ready', counts=manifest['counts'])
    return manifest


if __name__ == '__main__':
    with exclusive_run():
        try:
            status('verifying_training_inputs')
            build_data(prepare(), 6)
        except Exception as error:
            status('branch_data_failed', state='failed', error=str(error))
            raise
