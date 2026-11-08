"""Freeze independent maps and collect resumable V7 simulation traces."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from time import perf_counter
import ctypes
import multiprocessing as mp
import msvcrt
import os

import numpy as np

from fast_halfsec.context import registry
from fast_halfsec.protocol import KINDS, choose, make_spec
from forecast_rebuild.contracts import BASELINE, digest, now, read, sha, verify_prior, write

from . import ARTIFACTS, MAP_ARTIFACTS, ROOT, WORKSPACE
from .capture import collect_flight

PLAN = ARTIFACTS / 'protocol.json'
MANIFEST = ARTIFACTS / 'development_manifest.json'
VERSION = 'Final V1 Transformer upgrade'
MAP_COUNTS = {'new_train': 144, 'legacy_train': 36, 'calibration': 24, 'confirmation': 12}


def status(stage, **values):
    write(ARTIFACTS / 'status.json', dict(version=VERSION, stage=stage,
                                         state=values.pop('state', 'running'),
                                         updated_at=now(), **values))


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
    verify_prior()
    old_v1 = read(WORKSPACE / 'final_v1/artifacts/protocol.json')
    for name, expected in old_v1['source_sha256'].items():
        if sha(WORKSPACE / 'final_v1' / name) != expected:
            raise ValueError(f'Frozen Final V1 source changed: {name}')
    for name, expected in old_v1['dependency_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Frozen Final V1 dependency changed')
    files = (BASELINE, MAP_ARTIFACTS / 'map_registry.json', MAP_ARTIFACTS / 'protocol.json',
             WORKSPACE / 'halfsec_choice/artifacts/protocol.json',
             WORKSPACE / 'final_v1/artifacts/protocol.json', WORKSPACE / 'final_v1/artifacts/release.json',
             ROOT / '__init__.py', ROOT / 'capture.py', ROOT / 'collection.py')
    pins = {str(path): sha(path) for path in files}
    if PLAN.exists():
        plan = read(PLAN)
        if plan['pins'] != pins:
            raise ValueError('Previously frozen upgrade inputs changed')
        return plan

    maps = registry()
    previous = read(MAP_ARTIFACTS / 'protocol.json')['confirmation']
    previous += read(WORKSPACE / 'halfsec_choice/artifacts/protocol.json')['fresh_confirmation']
    excluded = {maps[spec['region']]['study_group'] for spec in previous}
    train = [row for row in maps.values() if row['study_role'] == 'train']
    stratum = lambda row: row.get('source_archive', row['key'].split('_', 1)[0])
    old = choose([row for row in train if not row['new_map']], MAP_COUNTS['legacy_train'],
                 'transformer_legacy', stratum)
    used = {row['study_group'] for row in old}
    new = choose([row for row in train if row['new_map'] and row['study_group'] not in used],
                 MAP_COUNTS['new_train'], 'transformer_new', stratum)
    calibration = choose([row for row in maps.values() if row['study_role'] == 'calibration'],
                         MAP_COUNTS['calibration'], 'transformer_calibration', stratum)
    confirmation = choose([row for row in maps.values() if row['study_role'] == 'confirmation'
                           and row['study_group'] not in excluded],
                          MAP_COUNTS['confirmation'], 'transformer_confirmation', stratum)
    geography = [{row['study_group'] for row in cohort} for cohort in (new + old, calibration, confirmation)]
    if any(geography[i] & geography[j] for i in range(3) for j in range(i)):
        raise ValueError('Upgrade study groups overlap')
    cohorts = {
        'train': [make_spec(row, 'transformer_train', KINDS[i % len(KINDS)])
                  for i, row in enumerate(new + old)],
        'calibration': [make_spec(row, 'transformer_calibration', KINDS[i % len(KINDS)])
                        for i, row in enumerate(calibration)],
        'confirmation': [make_spec(row, 'transformer_confirmation', kind)
                         for row in confirmation for kind in KINDS],
    }
    plan = dict(version=VERSION, created_at=now(), pins=pins, cohorts=cohorts,
                groups={key: sorted(groups) for key, groups in zip(cohorts, geography, strict=True)},
                excluded_confirmation_groups=sorted(excluded),
                clocks_s=dict(radio=.5, observation_history=20., control=3.),
                alert_cap_per_fleet_hour=4., alert_window_s=[3., 7.],
                promotion=dict(all_missions_completed=True, maximum_pdr_loss_pp=.25,
                               maximum_mission_duration_fraction=.02, no_more_cut_seconds_per_h=True,
                               require_upper_95_outage_rate_delta_below_zero=True,
                               require_alert_recall_improvement=True),
                selection='Fit with training maps, choose checkpoints and alert cutoffs on calibration only.')
    write(PLAN, plan)
    return plan


def paths_for(folder, spec):
    prefix = ARTIFACTS / folder / digest(spec)[:20]
    return prefix.with_suffix('.json'), prefix.with_suffix('.npz')


def cached(paths, contract):
    meta, archive = paths
    if not (meta.is_file() and archive.is_file()):
        return None
    row = read(meta)
    if row['contract'] != contract or row['archive_sha256'] != sha(archive):
        raise ValueError(f'Inconsistent cached simulation: {meta}')
    return row


def save(paths, contract, meta, arrays):
    output, archive = paths
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix('.pending.npz')
    np.savez_compressed(pending, **arrays)
    os.replace(pending, archive)
    row = dict(meta, contract=contract, archive_path=str(archive), archive_sha256=sha(archive))
    write(output, row)
    return row


def capture_job(job):
    role, spec = job['role'], job['spec']
    if sha(PLAN) != job['protocol'] or registry()[spec['region']]['study_role'] != role:
        raise ValueError('Capture differs from frozen cohort')
    if role == 'confirmation':
        if sha(ARTIFACTS / 'freeze.json') != job['freeze']:
            raise ValueError('Confirmation forecast cannot be collected before freeze')
    paths = paths_for(f'capture/{role}', spec)
    previous = cached(paths, job)
    if previous is not None:
        return previous
    row, arrays = collect_flight(spec)
    row.update(role=role, group=registry()[spec['region']]['study_group'],
               link_rows=int((arrays['link_y'] >= 0).sum()),
               link_early=int(((arrays['link_y'] >= 6) & (arrays['link_y'] < 14)).sum()),
               network_rows=int(((arrays['network_y'] >= 0) & np.isfinite(arrays['route'][:, 0])).sum()),
               network_early=int(((arrays['network_y'] >= 2) & np.isfinite(arrays['route'][:, 0])).sum()))
    return save(paths, job, row, arrays)


def capture_jobs(jobs):
    return [capture_job(job) for job in jobs]


def gather(groups, worker, stage, workers):
    total, rows, start = sum(map(len, groups)), [], perf_counter()
    status(stage, complete=0, total=total, workers=workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        pending = [pool.submit(worker, jobs) for jobs in groups]
        for task in as_completed(pending):
            results = task.result()
            rows.extend(results)
            elapsed = perf_counter() - start
            status(stage, complete=len(rows), total=total, workers=workers,
                   last_map=results[-1]['spec']['region'], wall_s=elapsed,
                   eta_s=elapsed * (total - len(rows)) / len(rows))
            for result in results:
                spec = result['spec']
                print(f"{stage} {len(rows)}/{total} {spec['region']} {spec['kind']} "
                      f"{result.get('upgrade_arm', result['arm'])} onsets={result['persistent_onsets']} "
                      f"complete={result['mission_completed']} wall={result['wall_s']:.1f}s", flush=True)
    return sorted(rows, key=lambda row: (row['spec']['region'], row['spec']['kind'],
                                         row.get('upgrade_arm', row['arm'])))


def collect(plan, workers=6):
    expected = sha(PLAN)
    if MANIFEST.exists():
        manifest = read(MANIFEST)
        if manifest['protocol_sha256'] != expected:
            raise ValueError('Collection manifest has a different protocol')
        return manifest
    train = plan['cohorts']['train']
    calibration = plan['cohorts']['calibration']
    # Collect both roles from the start so progress shows every geography.
    groups = [[dict(role='calibration', spec=spec, protocol=expected)] for spec in calibration]
    groups.extend([dict(role='train', spec=spec, protocol=expected)] for spec in train)
    rows = gather(groups, capture_jobs, 'collecting_development_flights', workers)
    counts = defaultdict(dict)
    for role in ('train', 'calibration'):
        group = [row for row in rows if row['role'] == role]
        counts[role].update(flights=len(group), complete=sum(row['mission_completed'] for row in group),
                            hours=sum(row['duration_s'] for row in group) / 3600.)
        for name in ('link_rows', 'link_early', 'network_rows', 'network_early'):
            counts[role][name] = sum(row[name] for row in group)
    manifest = dict(protocol_sha256=expected, counts=dict(counts), records=rows)
    write(MANIFEST, manifest)
    status('development_data_ready', counts=manifest['counts'])
    return manifest


if __name__ == '__main__':
    with exclusive_run():
        try:
            status('verifying_inputs')
            collect(prepare())
        except Exception as error:
            status('collection_failed', state='failed', error=repr(error))
            raise
