"""Compare reserve-route prevention with the archived Final V1 0.5-second flights."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from adaptive_joint_expanded.contracts import lock
from fast_halfsec.evaluation import audit, pooled
from final_v1 import ARTIFACTS
from final_v1.halfsec_joint.run import retry_permission
from forecast_rebuild.contracts import digest, now, read, sha, write

from . import ROOT
from .release import MANIFEST
from .reserve_routes import run


BASELINE = ARTIFACTS / 'halfsec_main'
TRIAL = ARTIFACTS / 'halfsec_reserve_routes'
FOCUS = {'chungnam_20261113_b579f10221', 'chungnam_20261113_n2q_aad84fa167'}


def references(other):
    protocol = read(BASELINE / 'protocol.json')
    cases = [item['spec'] for item in protocol['cases']['separate_maps']
             if (item['spec']['region'] in FOCUS) != other]
    rows = {}
    for spec in cases:
        identifier = digest(spec)[:24]
        row = read(BASELINE / 'flights' / (identifier + '.json'))
        if row['spec'] != spec or row['protocol'] != sha(BASELINE / 'protocol.json'):
            raise ValueError(f'Unexpected Final V1 baseline for {spec}')
        archive = BASELINE / 'flights' / (identifier + '.npz')
        if sha(archive) != row['archive_sha256']:
            raise ValueError(f'Unexpected radio trace for {spec}')
        rows[identifier] = dict(row, archive_path=str(archive))
    return rows


def simulate(spec, folder, identifier):
    torch.set_num_threads(1)
    record = folder / (identifier + '.json')
    archive = folder / (identifier + '.npz')
    if record.exists():
        return read(record)
    row, arrays = retry_permission(run, spec)
    pending = archive.with_suffix('.pending.npz')
    names = ('links', 'network', 'positions', 'margin_db', 'applied_offsets')
    np.savez_compressed(pending, **{name: arrays[name] for name in names})
    retry_permission(Path.replace, pending, archive)
    row.update(archive_path=str(archive), archive_sha256=sha(archive))
    write(record, row)
    return row


def comparison_fields(row):
    keys = ('onsets', 'cut_seconds', 'duration_s', 'mission_completed', 'timely_pdr',
            'prevention_requests', 'relay_travel_m', 'planner_p95_ms', 'planner_over_200ms',
            'plan_ticks', 'cuts')
    return dict({key: row[key] for key in keys},
                max_cut_s=max((end - start for start, end in row['cuts']), default=0.),
                reserve_requests=row.get('reserve_requests', 0),
                reserve_trigger_ticks=row.get('reserve_trigger_ticks', 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--other-maps', action='store_true', help='Run the other six archived map conditions')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    label = 'other_maps' if args.other_maps else 'chungnam'
    source = {'candidate_sha256': sha(ROOT / 'reserve_routes.py'),
              'test_runner_sha256': sha(ROOT / 'reserve_trial.py'),
              'baseline_protocol_sha256': sha(BASELINE / 'protocol.json'),
              'checkpoint_manifest_sha256': sha(MANIFEST)}
    folder = TRIAL / digest(source)[:12]
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'sources.json', source)
    before = references(args.other_maps)
    print(f'Output: {folder}', flush=True)
    with lock(folder / (label + '.lock')):
        with ProcessPoolExecutor(max_workers=min(args.workers, len(before))) as pool:
            tasks = {pool.submit(simulate, row['spec'], folder, key): key for key, row in before.items()}
            after = {}
            for task in as_completed(tasks):
                identifier = tasks[task]
                row = after[identifier] = task.result()
                print(row['spec']['region'], comparison_fields(row), flush=True)
        pairs = [dict(spec=before[key]['spec'], before=comparison_fields(before[key]),
                      after=comparison_fields(after[key])) for key in sorted(before)]
        verdict = dict(completed_at=now(), source=source, cases=pairs,
                       before=pooled(list(before.values())), after=pooled(list(after.values())),
                       audit=audit(list(before.values()) + list(after.values())))
        write(folder / (label + '.json'), verdict)


if __name__ == '__main__':
    main()
