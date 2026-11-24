"""Evaluate planned relay passages against the archived Final V1 missions."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from adaptive_joint_expanded.contracts import lock
from fast_halfsec.evaluation import audit, pooled
from final_v1 import ARTIFACTS
from final_v1.halfsec_joint.run import retry_permission
from forecast_rebuild.contracts import digest, now, read, sha, write

from . import ROOT
from .release import MANIFEST, WEIGHTS
from .reserve_trial import BASELINE, comparison_fields, references
from .staged_passage import Passage, run


TRIAL = ARTIFACTS / 'halfsec_staged_passage'


def simulate(spec, passages, folder):
    torch.set_num_threads(1)
    identifier = digest(spec)[:24]
    target = folder / (identifier + '.json')
    archive = target.with_suffix('.npz')
    if target.exists():
        return read(target)
    row, arrays = retry_permission(run, spec, tuple(Passage(**item) for item in passages))
    pending = archive.with_suffix('.pending.npz')
    np.savez_compressed(pending, **{key: arrays[key] for key in ('network', 'links', 'positions', 'margin_db')})
    retry_permission(pending.replace, archive)
    row.update(archive_path=str(archive), archive_sha256=sha(archive))
    write(target, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('schedule', help='JSON with archived case IDs and their relay passages')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    schedule = read(args.schedule)
    baseline_all = references(False) | references(True)
    if not schedule['cases'] or not schedule['cases'].keys() <= baseline_all.keys():
        raise ValueError('Unknown or empty archived flight schedule')
    before = {key: baseline_all[key] for key in schedule['cases']}
    source = dict(schedule=schedule, schedule_sha256=sha(args.schedule),
                  code_sha256={name: sha(ROOT / name) for name in ('staged_passage.py', 'staged_trial.py')},
                  baseline_sha256=sha(BASELINE / 'protocol.json'), checkpoint_sha256=sha(WEIGHTS),
                  checkpoint_manifest_sha256=sha(MANIFEST))
    folder = TRIAL / digest(source)[:12]
    folder.mkdir(parents=True, exist_ok=True)
    print(f'Output: {folder}', flush=True)
    with lock(folder / 'trial.lock'):
        write(folder / 'protocol.json', source)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(before))) as pool:
            futures = {pool.submit(simulate, row['spec'], schedule['cases'][key], folder): key
                       for key, row in before.items()}
            after = {}
            for task in as_completed(futures):
                row = after[futures[task]] = task.result()
                print(row['spec']['region'], 'cuts', row['onsets'], row['cut_seconds'],
                      'duration', row['duration_s'], 'passages', row['staged_passages'], flush=True)
        cases = [dict(spec=before[key]['spec'], before=comparison_fields(before[key]),
                      after=comparison_fields(after[key]), planned=after[key]['planned_passages'],
                      staged=after[key]['staged_passages']) for key in sorted(before)]
        result = dict(completed_at=now(), protocol_sha256=sha(folder / 'protocol.json'), cases=cases,
                      before=pooled(before.values()), after=pooled(after.values()),
                      audit=audit([*before.values(), *after.values()]))
        write(folder / 'results.json', result)


if __name__ == '__main__':
    main()
