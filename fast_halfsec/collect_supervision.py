"""Generate the frozen 0.5-second supervised train and calibration cohorts."""
import argparse

from forecast_rebuild.contracts import write
from . import ARTIFACTS
from .protocol import prepare, status
from .rollouts import collect, job, record_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    plan = prepare()
    try:
        for role in ('train','calibration'):
            requests = [job(role,spec,'physics_05s',collect=True) for spec in plan[role]]
            rows = collect(requests, 'supervision_'+role, args.workers)
            write(ARTIFACTS/(role+'_manifest.json'), dict(records=[dict(
                metadata=str(record_paths(row['contract'])[0]), archive=row['archive_path'],
                sha256=row['archive_sha256']) for row in rows]))
        status('supervision_ready', state='ready', train=len(plan['train']), calibration=len(plan['calibration']))
    except BaseException as error:
        status('supervision_failed', state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
