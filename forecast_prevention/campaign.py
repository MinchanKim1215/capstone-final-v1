"""Complete the wide cause survey and paired closed-loop study in order."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

from . import ROOT, ARTIFACTS
from .protocol import prepare, read, write, sha, now, status, verify_freeze
from adaptive_joint_expanded.contracts import lock

WORKERS = 5


def run_jobs(stage, function, jobs, initializer=None):
    rows = {}
    status(stage, complete=0, total=len(jobs))
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn'), initializer=initializer) as pool:
        pending = {pool.submit(function, task): i for i, task in enumerate(jobs)}
        for future in as_completed(pending):
            rows[pending[future]] = future.result()
            status(stage, complete=len(rows), total=len(jobs))
            if len(rows) % 5 == 0 or len(rows) == len(jobs):
                print(stage, len(rows), '/', len(jobs), flush=True)
    return [rows[i] for i in range(len(jobs))]


def main():
    from .patterns import survey_one
    from .runner import initialize, simulate, archived_baseline
    from .evaluation import aggregate, comparison, select_prevention, survey_summary
    if (ARTIFACTS / 'completed.json').exists():
        verify_freeze()
        print('V17 completed. Read', ROOT / 'RESULTS.md', flush=True)
        return
    plan = prepare()
    if not read(ARTIFACTS / 'preflight.json')['passed']:
        raise ValueError('Native preflight is required')
    survey_path = ARTIFACTS / 'survey_manifest.json'
    if not survey_path.exists():
        original = run_jobs('archived_cause_survey', survey_one, plan['archived_records'])
        historical = run_jobs('new_historical_cause_flights', simulate,
                              [(spec, 'baseline', 'historical') for spec in plan['historical_specs']], initialize)
        write(survey_path, dict(archived=original, historical=historical, created_at=now()))
    if not (ARTIFACTS / 'cause_summary.json').exists():
        survey_summary(read(survey_path), plan)
    freeze_path = ARTIFACTS / 'freeze.json'
    if not freeze_path.exists():
        jobs = [(r['spec'], arm, 'selection') for r in plan['selection_sources'] for arm in plan['selection_arms']]
        rows = run_jobs('selection_closed_loop', simulate, jobs, initialize)
        keys = {source['key'] for source in plan['selection_sources']}
        baselines = [archived_baseline(item) for item in plan['archived_records'] if item['key'] in keys]
        results = [read(row['path']) for row in rows]
        selected = select_prevention(baselines, results, plan)
        path = ARTIFACTS / 'selection.json'
        write(path, dict(chosen=selected, arms=aggregate(baselines + results), rows=rows, baseline=baselines,
                         comparisons=comparison(baselines, results, plan), created_at=now()))
        names = ('protocol.py', 'runner.py', 'controller.py', 'diagnostics.py', 'patterns.py', 'evaluation.py', 'campaign.py')
        write(freeze_path, dict(chosen=selected, protocol_sha256=sha(ARTIFACTS / 'protocol.json'),
                               selection_sha256=sha(path), source_sha256={name: sha(ROOT / name) for name in names},
                               cause_summary_sha256=sha(ARTIFACTS / 'cause_summary.json'), created_at=now()))
    frozen = verify_freeze()
    fresh_path = ARTIFACTS / 'fresh_manifest.json'
    if not fresh_path.exists():
        write(ARTIFACTS / 'fresh_access.json', dict(opened_at=now(), freeze_sha256=sha(freeze_path)))
        jobs = [(spec, arm, 'fresh') for spec in plan['fresh_specs']
                for arm in ('baseline', 'reactive', frozen['chosen'])]
        rows = run_jobs('fresh_closed_loop', simulate, jobs, initialize)
        write(fresh_path, dict(rows=rows, freeze_sha256=sha(freeze_path), created_at=now()))
    status('paired_evaluation')
    from .report import finish
    finish(plan, frozen)


if __name__ == '__main__':
    import ctypes
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with lock(ARTIFACTS / 'campaign.lock'):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            main()
        except BaseException as error:
            status('campaign', state='failed', error=repr(error))
            traceback.print_exc()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
