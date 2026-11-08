"""Run the preregistered comparisons and persist resumable evidence."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

import numpy as np

from . import ARTIFACTS, ROOT
from .protocol import CANDIDATES, SOURCES, V17, now, prepare, read, sha, status, write


def initialize():
    from forecast_prevention.runner import initialize as load
    load()


def simulate(task):
    from .runner import run
    spec, arm, scope, shadow = task
    path = ARTIFACTS / scope / (f"{spec['region']}_{spec['kind']}_{arm}.json")
    archive = path.with_suffix('.npz')
    contract = dict(spec=spec, arm=arm, scope=scope, shadow=shadow,
                    sources={name: sha(ROOT / name) for name in SOURCES})
    if path.exists():
        row = read(path)
        if row['contract'] != contract or sha(archive) != row['archive_sha256']:
            raise ValueError('Existing trial changed: ' + str(path))
    else:
        row, evidence = run(spec, arm, shadow_arm=shadow)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = archive.with_suffix('.pending.npz')
        np.savez_compressed(temp, **evidence)
        temp.replace(archive)
        row.update(contract=contract, archive_path=str(archive), archive_sha256=sha(archive))
        write(path, row)
    return dict(path=str(path), sha256=sha(path))


def branch(task):
    from .evaluation import branch_result
    from .runner import run
    target, arm = task
    step = int((target['time_s']+30)/3)
    row, data = run(target['spec'], arm, max_steps=step, branch_at=target['time_s'])
    return branch_result(target, row, data)


def jobs(stage, work, tasks):
    status(stage, complete=0, total=len(tasks))
    out = [None] * len(tasks)
    if tasks:
        with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
            pending = {pool.submit(work, task): i for i, task in enumerate(tasks)}
            for count, future in enumerate(as_completed(pending), 1):
                out[pending[future]] = future.result()
                status(stage, complete=count, total=len(tasks))
                if count % 5 == 0 or count == len(tasks):
                    print(stage, count, '/', len(tasks), flush=True)
    return out


def preflight():
    from .controller import fragile_margin, onset_mask
    from .runner import run
    initialize()
    record = read(V17 / 'protocol.json')['archived_records'][0]['source_record']
    source = record['array_path']
    if sha(source) != record['array_sha256']:
        raise ValueError('Archived preflight source changed')
    row, data = run(record['spec'], 'baseline', shadow_arm='joint_radio', max_steps=18,
                    check_every_projection=True)
    with np.load(source, allow_pickle=False) as old:
        for name in ('commands', 'nodes', 'links'):
            np.testing.assert_array_equal(data[name], old[name][:len(data[name])])
    if row['exact_endpoint_checks'] != 17 or not row['advisory']:
        raise ValueError('Planner did not reach normal entry point')
    np.testing.assert_array_equal(onset_mask(np.array([1., -1., -1., 1., -1.])), [True, False, False])
    graph = np.array([[[99., 5., -8.], [5., 99., 3.], [-8., 3., 99.]]])
    np.testing.assert_equal(fragile_margin(graph, 2), [-8.])
    write(ARTIFACTS / 'preflight.json', dict(passed=True, exact_endpoint_checks=17,
          archive_motion_and_channel_equal=True, persistence_and_backup_route_checks=True, time=now()))


def rows_from(records):
    rows = []
    for record in records:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Trial result changed')
        row = read(record['path'])
        if sha(row['archive_path']) != row['archive_sha256']:
            raise ValueError('Trial trace changed')
        rows.append(row)
    return rows


def verify(plan, frozen):
    pins = {name: sha(ROOT / name) for name in SOURCES}
    if pins != frozen['source_sha256'] or sha(ARTIFACTS / 'selection.json') != frozen['selection_sha256']:
        raise ValueError('Controller or selection evidence changed after freeze')
    if sha(ARTIFACTS / 'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Protocol changed after freeze')
    if any(sha(path) != expected for path, expected in plan['dependencies'].items()):
        raise ValueError('Frozen source changed')


def main():
    from .evaluation import advisory_summary, branch_targets, recheck_survey
    from forecast_prevention.evaluation import aggregate, comparison
    plan = prepare()
    freeze_path = ARTIFACTS / 'freeze.json'
    if (ARTIFACTS / 'completed.json').exists():
        verify(plan, read(freeze_path))
        print('See', ROOT / 'RESULTS.md', flush=True)
        return
    if not freeze_path.exists():
        status('preflight_and_cause_recheck')
        preflight()
        recheck_survey()
        dev = jobs('development', simulate, [(spec, arm, 'development', None)
                   for spec in plan['selection_specs'] for arm in CANDIDATES])
        results = rows_from(dev)
        baseline = plan['development_reference']
        prior = [r for r in rows_from(read(V17 / 'selection.json')['rows']) if r['arm']=='geometry']
        guards = dict(decision_guards=plan['guard'])
        stats = aggregate(baseline + prior + results)
        against_base = comparison(baseline, prior + results, guards)
        against_previous = comparison(prior, results, guards)
        chosen = min(CANDIDATES, key=lambda arm: (not against_base[arm]['feasible'],
                     stats[arm]['onsets_per_hour'], stats[arm]['cut_seconds_per_hour'], stats[arm]['relay_travel_m'], arm))
        write(ARTIFACTS / 'selection.json', dict(chosen=chosen, rows=dev, arms=stats,
              vs_baseline=against_base, vs_v17=against_previous, created_at=now()))
        write(freeze_path, dict(chosen=chosen, selection_sha256=sha(ARTIFACTS / 'selection.json'),
              protocol_sha256=sha(ARTIFACTS / 'protocol.json'), source_sha256={name: sha(ROOT/name) for name in SOURCES},
              created_at=now()))
    frozen = read(freeze_path)
    verify(plan, frozen)
    chosen = frozen['chosen']
    path = ARTIFACTS / 'fresh_manifest.json'
    if not path.exists():
        write(ARTIFACTS / 'fresh_access.json', dict(freeze_sha256=sha(freeze_path), opened_at=now()))
        fresh = jobs('fresh_full_missions', simulate, [(spec, arm, 'fresh', chosen if arm=='baseline' else None)
                     for spec in plan['fresh_specs'] for arm in ('baseline', 'geometry', chosen)])
        write(path, dict(rows=fresh, created_at=now(), freeze_sha256=sha(freeze_path)))
    if read(path)['freeze_sha256'] != sha(freeze_path):
        raise ValueError('Confirmation predates controller freeze')
    full = rows_from(read(path)['rows'])
    baseline = [row for row in full if row['arm']=='baseline']
    geometry = [row for row in full if row['arm']=='geometry']
    improved = [row for row in full if row['arm']==chosen]
    branch_path = ARTIFACTS / 'branches.json'
    if not branch_path.exists():
        targets = branch_targets(baseline)
        write(ARTIFACTS / 'branch_targets.json', dict(targets=targets, created_at=now()))
        branches = jobs('identical_history_branches', branch, [(target, chosen) for target in targets])
        write(branch_path, dict(rows=branches, targets_sha256=sha(ARTIFACTS / 'branch_targets.json')))
    guards = dict(decision_guards=plan['guard'])
    base_cmp = comparison(baseline, geometry + improved, guards)
    v17_cmp = comparison(geometry, improved, guards)
    adoption = all(check[chosen]['feasible'] and check[chosen]['paired_uncertainty']['onsets_per_hour_ci95'][1] < 0
                   for check in (base_cmp, v17_cmp))
    development = read(ARTIFACTS / 'selection.json')
    adoption &= all(development[key][chosen]['onsets_per_hour_delta'] < 0 for key in ('vs_baseline', 'vs_v17'))
    result = dict(chosen=chosen, arms=aggregate(full), vs_baseline=base_cmp, vs_v17=v17_cmp,
                  exact_endpoint_checks=sum(row['exact_endpoint_checks'] for row in improved),
                  advisory=advisory_summary(baseline), branches=read(branch_path)['rows'],
                  adoption_supported=adoption, created_at=now())
    final = ARTIFACTS / 'final_results.json'
    verify(plan, frozen)
    write(final, result)
    from .report import finish
    finish(result, development, read(ARTIFACTS / 'survey_recheck.json'))
    write(ARTIFACTS / 'completed.json', dict(results_sha256=sha(final), report_sha256=sha(ROOT / 'RESULTS.md'),
          freeze_sha256=sha(freeze_path), fresh_manifest_sha256=sha(path), created_at=now()))
    status('evaluated', state='completed', fresh_cases=len(baseline), branches=len(result['branches']),
           adoption_supported=adoption)


if __name__ == '__main__':
    import ctypes
    from adaptive_joint_expanded.contracts import lock
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
