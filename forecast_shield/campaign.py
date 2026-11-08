"""Run and seal the paired forecast-only evaluation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

import numpy as np

from . import ARTIFACTS, ROOT
from .protocol import V18, now, prepare, read, sha, status, write


def initialize():
    from forecast_prevention.runner import initialize as load
    load()


def trial(scope, item):
    from .runner import run
    reference = None
    if scope != 'fresh':
        if sha(item['path']) != item['sha256']:
            raise ValueError('V19 flight metadata changed')
        reference = read(item['path'])
        spec = reference['spec']
    else:
        spec = item
    path = ARTIFACTS/scope/f"{spec['region']}_{spec['kind']}.json"
    array = path.with_suffix('.npz')
    contract = dict(spec=spec, scope=scope, source=None if reference is None else reference['archive_sha256'],
                    protocol=sha(ARTIFACTS/'protocol.json'))
    if path.exists():
        row = read(path)
        if row['contract'] != contract or sha(array) != row['archive_sha256']:
            raise ValueError('Completed experiment changed')
    else:
        if reference is None:
            row,data = run(spec)
        else:
            source = reference['archive_path']
            if sha(source) != reference['archive_sha256']:
                raise ValueError('V19 commands changed')
            with np.load(source,allow_pickle=False) as old:
                row,data = run(spec,reference=old['commands'])
                for name in ('nodes','commands','links'):
                    np.testing.assert_array_equal(data[name],old['actual_links' if name=='links' else name])
            for name in ('timely_pdr','generated_packets','delivered_packets','retries','beacon_rx'):
                if row[name] != reference['mission'][name]:
                    raise ValueError('V19 packet or beacon replay changed: '+name)
        path.parent.mkdir(parents=True,exist_ok=True)
        pending = array.with_suffix('.pending.npz')
        np.savez_compressed(pending,**data)
        pending.replace(array)
        write(path,dict(row,contract=contract,archive_path=str(array),archive_sha256=sha(array)))
    return dict(path=str(path),sha256=sha(path))


def group(task):
    scope,items = task
    return [trial(scope,item) for item in items]


def run_stage(scope, batches):
    path = ARTIFACTS/(scope+'_manifest.json')
    if path.exists():
        return read(path)['rows']
    total = sum(len(x) for x in batches)
    status(scope,complete=0,total=total)
    rows = {}
    with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures = {pool.submit(group,(scope,items)):i for i,items in enumerate(batches)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            rows[i] = future.result()
            done += len(rows[i])
            status(scope,complete=done,total=total)
            print(scope,done,'/',total,flush=True)
    records = [row for i in sorted(rows) for row in rows[i]]
    write(path,dict(rows=records,protocol_sha256=sha(ARTIFACTS/'protocol.json'),created_at=now()))
    return records


def preflight():
    from dataclasses import replace
    from .forecaster import TIMES, evaluate_terms, sample_plan
    from .runner import run
    from forecast_prevention.controller import RadioState
    initialize()
    old = read(read(V18/'artifacts'/'fresh_manifest.json')['rows'][0]['path'])
    assert old['arm']=='baseline'
    if sha(old['archive_path']) != old['archive_sha256']:
        raise ValueError('Prior preflight trace changed')
    row,data = run(old['spec'],max_steps=18)
    with np.load(old['archive_path'],allow_pickle=False) as trace:
        for key in ('nodes','links','commands'):
            np.testing.assert_array_equal(data[key],trace[key][:len(data[key])])
    old_advisory = {r['time_s']:r for r in old['advisory']}
    for now_s,score in data['scores'][data['scores'][:,0]%3==0,:2]:
        tick = old_advisory[now_s]
        expected = np.nan if tick['expert'] is None else float(tick['estimated_connected'] and any(3<=h<7 for h in tick['predicted']))
        np.testing.assert_equal(score,expected)
    loss = np.full((len(TIMES),3,3),90.)
    extra = np.zeros_like(loss)
    extra[TIMES>=4.,0,1:] = 55.
    extra[TIMES>=4.,1:,0] = 55.
    state = RadioState(np.zeros((3,9,2)),np.array([True,False,True]),np.zeros(3),np.zeros(3),
                       np.full((3,3),np.nan),np.full(3,-107.),np.zeros(3),False,False,True)
    test = evaluate_terms(loss+extra,extra,state,2,0.,explain=True)
    assert test.building_support>.9 and test.shield_score>.9 and test.onset_in_s==4. and len(test.critical_links)==2
    assert np.isnan(test.legacy_score) and test.planner_score==1. and test.missing_search_beacons==1
    assert evaluate_terms(loss,extra,state,2,0.).shield_score==0.
    assert evaluate_terms(loss+extra,extra,replace(state,local_rf_available=False),2,0.).status=='no_recent_rf'
    knots = np.repeat(np.arange(5.)[:,None,None]*3.,6,axis=1).reshape(5,3,2)
    np.testing.assert_array_equal(sample_plan(knots,2.)[:,0,0],TIMES+2.)
    write(ARTIFACTS/'preflight.json',dict(passed=True,old_v18_scores_exact=True,old_trajectory_and_rf_exact=True,
          planner_endpoints=row['exact_endpoints'],one_second_ticks=len(data['scores']),
          synthetic_building_cut_and_missing_beacon_checked=True,created_at=now()))


def main():
    from .evaluate import load, paired, select, summarize
    plan = prepare()
    final = ARTIFACTS/'completed.json'
    if final.exists():
        if sha(ROOT/'RESULTS.md') != read(final)['report_sha256']:
            raise ValueError('Completed report changed')
        return
    if not (ARTIFACTS/'preflight.json').exists():
        status('native_preflight')
        preflight()
    freeze = ARTIFACTS/'freeze.json'
    if not freeze.exists():
        cohorts = {}
        for split,references in plan['development'].items():
            cohorts[split] = [load(r) for r in run_stage(split,[[ref] for ref in references])]
        selection = select(cohorts['calibration'],cohorts['selection'])
        write(freeze,dict(selection,protocol_sha256=sha(ARTIFACTS/'protocol.json'),created_at=now(),
                     calibration_sha256=sha(ARTIFACTS/'calibration_manifest.json'),
                     selection_sha256=sha(ARTIFACTS/'selection_manifest.json')))
    frozen = read(freeze)
    if frozen['protocol_sha256'] != sha(ARTIFACTS/'protocol.json'):
        raise ValueError('Threshold selection was made on another protocol')
    access = ARTIFACTS/'fresh_access.json'
    if not access.exists():
        write(access,dict(freeze_sha256=sha(freeze),opened_at=now()))
    elif read(access)['freeze_sha256'] != sha(freeze):
        raise ValueError('Profile changed after opening confirmation')
    specs = plan['fresh_specs']
    cohort = [load(r) for r in run_stage('fresh',[specs[i:i+3] for i in range(0,len(specs),3)])]
    if [row['spec'] for row,_ in cohort] != specs:
        raise ValueError('Fresh comparison does not cover the pinned cases')
    results = dict(v18=summarize(cohort,'v18',.5,True))
    for name,profile in frozen['profiles'].items():
        variant,threshold = profile['variant'],profile['threshold']
        results[name] = dict(variant=variant,threshold=threshold,calibration_budget=profile['cap'],
                             **summarize(cohort,variant,threshold,True), paired=paired(cohort,variant,threshold))
    # Keep the other development-calibrated variants visible as ablations.
    results['ablations'] = {k:dict(threshold=v['threshold'],**summarize(cohort,k,v['threshold'],True))
                           for k,v in frozen['profiles']['priority']['options'].items()}
    durations = np.concatenate([score['elapsed_ms'] for _,score in cohort])
    results.update(live_forecast_count=len(durations), latency_p95_ms=float(np.percentile(durations,95)),
                   exact_first_segments=sum(row['exact_endpoints'] for row,_ in cohort),
                   native_missions=len(cohort), mission_completed=sum(row['mission_completed'] for row,_ in cohort),
                   safety_violations=sum(row['actual_safety_violations'] for row,_ in cohort),created_at=now())
    write(ARTIFACTS/'final_results.json',results)
    from .report import finish
    finish(frozen,results)
    write(final,dict(report_sha256=sha(ROOT/'RESULTS.md'),freeze_sha256=sha(freeze),
                     results_sha256=sha(ARTIFACTS/'final_results.json'),created_at=now()))
    status('audited_forecast_evaluation',state='completed',fresh_flights=len(cohort),profile=results['priority']['variant'])


if __name__=='__main__':
    import ctypes
    from adaptive_joint_expanded.contracts import lock
    with lock(ARTIFACTS/'campaign.lock'):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            main()
        except BaseException as error:
            status('campaign',state='failed',error=repr(error))
            traceback.print_exc()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
