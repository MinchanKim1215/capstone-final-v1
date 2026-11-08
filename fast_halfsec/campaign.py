"""Fit, freeze, evaluate and resume the registered half-second study."""
import argparse
import ctypes
import os
import time

from adaptive_joint_expanded.contracts import lock
from forecast_rebuild.contracts import now, read, sha, write
from . import ARTIFACTS, ROOT
from .evaluation import summarize
from .learning import PROFILE, fit
from .ppo import pin, train
from .protocol import PLAN, prepare, status, verify
from .report import render
from .rollouts import collect, job, record_paths

FREEZE = ARTIFACTS/'freeze.json'
LEARNING_SOURCES = ('learning.py','ppo.py','evaluation.py','campaign.py','report.py')


def input_ready(wait):
    paths = [ARTIFACTS/(role+'_manifest.json') for role in ('train','calibration')]
    deadline = time.monotonic()+3600.
    while not all(path.exists() for path in paths):
        marker = read(ARTIFACTS/'status.json')
        if not wait or marker.get('state')=='failed' or time.monotonic()>=deadline:
            raise RuntimeError('Complete or resume the supervision collection before fitting')
        time.sleep(5.)


def freeze(plan, workers):
    sources = {name:sha(ROOT/name) for name in LEARNING_SOURCES}
    if FREEZE.exists():
        seal = read(FREEZE)
        if seal['protocol_sha256']!=sha(PLAN) or seal['learning_sources']!=sources:
            raise ValueError('Learning implementation changed after confirmation freeze')
        for name in ('forecast','actor'):
            if sha(seal[name]['path'])!=seal[name]['sha256']:
                raise ValueError('A frozen model file changed')
        if sha(ARTIFACTS/'ppo_training.json')!=seal['ppo_training_sha256']:
            raise ValueError('PPO training evidence changed')
        return seal
    if (ARTIFACTS/'confirmation_open.json').exists():
        raise ValueError('Confirmation cannot be opened before the models are frozen')
    status('fitting_forecast', train=len(plan['train']), calibration=len(plan['calibration']))
    fit()
    forecast = pin(PROFILE)
    trained = train(plan,forecast,workers)
    verify(plan)
    seal = dict(created_at=now(),protocol_sha256=sha(PLAN),forecast=forecast,actor=trained['final'],
                ppo_training_sha256=sha(ARTIFACTS/'ppo_training.json'),learning_sources=sources,
                train_manifest_sha256=sha(ARTIFACTS/'train_manifest.json'),
                calibration_manifest_sha256=sha(ARTIFACTS/'calibration_manifest.json'),
                model_selection='Fixed update 2. Confirmation outcomes not read during fitting.')
    write(FREEZE,seal)
    return seal


def execute(workers, wait):
    input_ready(wait)
    plan = prepare()
    done = ARTIFACTS/'completed.json'
    if done.exists():
        expected = read(done)
        if sha(ARTIFACTS/'final_results.json') != expected['result_sha256'] or sha(FREEZE)!=expected['freeze_sha256']:
            raise ValueError('Completed experiment evidence changed')
        print(expected,flush=True)
        return
    seal = freeze(plan,workers)
    open_path = ARTIFACTS/'confirmation_open.json'
    if open_path.exists():
        if read(open_path)['freeze_sha256']!=sha(FREEZE):
            raise ValueError('Confirmation started under other models')
    else:
        write(open_path,dict(opened_at=now(),freeze_sha256=sha(FREEZE),cohort=plan['confirmation']))
    requests = []
    for spec in plan['confirmation']:
        for mode in plan['modes']:
            requests.append(job('confirmation',spec,mode,
                                seal['forecast'] if mode in ('learned_05s','ppo_05s') else None,
                                seal['actor'] if mode=='ppo_05s' else None,collect=mode=='physics_05s'))
    rows = collect(requests,'confirmation_whole_missions',workers)
    status('auditing_results',complete=len(rows),total=len(requests))
    verify(plan)
    result = summarize(rows,plan,seal['forecast'])
    result.update(created_at=now(),freeze_sha256=sha(FREEZE),
                  confirmation_manifest=[dict(path=str(record_paths(row['contract'])[0]),
                                              archive_sha256=row['archive_sha256']) for row in rows])
    write(ARTIFACTS/'final_results.json',result)
    render(result,read(PROFILE),read(ARTIFACTS/'ppo_training.json'),plan,read(ARTIFACTS/'map_registry.json')['counts'])
    report = dict(finished_at=now(),freeze_sha256=sha(FREEZE),result_sha256=sha(ARTIFACTS/'final_results.json'),
                  report_sha256=sha(ROOT/'RESULTS.md'), confirmation_flights=len(rows),
                  trained=True,production_replaced=False)
    write(done,report)
    status('finished',state='completed',**report)
    print(report,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--wait-for-inputs',action='store_true')
    args = parser.parse_args()
    with lock(ARTIFACTS/'campaign.lock'):
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            execute(args.workers,args.wait_for_inputs)
        except BaseException as error:
            status('campaign_failed',state='failed',error=repr(error))
            raise
        finally:
            if os.name=='nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    main()
