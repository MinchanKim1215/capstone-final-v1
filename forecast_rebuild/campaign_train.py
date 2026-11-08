"""Finite isolated campaign. No deployment, test opening or silent retries."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import ctypes
import json
import os
import subprocess
import sys
from . import ROOT,WORKSPACE,ARTIFACTS
from .contracts import read,write,sha,now,verify_prior


def run(module,args,tag):
    folder=ARTIFACTS/'campaign';folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-B','-m',module,*args]
    with (folder/(tag+'.stdout.log')).open('w',encoding='utf-8') as output,(folder/(tag+'.stderr.log')).open('w',encoding='utf-8') as error:
        result=subprocess.run(command,cwd=WORKSPACE,stdout=output,stderr=error,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    if result.returncode:raise RuntimeError(tag+' failed; committed checkpoints/logs preserved')
    return tag


def core_cpu():
    for energy,tag in [(False,'expanded_lgbm_base'),(True,'expanded_lgbm_energy')]:
        args=['--samples','forecast_rebuild/artifacts/samples/expanded_core','--tag',tag,'--rounds','700']
        if not energy:args.append('--no-energy')
        run('forecast_rebuild.train_lgbm',args,tag)
    return 'core_cpu_completed'


def core_gpu():
    for energy,tag in [(False,'expanded_residual_base'),(True,'expanded_residual_energy')]:
        args=['--samples','forecast_rebuild/artifacts/samples/expanded_core','--tag',tag,'--family','residual_transformer','--steps','5000']
        if not energy:args.append('--no-energy')
        run('forecast_rebuild.train_neural',args,tag)
    return 'core_gpu_completed'


def planned_cpu():
    return run('forecast_rebuild.train_lgbm',['--samples','forecast_rebuild/artifacts/samples/expanded_planned',
        '--tag','expanded_lgbm_planned_energy','--rounds','700'],'expanded_lgbm_planned_energy')


def planned_gpu():
    return run('forecast_rebuild.train_neural',['--samples','forecast_rebuild/artifacts/samples/expanded_planned',
        '--tag','expanded_residual_planned_energy','--family','residual_transformer','--steps','5000'],'expanded_residual_planned_energy')


def main():
    folder=ARTIFACTS/'campaign';folder.mkdir(parents=True,exist_ok=True)
    if not (ARTIFACTS/'samples/expanded_core/manifest.json').exists():raise ValueError('Expanded training samples not ready')
    verify_prior();held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));started=now()
    snapshot=folder/'source';snapshot.mkdir(exist_ok=True)
    pins={}
    for path in ROOT.glob('*.py'):
        (snapshot/path.name).write_bytes(path.read_bytes());pins[path.name]=sha(path)
    write(folder/'source_manifest.json',dict(created_at=started,files=pins))
    try:
        write(folder/'status.json',dict(state='training',stage='core_and_planning',started_at=started,pid=os.getpid(),
            automatic_retries=False,final_test_opened=False,system_sleep_inhibited=held))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(core_cpu),pool.submit(core_gpu),
                     pool.submit(run,'forecast_rebuild.prepare_plans',['--workers','2'],'prepare_public_plans')]
            completed=[]
            for future in as_completed(futures):
                completed.append(future.result())
                write(folder/'status.json',dict(state='training',stage='core_and_planning',completed=completed,updated_at=now(),
                    automatic_retries=False,final_test_opened=False,system_sleep_inhibited=held))
        run('forecast_rebuild.dataset',['--tag','expanded_planned','--planning'],'prepare_planned_samples')
        write(folder/'status.json',dict(state='training',stage='planned_candidates',updated_at=now(),automatic_retries=False))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(planned_cpu),pool.submit(planned_gpu)]):future.result()
        verify_prior()
        write(folder/'status.json',dict(state='training_completed',stage='awaiting_full_evaluation',started_at=started,ended_at=now(),
            candidates=6,final_test_opened=False,production_ready=False,automatic_retries=False))
    except BaseException as error:
        write(folder/'status.json',dict(state='failed',error=repr(error),updated_at=now(),automatic_retries=False,
                                       committed_checkpoints_preserved=True,original_models_modified=False));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
