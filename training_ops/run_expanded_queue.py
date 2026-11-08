"""After v6 finishes: bind weights, verify v7, then run the added-map study."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import ctypes
import math
import msvcrt
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from adaptive_joint_expanded.contracts import read_json,atomic_json,sha,digest,lock
from adaptive_joint_expanded.initialization import parent_finished,PARENT_RUN,PARENT,BINDING

QUEUE=ROOT/'additional_maps_20261030/queue'
STUDY=ROOT/'adaptive_joint_expanded'
NEW_RUN='adaptive_v7_expanded100_20261031'
PYTHON=ROOT/'drone_forecast_joint_training/.venv/Scripts/python.exe'


def now():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def held(path):
    if not path.exists():return False
    with path.open('r+b') as f:
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:return True
        msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        return False


def dependency():
    p=read_json(PARENT_RUN/'status.json')
    active=held(PARENT_RUN/'run.lock')
    if parent_finished(p) and not active:return 'ready',p
    if p.get('state')=='running' and active:return 'waiting_for_current_training',p
    if parent_finished(p) and active:return 'waiting_for_parent_exit',p
    return 'blocked_parent_interrupted_or_failed',p


def bind_parent():
    if (STUDY/'artifacts/runs').exists():raise ValueError('Expanded study already has a training run')
    p=read_json(PARENT/'artifacts/protocol.json')
    for name,expected in p['source_sha256'].items():
        if sha(PARENT/name)!=expected:raise ValueError('Parent source integrity failed')
    candidates=[]
    for record in read_json(PARENT_RUN/'validation_history.json'):
        if not record['feasible']:continue
        path=PARENT_RUN/'validation'/f"u{record['step']:07d}"/'summary.json'
        summary=read_json(path)
        score=summary['prediction']['next_onset_lt5s']['average_precision']
        if score is not None and math.isfinite(score):candidates.append((score,record['step'],record,path,summary))
    if not candidates:raise ValueError('No mission-feasible evaluated predictor exists for transfer')
    score,step,record,summary_path,summary=max(candidates,key=lambda r:(r[0],r[1]))
    checkpoint=Path(record['checkpoint'])
    if checkpoint.parent.resolve()!=(PARENT_RUN/'checkpoints').resolve():raise ValueError('Checkpoint escaped parent run')
    if sha(checkpoint)!=record['sha256'] or summary['checkpoint_sha256']!=record['sha256']:
        raise ValueError('Parent validation/checkpoint hash differs')
    if summary['protocol_sha256']!=digest(p):raise ValueError('Parent validation protocol differs')
    binding=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=record['sha256'],step=step,
                 format=p['version'],protocol_sha256=digest(p),bound_at=now(),
                 source_validation=str(summary_path.resolve()),source_validation_sha256=sha(summary_path),
                 selection_rule='Highest next-5s AP among mission-feasible parent selection evaluations; predictive transfer initialization, not an independent performance claim',
                 selected_development_ap=score,fresh_optimizer_for_new_dataset=True,
                 parent_communication_best_step=read_json(PARENT_RUN/'best.json')['step'])
    atomic_json(BINDING,binding)
    return binding


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    state,parent=dependency()
    if args.check:
        print(dict(dependency=state,parent=parent,expanded_run_exists=(STUDY/'artifacts/runs').exists()))
        return 0
    QUEUE.mkdir(parents=True,exist_ok=True)
    with lock(QUEUE/'queue.lock'):
        api=ctypes.WinDLL('kernel32',use_last_error=True).SetThreadExecutionState
        api.argtypes=[ctypes.c_uint32];api.restype=ctypes.c_uint32
        record=dict(pid=os.getpid(),started_at=now(),source_script_sha256=sha(__file__),
                    auto_retry=False,next_run=NEW_RUN,current_training_unmodified=True,
                    system_sleep_inhibited=bool(api(0x80000001)))
        def save(stage,**fields):
            record.update(stage=stage,updated_at=now(),**fields)
            atomic_json(QUEUE/'status.json',record)
        try:
            if not read_json(STUDY/'artifacts/dataset_ready.json')['ready']:
                raise ValueError('Added-map dataset acceptance is incomplete')
            while True:
                if (QUEUE/'cancel_requested.json').exists():save('cancelled');return 0
                state,parent=dependency();save(state,parent=parent)
                if state=='ready':break
                if state.startswith('blocked'):return 3
                time.sleep(30)
            binding=bind_parent();save('binding_completed',parent_initialization=binding)
            commands=[('prepare',['prepare','--refresh']),('assets',['assets','--workers','4']),
                      ('preflight',['preflight']),('train',['train','--run',NEW_RUN])]
            env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            for stage,arguments in commands:
                stdout=QUEUE/f'{stage}.stdout.log';stderr=QUEUE/f'{stage}.stderr.log'
                save(stage,stdout=str(stdout),stderr=str(stderr))
                with stdout.open('ab',buffering=0) as out,stderr.open('ab',buffering=0) as err:
                    child=subprocess.Popen([str(PYTHON),'-u','-B','-m','adaptive_joint_expanded.run',*arguments],
                                           cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                    save(stage,child_pid=child.pid)
                    code=child.wait()
                if code:raise RuntimeError(f'Expanded {stage} failed, exit {code}; see {stderr}')
            save('completed',result=read_json(STUDY/'artifacts/runs'/NEW_RUN/'status.json'))
            return 0
        except BaseException as error:
            save('failed',error=repr(error));raise
        finally:api(0x80000000)


if __name__=='__main__':sys.exit(main())
