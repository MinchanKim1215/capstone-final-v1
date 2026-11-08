"""Explicit commands. No import, doctor, prepare or preflight starts a campaign."""
from pathlib import Path
import argparse
import contextlib
import json
import multiprocessing
import os
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from . import ROOT,LEGACY,ARTIFACTS,VERSION
from .contracts import DEFAULT,atomic_json,read_json,sha,digest,legacy_hashes


def parser():
    p=argparse.ArgumentParser(description='v6 deployment selection + 1-second next-event predictor + joint PPO')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor');sub.add_parser('selftest');sub.add_parser('plan')
    c=sub.add_parser('prepare');c.add_argument('--config',default=str(ROOT/'config.json'));c.add_argument('--refresh',action='store_true')
    c=sub.add_parser('preflight');c.add_argument('--quick',action='store_true',help='A quick check cannot grant long-training readiness')
    c=sub.add_parser('assets');c.add_argument('--workers',type=int,default=4)
    c=sub.add_parser('train');c.add_argument('--run',default='adaptive_v6_20261030');c.add_argument('--updates',type=int);c.add_argument('--resume',action='store_true')
    c=sub.add_parser('evaluate');c.add_argument('--checkpoint',required=True);c.add_argument('--cases',type=int,default=8);c.add_argument('--paired',action='store_true')
    c=sub.add_parser('replay');c.add_argument('--checkpoint',required=True);c.add_argument('--rollouts',required=True);c.add_argument('--device',default='cuda')
    c=sub.add_parser('calibrate');c.add_argument('--checkpoint',required=True)
    c=sub.add_parser('confirm');c.add_argument('--checkpoint',required=True);c.add_argument('--alarm',required=True)
    c=sub.add_parser('freeze');c.add_argument('--run',required=True);c.add_argument('--alarm',required=True)
    c=sub.add_parser('finaltest');c.add_argument('--freeze',required=True)
    return p


def doctor():
    import shutil
    import torch,numpy,scipy,sklearn,shapely
    from .experiment import development_splits
    existing=ARTIFACTS/'legacy_before.json'
    preserved=not existing.exists() or read_json(existing)==legacy_hashes()
    return dict(version=VERSION,python=sys.version,executable=sys.executable,
                torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,shapely=shapely.__version__,
                original_preserved=preserved,new_artifacts=str(ARTIFACTS),
                free_gib=round(shutil.disk_usage(ROOT).free/1024**3,2),
                development_fold_maps={k:len(v) for k,v in development_splits().items()},
                old_run_untouched=True,long_training_started=(ARTIFACTS/'runs').exists())


def protocol():
    path=ARTIFACTS/'protocol.json'
    if not path.exists():raise ValueError('Run prepare first')
    return read_json(path)


def training_checkpoint(path,p):
    from .experiment import load_checkpoint
    _,_,ck=load_checkpoint(path,p,'cpu')
    if ck.get('purpose')!='training' or ck['step']<1:
        raise ValueError('A preflight/untrained checkpoint is not performance evidence')
    return ck


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for s in self.streams:s.write(value);s.flush()
    def flush(self):
        for s in self.streams:s.flush()


def main():
    args=parser().parse_args()
    if args.command=='doctor':
        result=doctor();atomic_json(ARTIFACTS/'environment.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if args.command=='selftest':
        from .preflight import unit_tests
        result=unit_tests();print(result['output']);return 0 if result['passed'] else 1
    if args.command=='prepare':
        from .experiment import prepare
        c=read_json(args.config)
        p=prepare(c,args.refresh)
        print(json.dumps(dict(protocol=str(ARTIFACTS/'protocol.json'),sha256=digest(p),
                              prepared=True,long_training_started=False),ensure_ascii=False));return 0
    p=protocol()
    from .experiment import assert_protocol
    if args.command!='finaltest':assert_protocol(p)
    if args.command=='plan':
        print(json.dumps(dict(config=p['config'],deployments=p['deployments'],
              development_geography=p['development_geography'],warm_predictor=p['warm_predictor'],
              predictor_targets=p['targets'],ready=read_json(ARTIFACTS/'readiness.json') if (ARTIFACTS/'readiness.json').exists() else None,
              campaign_will_not_start_from_this_command=True),ensure_ascii=False,indent=2));return 0
    if args.command=='assets':
        from .experiment import build_assets
        if args.workers<1:raise ValueError('Positive asset worker count required')
        result=build_assets(p,args.workers)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))
        return 0 if result['ready'] else 1
    if args.command=='preflight':
        from .preflight import run
        result=run(p,not args.quick);print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['ready'] else 1
    if args.command=='train':
        from .driver import train
        folder=ARTIFACTS/'runs'/args.run
        # Validate the path before creating log folders from CLI text.
        import re
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.run):raise ValueError('Invalid run name')
        logs=ARTIFACTS/'logs';logs.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
        import uuid
        token=uuid.uuid4().hex
        os.environ['ADAPTIVE_RUN_TOKEN']=token
        with (logs/f'{args.run}_{stamp}.log').open('w',encoding='utf-8') as stream:
            with contextlib.redirect_stdout(Tee(sys.stdout,stream)),contextlib.redirect_stderr(Tee(sys.stderr,stream)):
                try:
                    result=train(args.run,p,args.updates,args.resume)
                except BaseException as error:
                    attempt=folder/'attempt.json'
                    if attempt.exists() and read_json(attempt).get('token')==token:
                        atomic_json(folder/'status.json',dict(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),previous_committed_checkpoint_preserved=True))
                    raise
        print(result);return 0
    if args.command in ('evaluate','calibrate','confirm'):
        from .driver import evaluate,read_episodes
        from .experiment import collect,validation_specs
        from .metrics import fit_alarm,summarize
        checkpoint=Path(args.checkpoint);training_checkpoint(checkpoint,p)
        tag=sha(checkpoint)[:16]
        if args.command=='evaluate':
            result=evaluate(checkpoint,p,ARTIFACTS/'evaluation'/tag/'selection',count=args.cases,paired=args.paired)
        elif args.command=='calibrate':
            folder=ARTIFACTS/'evaluation'/tag/'calibration'
            rows=collect(checkpoint,p,validation_specs(0,'calibration'),folder/'rollouts',deterministic=True,load_arrays=False)
            result=fit_alarm(list(read_episodes(rows)))
            result.update(checkpoint_sha256=sha(checkpoint),protocol_sha256=digest(p),source_scope='calibration')
            atomic_json(folder/'alarm.json',result)
        else:
            alarm=read_json(args.alarm)
            if alarm['checkpoint_sha256']!=sha(checkpoint) or alarm['protocol_sha256']!=digest(p):raise ValueError('Alarm belongs to another frozen model')
            folder=ARTIFACTS/'evaluation'/tag/'confirmation'
            rows=collect(checkpoint,p,validation_specs(0,'confirmation'),folder/'rollouts',deterministic=True,load_arrays=False)
            threshold=alarm['selected']['threshold'] if alarm['enabled'] else .5
            result=summarize(read_episodes(rows),threshold)
            result['alarm_enabled']=alarm['enabled'];result['scope']='Held-out development fold; historically seen by older experiments, not the final test'
            atomic_json(folder/'summary.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if args.command=='replay':
        training_checkpoint(args.checkpoint,p)
        from .replay import compare
        result=compare(args.checkpoint,p,args.rollouts,args.device)
        path=ARTIFACTS/'prediction_replay'/sha(args.checkpoint)[:16]/'summary.json'
        atomic_json(path,result);print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if args.command=='freeze':
        folder=ARTIFACTS/'runs'/args.run
        status=read_json(folder/'status.json')
        if status['state'] not in ('completed','early_stopped'):raise ValueError('Finish the campaign before freezing')
        best=read_json(folder/'best.json');checkpoint=Path(best['checkpoint'])
        training_checkpoint(checkpoint,p)
        alarm=read_json(args.alarm)
        if alarm['checkpoint_sha256']!=sha(checkpoint) or alarm['protocol_sha256']!=digest(p):raise ValueError('Calibration model mismatch')
        result=dict(protocol_sha256=digest(p),checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),
                    alarm=str(Path(args.alarm).resolve()),alarm_sha256=sha(args.alarm),run=args.run,test_used=False)
        path=ARTIFACTS/'frozen_selection.json';atomic_json(path,result);print(path);return 0
    if args.command=='finaltest':
        # This command is deliberately never called during preparation/preflight.
        assert_protocol(p,allow_final_test=True)
        freeze=Path(args.freeze).resolve();f=read_json(freeze)
        if f['protocol_sha256']!=digest(p) or sha(f['checkpoint'])!=f['checkpoint_sha256'] or sha(f['alarm'])!=f['alarm_sha256']:
            raise ValueError('Frozen model/calibration changed')
        marker=dict(freeze_sha256=sha(freeze),protocol_sha256=digest(p),opened_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat())
        path=ARTIFACTS/'test_access.json'
        if path.exists():
            old=read_json(path)
            if old['freeze_sha256']!=marker['freeze_sha256']:raise ValueError('A different final test is already open')
        else:atomic_json(path,marker)
        os.environ['ADAPTIVE_FINAL_TEST_FREEZE']=str(freeze)
        from .contracts import plan
        from .driver import evaluate,read_episodes
        from .metrics import summarize
        result=evaluate(f['checkpoint'],p,ARTIFACTS/'final_test',scope='sealed_test',paired=True,
                        explicit_specs=plan()['test'],allow_final_test=True)
        alarm=read_json(f['alarm']);threshold=alarm['selected']['threshold'] if alarm['enabled'] else .5
        # The paired report describes the frozen selector; apply the pre-frozen
        # alarm to its actual selected rows, never fit thresholds on test truth.
        rows=[]
        for metadata in result['selected_case_metadata']:
            row=read_json(metadata);row['array_path']=str(Path(metadata).with_suffix('.npz'));rows.append(row)
        result['frozen_alarm_evaluation']=summarize(read_episodes(rows),threshold)
        result['frozen_alarm_evaluation']['scope']='Sealed test; alarm threshold frozen before test'
        result.update(scope='Sealed test opened after model and alarm freeze',alarm_enabled=alarm['enabled'],freeze_sha256=sha(freeze))
        atomic_json(ARTIFACTS/'final_test/summary.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    raise ValueError('Unknown command')


if __name__=='__main__':
    multiprocessing.freeze_support()
    try:raise SystemExit(main())
    except Exception as error:
        print('ERROR:',error,file=sys.stderr);traceback.print_exc();raise SystemExit(1)
