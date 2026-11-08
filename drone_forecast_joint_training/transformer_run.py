#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(name,'1')
import argparse,json,sys,multiprocessing,signal,traceback,re
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(description='Standalone Transformer outage prediction + joint PPO. Version 5.')
    s=p.add_subparsers(dest='command',required=True)
    for name in ['doctor','selftest','verify']:s.add_parser(name)
    c=s.add_parser('cuda-check');c.add_argument('--device',default='cuda')
    c=s.add_parser('preflight');c.add_argument('--workers',type=int,default=4)
    c=s.add_parser('collect');c.add_argument('--episodes',type=int,default=192);c.add_argument('--validation-cases',type=int,default=8);c.add_argument('--workers',type=int,default=4)
    c=s.add_parser('pretrain');c.add_argument('--steps',type=int,default=3000);c.add_argument('--batch-links',type=int,default=256)
    c.add_argument('--validate-every',type=int,default=200);c.add_argument('--patience',type=int,default=8);c.add_argument('--device',default='auto')
    c=s.add_parser('train');c.add_argument('--run',default='transformer_main');c.add_argument('--mode',choices=['physics','fixed','joint'],default='joint')
    c.add_argument('--updates',type=int,default=256);c.add_argument('--workers',type=int,default=4);c.add_argument('--device',default='auto')
    for key in ['seed','batch_episodes','minibatch','epochs','keep_rollouts','validation_every','patience']:c.add_argument('--'+key.replace('_','-'),type=int,default=None)
    for key in ['actor_lr','encoder_lr','predictor_lr','critic_lr','prediction_coef','target_kl']:c.add_argument('--'+key.replace('_','-'),type=float,default=None)
    c=s.add_parser('pipeline');c.add_argument('--run',default='transformer_main');c.add_argument('--episodes',type=int,default=192)
    c.add_argument('--pretrain-steps',type=int,default=3000);c.add_argument('--updates',type=int,default=256);c.add_argument('--workers',type=int,default=4);c.add_argument('--device',default='cuda')
    c=s.add_parser('evaluate');c.add_argument('--run',default=None);c.add_argument('--checkpoint',default=None);c.add_argument('--cases',type=int,default=0);c.add_argument('--workers',type=int,default=4)
    c=s.add_parser('select');c.add_argument('--runs',nargs='+',default=['transformer_main']);c.add_argument('--cases',type=int,default=0);c.add_argument('--workers',type=int,default=4)
    c=s.add_parser('benchmark-lightgbm')
    c=s.add_parser('finaltest');c.add_argument('--workers',type=int,default=4);c.add_argument('--freeze',default='results/transformer/frozen_selection.json')
    return p


def main():
    args=parser().parse_args()
    from solo.contracts import ROOT,atomic_json,load_plan,sha,sources,object_hash
    os.chdir(ROOT)
    if args.command in ['doctor','verify','selftest','cuda-check']:
        from solo.checks import doctor,verify,selftest,cuda_check
        if args.command=='selftest':return 0 if selftest() else 1
        result=cuda_check(args.device) if args.command=='cuda-check' else (doctor() if args.command=='doctor' else verify())
        print(json.dumps(result,indent=2,ensure_ascii=False));atomic_json(ROOT/'results/transformer/verification'/(args.command+'.json'),result);return 0
    if args.command=='preflight':
        from solo.preflight import run
        result=run(args.workers);print(json.dumps({k:v for k,v in result.items() if k!='rows'}));return 0
    from solo.training import bootstrap,pretrain,train,best_path,latest,DEFAULTS
    if args.command=='collect':bootstrap(args.episodes,args.workers,args.validation_cases)
    elif args.command=='pretrain':pretrain(args.steps,args.batch_links,args.device,validate_every=args.validate_every,patience=args.patience)
    elif args.command=='train':train(args.run,args.mode,args.updates,args.workers,args.device,{k:getattr(args,k) for k in DEFAULTS})
    elif args.command=='pipeline':
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.run):raise ValueError('Invalid run name')
        from solo.checks import cuda_check,doctor
        from solo.preflight import run as preflight
        if args.device.startswith('cuda'):print(json.dumps(cuda_check(args.device)),flush=True)
        else:doctor()
        preflight(args.workers);bootstrap(args.episodes,args.workers);risk=pretrain(args.pretrain_steps,device=args.device)
        import torch
        from solo.model import RiskTransformer
        from solo.evaluate import score_predictor,select
        predictor=RiskTransformer();predictor.load_state_dict(torch.load(risk,map_location='cpu',weights_only=False)['model'])
        manifest=json.loads((ROOT/'results/transformer/bootstrap/complete.json').read_text())
        metrics=score_predictor(predictor,[ROOT/p for p in manifest['validation_paths']],limit_decisions=96)
        m=metrics['risk_5s']
        atomic_json(ROOT/'results/transformer/predictor_readiness.json',dict(metrics=metrics,checkpoint=risk.relative_to(ROOT).as_posix(),real_flight_approved=False))
        if not m.get('positives') or m['average_precision']<=m['positive_fraction']:
            raise ValueError('Predictor has not exceeded the constant-score AP baseline on validation. Check predictor_readiness.json and extend/fix supervised learning before a long PPO run.')
        train(args.run,'joint',args.updates,args.workers,args.device)
        select([args.run],args.workers,0)
    elif args.command=='evaluate':
        from solo.evaluate import evaluate_checkpoint,reference_checkpoint
        if args.run and args.checkpoint:raise ValueError('Choose run OR checkpoint')
        path=best_path(ROOT/'results/transformer/train'/args.run) if args.run else (Path(args.checkpoint) if args.checkpoint else reference_checkpoint())
        evaluate_checkpoint(path,workers=args.workers,cases=args.cases)
    elif args.command=='select':
        from solo.evaluate import select
        select(args.runs,args.workers,args.cases)
    elif args.command=='benchmark-lightgbm':
        from solo.evaluate import compare_lightgbm
        print(json.dumps(compare_lightgbm(),indent=2))
    elif args.command=='finaltest':
        from solo.evaluate import evaluate_checkpoint
        freeze=ROOT/args.freeze;s=json.loads(freeze.read_text())
        for path in dict.fromkeys([s['reference_checkpoint'],s['selected_checkpoint']]):evaluate_checkpoint(ROOT/path,'test',args.workers,0,freeze)
    return 0


if __name__=='__main__':
    multiprocessing.freeze_support();signal.signal(signal.SIGINT,signal.default_int_handler);signal.signal(signal.SIGTERM,signal.default_int_handler)
    try:sys.exit(main())
    except KeyboardInterrupt:
        print('Stopped. Previous committed checkpoint preserved. Repeat the same command to resume.',file=sys.stderr);sys.exit(130)
    except Exception as e:
        print('ERROR:',str(e),file=sys.stderr);traceback.print_exc();sys.exit(1)
