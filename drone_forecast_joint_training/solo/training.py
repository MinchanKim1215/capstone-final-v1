from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import copy,json,multiprocessing as mp,os,time,re,shutil
import numpy as np
import torch
from torch import nn
from solo.contracts import (ROOT,SCHEMA,sha,object_hash,atomic_json,atomic_torch,sources,seed_all,
                            rng_state,restore_rng,device_for,deterministic,file_lock,load_plan)
from solo.model import Controller,RiskTransformer
from solo.data import EpisodeBatch,actor_inputs,prediction_loss
from solo.features import training_specs
from solo.rollout import initialize_worker,worker


def latest(folder):
    path=Path(folder)/'latest.json'
    if not path.exists():return None
    pointer=json.loads(path.read_text(encoding='utf-8'));checkpoint=ROOT/pointer['path']
    if sha(checkpoint)!=pointer['sha256']:raise ValueError('Committed checkpoint hash mismatch')
    return checkpoint,torch.load(checkpoint,map_location='cpu',weights_only=False)


def commit(folder,checkpoint,step):
    folder=Path(folder);path=folder/'checkpoints'/f's{step:07d}.pt'
    atomic_torch(path,checkpoint)
    atomic_json(folder/'latest.json',dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),step=step))
    return path


def execution_signature(plan,phase):
    return object_hash(dict(plan=object_hash(plan),source=sources(),phase=phase,schema=SCHEMA))


def collect(plan,snapshot,specs,folder,workers=4,deterministic_policy=False,phase='train'):
    """Completion order never changes training sample order; stop is polled by spawned workers."""
    if workers<1 or not specs:raise ValueError('At least one worker and episode are required')
    snapshot=Path(snapshot);folder=Path(folder);signature=execution_signature(plan,phase)
    # Eight full missions can use hundreds of MB, depending on duration and compressibility.
    if shutil.disk_usage(ROOT).free < max(512*1024**2,len(specs)*40*1024**2):
        raise OSError('Insufficient free disk space for this rollout batch; archive/prune committed raw rollouts first')
    ck=torch.load(snapshot,map_location='cpu',weights_only=False)
    if ck.get('source_sha256')!=sources():raise ValueError('Snapshot source differs from current code')
    tasks=[]
    for spec in specs:
        stem=f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
        tasks.append((str(snapshot),str(folder/stem),spec,signature,deterministic_policy))
    ctx=mp.get_context('spawn');stop=ctx.Event()
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize_worker,
                             initargs=(str(snapshot),signature,stop))
    completed={};futures=[]
    try:
        for task in tasks:futures.append(pool.submit(worker,task))
        for future in as_completed(futures):
            row=future.result()
            if row.get('stopped'):raise InterruptedError('Rollout stopped')
            key=(row['region'],row['direction'],row['kind'],row['seed']);completed[key]=row
            print(json.dumps(dict(stage=phase,complete=len(completed),total=len(tasks),region=row['region'],
                  CTR=row['search_connected_mean'],PDR=row['pdr'],mission_completed=row['mission_completed'])),flush=True)
    except BaseException:
        stop.set()
        for future in futures:future.cancel()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    rows=[completed[(s['region'],s['direction'],s['kind'],s['seed'])] for s in specs]
    prefixes=[Path(task[1]) for task in tasks]
    return prefixes,rows


def checkpoint(model,optimizer,config,updates,history,plan,**extras):
    return dict(format='standalone-transformer-ppo-v5',schema=SCHEMA,mode=config['mode'],model=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer else None,config=config,updates=updates,history=history,
        plan_sha256=object_hash(plan),source_sha256=sources(),rng=rng_state(),**extras)


def make_optimizer(model,config):
    groups=[]
    parts=[(model.base,config['actor_lr']),(model.adapter,config['actor_lr']),
           (model.risk,config['encoder_lr']),(model.risk.pred_head,config['predictor_lr']),
           (model.critic,config['critic_lr'])]
    for module,lr in parts:
        params=[p for p in (module.encoder_parameters() if module is model.risk else module.parameters()) if p.requires_grad]
        if params:groups.append(dict(params=params,lr=lr))
    return torch.optim.Adam(groups,eps=1e-5)


def move_optimizer(optimizer,device):
    for state in optimizer.state.values():
        for key,value in state.items():
            if torch.is_tensor(value):state[key]=value.cpu() if key=='step' and not optimizer.defaults.get('capturable',False) else value.to(device)


def assert_behavior(model,data,device,minibatch=32):
    largest=0.;model.eval()
    with torch.no_grad():
        for first in range(0,len(data),minibatch):
            b=data.batch(np.arange(first,min(first+minibatch,len(data))),device)
            logits,_=model.policy(*actor_inputs(b))
            logp=torch.distributions.Categorical(logits=logits).log_prob(b['chosen'])
            largest=max(largest,float((logp-b['logp']).abs().max().cpu()))
    tolerance=2e-4 if str(device).startswith('cuda') else 5e-5
    if not np.isfinite(largest) or largest>tolerance:
        raise ValueError(f'Behavior-policy mismatch ({largest:.3g}>{tolerance}); never update using stale cached neural features')
    return largest


def gradient_norm(module):
    values=[p.grad.detach().double().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(sum(values)).cpu()) if values else 0.


def ppo_update(model,optimizer,data,config,update,device):
    error=assert_behavior(model,data,device,config['minibatch'])
    generator=np.random.default_rng(config['seed']+update*1013)
    statistics=[];proof=dict(encoder_from_policy_max=0.,predictor_from_policy_max=0.)
    before={name:p.detach().cpu().clone() for name,p in model.risk.named_parameters()}
    actor_before={name:p.detach().cpu().clone() for name,p in model.base.named_parameters() if p.requires_grad}
    stop=False;model.train()
    for epoch in range(config['epochs']):
        permutation=generator.permutation(len(data))
        for begin in range(0,len(data),config['minibatch']):
            b=data.batch(permutation[begin:begin+config['minibatch']],device)
            logits,prediction=model.policy(*actor_inputs(b))
            distribution=torch.distributions.Categorical(logits=logits)
            logp=distribution.log_prob(b['chosen'])
            ratio=torch.exp(logp-b['logp'])
            if not torch.isfinite(ratio).all():raise FloatingPointError('Nonfinite importance ratio; update is not committed')
            advantage=b['advantage'][:,None]
            policy=-torch.minimum(ratio*advantage,ratio.clamp(.85,1.15)*advantage).mean()
            predicted_value=model.value(b['state'])
            value=((predicted_value-b['returns'])/data.value_scale).square().mean()
            entropy=distribution.entropy().mean()
            anchor=torch.distributions.Categorical(logits=b['teacher'].masked_fill(~b['mask'].bool(),-1e9))
            anchor_kl=torch.distributions.kl_divergence(distribution,anchor).mean()
            aux,labels=prediction_loss(prediction,b)
            loss=policy+.5*value-.003*entropy+.003*anchor_kl
            if model.mode=='joint':loss=loss+config['prediction_coef']*aux
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint loss; previous checkpoint preserved')
            # Measure gradients from PPO alone, rather than mislabelling supervised gradients as joint learning.
            if model.mode=='joint' and len(statistics)%10==0:
                parameters=list(model.risk.encoder_parameters())+list(model.risk.pred_head.parameters())
                gradients=torch.autograd.grad(policy,parameters,retain_graph=True,allow_unused=True)
                split=len(list(model.risk.encoder_parameters()))
                for key,part in [('encoder_from_policy_max',gradients[:split]),('predictor_from_policy_max',gradients[split:])]:
                    squares=[g.detach().double().square().sum() for g in part if g is not None]
                    norm=float(torch.sqrt(sum(squares)).cpu()) if squares else 0.
                    proof[key]=max(proof[key],norm)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=float(nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite gradient; previous checkpoint preserved')
            optimizer.step()
            with torch.no_grad():
                after_logits,_=model.policy(*actor_inputs(b))
                old=torch.distributions.Categorical(logits=b['old_logits'])
                new=torch.distributions.Categorical(logits=after_logits)
                exact_kl=float(torch.distributions.kl_divergence(old,new).mean().cpu())
            statistics.append([float(policy.detach().cpu()),float(value.detach().cpu()),float(aux.detach().cpu()),
                               float(entropy.detach().cpu()),exact_kl,labels])
            if exact_kl>config['target_kl']:stop=True;break
        if stop:break
    model.eval();means=np.mean(statistics,axis=0)
    changes={}
    for group,prefix in [('encoder','encoder'),('predictor','pred_head.')]:
        changes[group+'_change_l2']=float(np.sqrt(sum(float((p.detach().cpu()-before[name]).double().square().sum())
          for name,p in model.risk.named_parameters() if ((not name.startswith('pred_head.')) if group=='encoder' else name.startswith('pred_head.')))))
    changes['base_actor_change_l2']=float(np.sqrt(sum(float((p.detach().cpu()-actor_before[name]).double().square().sum())
          for name,p in model.base.named_parameters() if name in actor_before)))
    if model.mode!='joint' and any(changes[k]>0 for k in ['encoder_change_l2','predictor_change_l2']):raise ValueError('Frozen prediction arm changed')
    return dict(update=update,policy_loss=float(means[0]),value_loss=float(means[1]),prediction_loss=float(means[2]),
       entropy=float(means[3]),mean_kl=float(means[4]),maximum_kl=max(s[4] for s in statistics),epochs=epoch+1,
       minibatches=len(statistics),samples=len(data),behavior_logp_max_error=error,
       gradient_proof=proof,**changes)


DEFAULTS=dict(seed=2026102907,batch_episodes=8,minibatch=16,epochs=4,actor_lr=1e-4,
 encoder_lr=1e-5,predictor_lr=5e-5,critic_lr=3e-4,prediction_coef=.1,target_kl=.02,
 keep_rollouts=2,validation_every=16,patience=8)


def prune(folder,done,keep):
    if keep:
        for d in (folder/'rollouts').glob('u[0-9]*'):
            if d.is_dir() and int(d.name[1:])<=done-keep:
                for p in d.glob('*.npz'):p.unlink()


def bootstrap(episodes=192,workers=4,validation_cases=8,seed=2026102907):
    if min(episodes,workers,validation_cases)<1:raise ValueError('Positive episodes/workers/validation cases required')
    plan=load_plan();deterministic(True);folder=ROOT/'results/transformer/bootstrap';folder.mkdir(parents=True,exist_ok=True)
    with file_lock(folder/'run.lock'):
        snapshot=folder/'source.pt'
        if snapshot.exists():
            ck=torch.load(snapshot,weights_only=False,map_location='cpu')
            if ck['source_sha256']!=sources() or ck['plan_sha256']!=object_hash(plan) or ck['config']['seed']!=seed:raise ValueError('Bootstrap code/data/seed changed')
        else:
            seed_all(seed);model=Controller('physics');config=dict(DEFAULTS,mode='physics',seed=seed)
            atomic_torch(snapshot,checkpoint(model,None,config,0,[],plan,device='cpu'))
        specs=training_specs(plan,seed,0,episodes)
        paths,rows=collect(plan,snapshot,specs,folder/'train',workers,False,'bootstrap_train')
        from solo.evaluate import scenario_subset
        vspecs=scenario_subset(plan['validation'],validation_cases)
        vpaths,vrows=collect(plan,snapshot,vspecs,folder/'validation',workers,True,'bootstrap_validation')
        manifest=dict(source_sha256=sources(),plan_sha256=object_hash(plan),seed=seed,
            train_paths=[p.relative_to(ROOT).as_posix() for p in paths],train_hashes=[r['array_sha256'] for r in rows],
            validation_paths=[p.relative_to(ROOT).as_posix() for p in vpaths],validation_hashes=[r['array_sha256'] for r in vrows],
            train_episodes=len(rows),validation_episodes=len(vrows),lightgbm_used=False,gru_used=False)
        atomic_json(folder/'complete.json',manifest)
        return manifest


def pretrain(steps=3000,batch_links=256,device='auto',seed=2026102907,save_every=25,validate_every=200,patience=8):
    from solo.data import PredictorPool,link_loss
    from solo.evaluate import score_predictor
    if min(steps,batch_links,save_every,validate_every)<1 or patience<0:raise ValueError('Invalid pretraining settings')
    plan=load_plan();device=device_for(device);deterministic(True)
    manifest=json.loads((ROOT/'results/transformer/bootstrap/complete.json').read_text())
    if manifest['source_sha256']!=sources() or manifest['plan_sha256']!=object_hash(plan):raise ValueError('Bootstrap provenance changed')
    folder=ROOT/'results/transformer/pretrain';folder.mkdir(parents=True,exist_ok=True)
    with file_lock(folder/'run.lock'):
        found=latest(folder);seed_all(seed);model=RiskTransformer().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        rng=np.random.default_rng(seed);done=0;history=[];validations=[];best_score=None;stale=0
        if found:
            _,ck=found
            if ck['source_sha256']!=sources() or ck['plan_sha256']!=object_hash(plan):raise ValueError('Pretraining code/data changed')
            if (ck['batch_links'],ck['seed'],ck['validate_every'],ck['patience'])!=(batch_links,seed,validate_every,patience):raise ValueError('Pretraining resume configuration changed')
            for key in ['train_paths','train_hashes','validation_paths','validation_hashes']:
                if manifest[key][:len(ck['manifest'][key])]!=ck['manifest'][key]:raise ValueError('Supervised corpus must be an append-only extension')
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);move_optimizer(optimizer,device)
            restore_rng(ck['rng']);rng.bit_generator.state=ck['sampler_rng'];done=ck['steps'];history=ck['history'];validations=ck['validations'];best_score=ck['best_score'];stale=ck['stale']
            if ck['manifest']!=manifest:
                stale=0
                if ck['manifest']['validation_paths']!=manifest['validation_paths']:best_score=None;validations=[]
        pool=PredictorPool([ROOT/p for p in manifest['train_paths']])
        def save(step):
            return commit(folder,dict(format='standalone-transformer-pretrain-v5',model=model.state_dict(),optimizer=optimizer.state_dict(),
                steps=step,history=history,validations=validations,best_score=best_score,stale=stale,manifest=manifest,
                batch_links=batch_links,seed=seed,validate_every=validate_every,patience=patience,
                source_sha256=sources(),plan_sha256=object_hash(plan),rng=rng_state(),sampler_rng=rng.bit_generator.state,device=str(device)),step)
        if not found:save(0)
        for step in range(done+1,steps+1):
            model.train();batch,importance=pool.sample(rng,batch_links,device)
            _,probability=model.encode_links(batch['values'],batch['masks'],batch['intents'],batch['geometry'])
            loss=link_loss(probability,batch,importance)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite predictor loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite predictor gradient')
            optimizer.step();history.append(dict(step=step,loss=float(loss.detach().cpu())))
            improved=False
            if step%validate_every==0 or step==steps:
                state=rng_state()
                metrics=score_predictor(model,[ROOT/p for p in manifest['validation_paths']],device,limit_decisions=96)
                restore_rng(state)
                a=metrics['risk_5s']['average_precision'];b=metrics['onset_3to7s']['average_precision']
                score=.6*(a or 0)+.4*(b or 0)
                improved=best_score is None or score>best_score+1e-5
                if improved:best_score=score;stale=0
                else:stale+=1
                validations.append(dict(step=step,score=score,metrics=metrics,improved=improved))
                print(json.dumps(dict(stage='pretrain_validation',step=step,score=score,risk5_AP=a,onset37_AP=b,stale=stale)),flush=True)
            if step%save_every==0 or step==steps or improved or (patience and stale>=patience):
                path=save(step);atomic_json(folder/'history.json',history)
                if improved:atomic_json(folder/'best.json',dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),step=step,score=best_score))
                print(json.dumps(dict(stage='pretrain_committed',step=step,target=steps,loss=history[-1]['loss'])),flush=True)
            if patience and stale>=patience:
                print(json.dumps(dict(stage='pretrain_early_stop',step=step,reason='Validation plateau; best checkpoint retained')),flush=True);break
        return best_path(folder)


def best_path(folder):
    path=Path(folder)/'best.json'
    if path.exists():
        p=json.loads(path.read_text());file=ROOT/p['path']
        if sha(file)!=p['sha256']:raise ValueError('Best checkpoint hash changed')
        return file
    found=latest(folder)
    if not found:raise ValueError('No saved checkpoint')
    return found[0]


def validate_training(snapshot,folder,plan,config,workers):
    from solo.evaluate import evaluate_checkpoint
    state=rng_state()
    try:result=evaluate_checkpoint(snapshot,'validation',workers=workers,cases=8)
    finally:restore_rng(state)
    file=folder/'validation_history.json';records=json.loads(file.read_text()) if file.exists() else []
    ck=torch.load(snapshot,weights_only=False,map_location='cpu');step=ck['updates']
    if records and records[-1]['update']>=step:return records
    c=result['communication'];m=c['means'];feasible=c['complete']==c['episodes'] and c['safety_passed'] and c['max_duration_ratio']<=1.15
    score=[m['search_connected_mean'],m['pdr'],-c['longest_outage_s']]
    eligible=[r for r in records if r['feasible']]
    improved=feasible and (not eligible or tuple(score)>max(tuple(r['score']) for r in eligible))
    records.append(dict(update=step,checkpoint=snapshot.relative_to(ROOT).as_posix(),sha256=sha(snapshot),score=score,feasible=feasible,improved=improved,
        source_sha256=sources(),plan_sha256=object_hash(plan)))
    atomic_json(file,records)
    if improved:atomic_json(folder/'best.json',dict(path=snapshot.relative_to(ROOT).as_posix(),sha256=sha(snapshot),step=step,score=score))
    return records


def train(name='transformer_joint',mode='joint',updates=256,workers=4,device='auto',overrides=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):raise ValueError('Use a safe short run name')
    if min(updates,workers)<1:raise ValueError('Positive updates/workers required')
    plan=load_plan();device=device_for(device);deterministic(True)
    if (ROOT/'results/transformer/test_access.json').exists():raise ValueError('Final test opened. Start a new study with genuinely unseen test maps before further model development.')
    folder=ROOT/'results/transformer/train'/name;folder.mkdir(parents=True,exist_ok=True)
    with file_lock(folder/'run.lock'):
        found=latest(folder)
        if found:
            snapshot,ck=found;config=ck['config']
            if mode!=config['mode'] or ck['source_sha256']!=sources() or ck['plan_sha256']!=object_hash(plan):raise ValueError('Resume architecture/code/data differs')
            for k,v in (overrides or {}).items():
                if v is not None and config.get(k)!=v:raise ValueError('Resume configuration changed: '+k)
            if config.get('risk_init') and sha(ROOT/config['risk_init'])!=config['risk_init_sha256']:raise ValueError('Initial predictor file changed')
            seed_all(config['seed']);model=Controller(mode).to(device);model.load_state_dict(ck['model']);optimizer=make_optimizer(model,config)
            optimizer.load_state_dict(ck['optimizer']);move_optimizer(optimizer,device);restore_rng(ck['rng']);history=ck['history'];done=ck['updates']
        else:
            config=dict(DEFAULTS,mode=mode);config.update({k:v for k,v in (overrides or {}).items() if v is not None})
            for k in ['batch_episodes','minibatch','epochs']:
                if not isinstance(config[k],int) or config[k]<1:raise ValueError('Positive integer required: '+k)
            for k in ['keep_rollouts','validation_every','patience']:
                if not isinstance(config[k],int) or config[k]<0:raise ValueError('Nonnegative integer required: '+k)
            for k in ['actor_lr','encoder_lr','predictor_lr','critic_lr','prediction_coef','target_kl']:
                if not np.isfinite(config[k]) or config[k]<=0:raise ValueError('Finite positive setting required: '+k)
            seed_all(config['seed']);model=Controller(mode).to(device)
            if mode!='physics':
                path=best_path(ROOT/'results/transformer/pretrain');risk=torch.load(path,map_location='cpu',weights_only=False)
                if risk['source_sha256']!=sources() or risk['plan_sha256']!=object_hash(plan):raise ValueError('Predictor source/data changed')
                model.risk.load_state_dict(risk['model']);config.update(risk_init=path.relative_to(ROOT).as_posix(),risk_init_sha256=sha(path))
            optimizer=make_optimizer(model,config);history=[];done=0
            snapshot=commit(folder,checkpoint(model,optimizer,config,0,history,plan,device=str(device)),0)
        print(json.dumps(dict(stage='training',run=name,mode=mode,resume_update=done,target_updates=updates,device=str(device),standalone_transformer=True,lightgbm_used=False)),flush=True)
        if done and config['validation_every'] and done%config['validation_every']==0:
            validate_training(snapshot,folder,plan,config,workers)
        for update in range(done+1,updates+1):
            records=json.loads((folder/'validation_history.json').read_text()) if (folder/'validation_history.json').exists() else []
            if records and config['patience']:
                last=max((i for i,r in enumerate(records) if r['improved']),default=-1)
                if len(records)-last-1>=config['patience']:
                    print(json.dumps(dict(stage='training_early_stop',run=name,reason='Validation plateau; best checkpoint retained')),flush=True);break
            started=time.perf_counter();specs=training_specs(plan,config['seed'],(update-1)*config['batch_episodes'],config['batch_episodes'])
            paths,rows=collect(plan,snapshot,specs,folder/'rollouts'/f'u{update:07d}',workers,False,'train')
            data=EpisodeBatch(paths);stats=ppo_update(model,optimizer,data,config,update,device)
            stats.update(episodes=len(rows),complete=sum(r['mission_completed'] for r in rows),CTR=float(np.mean([r['search_connected_mean'] for r in rows])),
                PDR=float(np.mean([r['pdr'] for r in rows])),longest_outage_s=max(r['outage_max_s'] for r in rows),
                safety_passed=all(r['audit']['passed'] for r in rows),minimum_separation_m=min(r['audit']['minimum_swept_separation_m'] for r in rows),wall_s=time.perf_counter()-started)
            history.append(stats);snapshot=commit(folder,checkpoint(model,optimizer,config,update,history,plan,device=str(device)),update)
            atomic_json(folder/'history.json',history);atomic_json(folder/'progress.json',dict(committed_updates=update,training_episodes=update*config['batch_episodes'],last=stats,checkpoint=snapshot.relative_to(ROOT).as_posix()))
            prune(folder,update,config['keep_rollouts']);print(json.dumps(dict(stage='committed_update',run=name,**stats)),flush=True);del data
            if config['validation_every'] and update%config['validation_every']==0:validate_training(snapshot,folder,plan,config,workers)
        return snapshot
