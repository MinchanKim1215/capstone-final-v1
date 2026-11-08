"""Fine-tune the action ranker on newly simulated half-second trajectories."""
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from forecast_rebuild.contracts import now, read, sha, write
from . import ARTIFACTS
from .actor import ActorCritic
from .protocol import SEED, status
from .rollouts import collect, job

GAMMA_TICK = .99**(.1/3.)
LAMBDA_TICK = .95**(.1/3.)
PARTS = ('states','options','priors','valid','actions','logp')


def write_checkpoint(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_suffix('.pending')
    torch.save(value,pending)
    pending.replace(path)


def pin(path):
    return dict(path=str(path),sha256=sha(path))


def aggregate(rows):
    data = {name:[] for name in PARTS+('advantage','return')}
    for row in rows:
        if sha(row['archive_path']) != row['archive_sha256']:
            raise ValueError('Changed PPO rollout')
        with np.load(row['archive_path'],allow_pickle=False) as arrays:
            frames = arrays['action_frames'].astype(int)
            if not len(frames):
                continue
            reward, value = arrays['rewards'], arrays['values']
            stops = np.r_[frames[1:],len(reward)]
            advantages = np.zeros(len(frames))
            gae = 0.
            for i in reversed(range(len(frames))):
                start, stop = int(frames[i]),int(stops[i])
                n = stop-start
                if n <= 0:
                    raise ValueError('Invalid action decision timing')
                total = float(reward[start:stop] @ (GAMMA_TICK**np.arange(n)))
                following = float(value[i+1]) if i+1<len(frames) else 0.
                delta = total+(GAMMA_TICK**n)*following-float(value[i])
                gae = delta+((GAMMA_TICK*LAMBDA_TICK)**n)*gae
                advantages[i] = gae
            for name in PARTS:
                data[name].append(arrays[name])
            data['advantage'].append(advantages)
            data['return'].append(advantages+value)
    if sum(len(values) for values in data['actions']) < 30:
        raise ValueError('Fewer than 30 eligible policy decisions; insufficient PPO experience')
    merged = {key:np.concatenate(value) for key,value in data.items()}
    a = merged['advantage']
    merged['advantage'] = (a-a.mean())/max(1e-8,a.std())
    return merged


def optimize(policy, optimizer, data, device, update):
    tensors = {key:torch.as_tensor(values,device=device,dtype=torch.bool if key=='valid' else
                                 torch.long if key=='actions' else torch.float32)
               for key,values in data.items()}
    inputs = ('states','options','priors','valid')
    with torch.no_grad():
        logits,_ = policy(*(tensors[k] for k in inputs))
        old_logp = torch.distributions.Categorical(logits=logits).log_prob(tensors['actions'])
        error = float((old_logp-tensors['logp']).abs().max())
    if error > 5e-4:
        raise ValueError(f'On-policy log-probability mismatch {error:.7f}')
    before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
    losses, gradients = [], []
    rng = np.random.default_rng(SEED+update)
    scale = tensors['return'].std().clamp_min(.1)
    for _ in range(3):
        for batch in np.array_split(rng.permutation(len(data['actions'])),max(1,int(np.ceil(len(data['actions'])/128)))):
            ix = torch.as_tensor(batch,device=device)
            logits,value = policy(*(tensors[k][ix] for k in inputs))
            distribution = torch.distributions.Categorical(logits=logits)
            prior = torch.distributions.Categorical(logits=tensors['priors'][ix].masked_fill(~tensors['valid'][ix],-1e9))
            ratio = (distribution.log_prob(tensors['actions'][ix])-tensors['logp'][ix]).exp()
            advantage = tensors['advantage'][ix]
            objective = -torch.minimum(ratio*advantage,ratio.clamp(.85,1.15)*advantage).mean()
            critic = ((value-tensors['return'][ix])/scale).square().mean()
            kl = torch.distributions.kl_divergence(distribution,prior).mean()
            loss = objective+.5*critic+.02*kl-.005*distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = clip_grad_norm_(policy.parameters(),.5)
            if not torch.isfinite(gradient):
                raise ValueError('Non-finite PPO gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
            gradients.append(float(gradient))
    change = float(sum(float((new.detach()-old).square().sum()) for old,new in zip(before,policy.actor.parameters()))**.5)
    if change <= 0.:
        raise ValueError('PPO actor did not update')
    return dict(decisions=len(data['actions']), logp_error=error, loss=float(np.mean(losses)),
                max_gradient=max(gradients), actor_change_l2=change, device=str(device))


def train(plan, forecast, workers=6):
    root = Path(forecast['path']).parent
    start = root/'actor_0.pt'
    if not start.exists():
        torch.manual_seed(SEED)
        write_checkpoint(start,dict(state=ActorCritic().state_dict()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    updates = []
    for update,specs in enumerate(plan['ppo_batches'],1):
        previous = root/f'actor_{update-1}.pt'
        current = root/f'actor_{update}.pt'
        prior = pin(previous)
        binding = dict(previous=prior,forecast=forecast,specs=specs,source_sha256=sha(__file__))
        if current.exists():
            result = torch.load(current,map_location='cpu',weights_only=True)
            if result['binding'] != binding:
                raise ValueError('Existing PPO checkpoint uses other inputs')
        else:
            requests = [job(f'ppo_dense_{update}',spec,'ppo_05s',forecast,prior,explore=True) for spec in specs]
            rows = collect(requests,f'ppo_rollouts_{update}',workers)
            status(f'ppo_update_{update}',complete=len(rows),total=len(rows),device=str(device))
            batch = aggregate(rows)
            previous_state = torch.load(previous,map_location='cpu',weights_only=True)
            policy = ActorCritic().to(device)
            policy.load_state_dict(previous_state['state'])
            optimizer = torch.optim.AdamW(policy.parameters(),lr=.0007)
            if 'optimizer' in previous_state:
                optimizer.load_state_dict(previous_state['optimizer'])
            stats = optimize(policy,optimizer,batch,device,update)
            stats.update(flights=len(rows),completed=sum(row['mission_completed'] for row in rows),
                         cut_starts=sum(row['onsets'] for row in rows), update=update)
            result = dict(state=policy.cpu().state_dict(),optimizer=optimizer.state_dict(),binding=binding,stats=stats)
            write_checkpoint(current,result)
            print(f"PPO update {update}: {stats}",flush=True)
        updates.append(result['stats'])
    evidence = dict(created_at=now(),updates=updates,final=pin(current),forecast=forecast,
                    gamma_per_point_one_s=GAMMA_TICK, lambda_per_point_one_s=LAMBDA_TICK)
    write(ARTIFACTS/'ppo_training.json',evidence)
    return evidence
