"""On-policy shared-relay PPO + next-event supervision + episodic selector PPO."""
import numpy as np
import torch
from torch import nn
from .data import Episodes


def agent_mean(value, mask):
    return ((value*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()


def assert_behavior(model, data, device='cpu', mode='joint', batch_size=16):
    maximum=0.;model.eval()
    with torch.no_grad():
        for begin in range(0,len(data),batch_size):
            b,m=data.batch(np.arange(begin,min(begin+batch_size,len(data))),device)
            logits,_=model.policy(b,mode)
            logp=torch.distributions.Categorical(logits=logits).log_prob(m['chosen'])
            maximum=max(maximum,float((logp-m['logp'])[m['agent_mask']].abs().max()))
    tolerance=3e-4 if str(device).startswith('cuda') else 8e-5
    if not np.isfinite(maximum) or maximum>tolerance:
        raise ValueError(f'Behavior policy differs from raw-input replay: {maximum} > {tolerance}')
    return maximum


def norm_of(gradients):
    return float(torch.sqrt(sum((g.detach().double().square().sum() for g in gradients if g is not None),
                                 torch.zeros((),device=next(g.device for g in gradients if g is not None))))) if any(g is not None for g in gradients) else 0.


def update(model, optimizer, episodes, config, update_number, device='cpu'):
    if any(r.get('deterministic_policy',r.get('request',{}).get('deterministic',False)) for _,r in episodes):
        raise ValueError('Argmax evaluation trajectories cannot be used as on-policy PPO samples')
    mode=config.get('prediction_mode','joint')
    data=Episodes(episodes,config)
    error=assert_behavior(model,data,device,mode,config['minibatch'])
    generator=np.random.default_rng(config['seed']+update_number*1013)
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    proof=dict(encoder_from_policy=0.,timing_from_policy=0.)
    rows=[];stop=False
    for epoch in range(config['epochs']):
        order=generator.permutation(len(data))
        for first in range(0,len(data),config['minibatch']):
            b,m=data.batch(order[first:first+config['minibatch']],device)
            logits,timing=model.policy(b,mode)
            dist=torch.distributions.Categorical(logits=logits)
            ratio=(dist.log_prob(m['chosen'])-m['logp']).exp()
            advantage=m['advantage'][:,None]
            clip=config['clip']
            policy=agent_mean(-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage),m['agent_mask'])
            value=((model.value(m['state'])-m['returns'])/data.value_scale).square().mean()
            entropy=agent_mean(dist.entropy(),m['agent_mask'])
            teacher=torch.distributions.Categorical(logits=b['teacher'].masked_fill(~b['action_mask'],-1e9))
            anchor=agent_mean(torch.distributions.kl_divergence(dist,teacher),m['agent_mask'])
            aux=torch.zeros((),device=device)
            count=0
            if mode=='joint':
                good=m['label_valid']&b['support']&m['agent_mask'][...,None]
                if good.any():
                    aux=nn.functional.cross_entropy(timing[good],m['labels'][good]);count+=1
                sample=data.aux_batch(generator,128,device)
                if sample is not None:
                    _,q=model.forecaster.encode(sample['values'],sample['masks'],sample['intents'],sample['geometry'])
                    aux=aux+nn.functional.cross_entropy(q,sample['labels']);count+=1
                if count:
                    aux=aux/count
                if not rows or proof['encoder_from_policy']==0 or proof['timing_from_policy']==0:
                    encoder=list(model.forecaster.backbone.encoder_parameters())
                    head=list(model.forecaster.timing.parameters())
                    gradients=torch.autograd.grad(policy,encoder+head,retain_graph=True,allow_unused=True)
                    proof['encoder_from_policy']=max(proof['encoder_from_policy'],norm_of(gradients[:len(encoder)]))
                    proof['timing_from_policy']=max(proof['timing_from_policy'],norm_of(gradients[len(encoder):]))
            loss=policy+.5*value-config['entropy_coef']*entropy+config['anchor_coef']*anchor+config['prediction_coef']*aux
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint objective; checkpoint not committed')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5)
            if not torch.isfinite(gradient):
                raise FloatingPointError('Nonfinite joint gradient')
            optimizer.step()
            with torch.no_grad():
                after,_=model.policy(b,mode)
                old=torch.distributions.Categorical(logits=m['old_logits'])
                new=torch.distributions.Categorical(logits=after)
                kl=float(agent_mean(torch.distributions.kl_divergence(old,new),m['agent_mask']))
            rows.append([float(policy.detach()),float(value.detach()),float(aux.detach()),kl])
            if kl>config['target_kl']:
                stop=True;break
        if stop:
            break
    # The deployment action is sampled once per episode. Forced/balanced episodes
    # are excluded: applying PPO to their selector log-probabilities is invalid.
    selected=[(a,r) for a,r in episodes if r['selector_on_policy']]
    selector_loss=0.
    if selected:
        context=torch.as_tensor(np.stack([a['public_context'] for a,_ in selected]),device=device)
        masks=torch.as_tensor(np.stack([a['deployment_mask'] for a,_ in selected]),device=device)
        chosen=torch.as_tensor([int(a['selector_choice']) for a,_ in selected],device=device)
        old_logp=torch.as_tensor([float(a['selector_logp']) for a,_ in selected],device=device)
        returns=torch.as_tensor([r['selector_return'] for _,r in selected],device=device,dtype=torch.float32)
        old_value=torch.as_tensor([float(a['selector_value']) for a,_ in selected],device=device)
        adv=returns-old_value
        # Do not erase the learning signal in a batch containing one episode.
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-5) if len(selected)>1 else adv
        dist=torch.distributions.Categorical(logits=model.select(context,masks))
        ratio=(dist.log_prob(chosen)-old_logp).exp()
        pl=-torch.minimum(ratio*adv,ratio.clamp(1-config['clip'],1+config['clip'])*adv).mean()
        vl=(model.selector_value(context).squeeze(-1)-returns).square().mean()
        loss=config['selector_coef']*(pl+.5*vl-config['entropy_coef']*dist.entropy().mean())
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5)
        if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite selector gradient')
        optimizer.step();selector_loss=float(loss.detach())
    changes={prefix:float(np.sqrt(sum(float((v.detach().cpu()-before[k]).double().square().sum())
                       for k,v in model.state_dict().items() if k.startswith(prefix))))
             for prefix in ['forecaster.backbone','forecaster.timing','actor','critic','selector']}
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise FloatingPointError('Nonfinite updated state')
    return dict(update=update_number,samples=len(data),minibatches=len(rows),epochs=epoch+1,
                behavior_logp_max_error=error,policy_loss=float(np.mean(rows,axis=0)[0]),
                value_loss=float(np.mean(rows,axis=0)[1]),timing_nll=float(np.mean(rows,axis=0)[2]),
                maximum_kl=max(r[3] for r in rows),selector_loss=selector_loss,
                gradient_proof=proof,weight_change_l2=changes)
