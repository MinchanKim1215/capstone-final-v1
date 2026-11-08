"""Real streaming trajectories. Only raw causal inputs are retained for PPO."""
from pathlib import Path
import time
import numpy as np
import torch
from swarm_sim.motion import swept_minimum
from .contracts import (DEPLOYMENTS, DEPLOYMENT_BY_KEY, atomic_json, deployment_mask,
                         entry_for, public_context, sha, seed_all)
from .features import OFFSETS
from .model import tensor_observation
from .observation import Observer, policy_inputs, critic_state
from .targets import timing_labels
from .world import World


def trajectory_audit(env, arrays):
    nodes=arrays['nodes']
    pos=env.city.points[nodes]
    minimum=min(swept_minimum(a,b)[0] for a,b in zip(pos[:-1],pos[1:]))
    speed=float(np.linalg.norm(np.diff(pos,axis=0),axis=-1).max()/3)
    old,new=nodes[:-1].ravel(),nodes[1:].ravel()
    adjacent=np.asarray(env.city.graph[old,new]).ravel()>0
    bad=int(np.sum(~(adjacent | (old==new))))
    import shapely
    segments=shapely.linestrings(np.stack([pos[:-1],pos[1:]],axis=-2).reshape(-1,2,2))
    obstacles=int(shapely.intersects(segments,env.city.flight_blocker).sum())
    g=int(env.packets.ic[0])
    conserved=g==int(env.packets.ic[1]+env.packets.ic[5]+env.packets.ic[6]+env.packets.ic[7])
    reward_ok=np.allclose(arrays['reward_terms'].sum(1),arrays['rewards'],atol=1e-6,rtol=1e-6)
    search_ok=not arrays['commands'][:,:env.ns].any()
    passed=minimum>=5-1e-7 and speed<=5+1e-7 and bad==0 and obstacles==0 and conserved and reward_ok and search_ok
    return dict(passed=bool(passed),minimum_swept_separation_m=float(minimum),maximum_speed_mps=speed,
                invalid_graph_edges=bad,obstacle_segments=obstacles,packet_conserved=bool(conserved),
                reward_conserved=bool(reward_ok),search_actions_zero=bool(search_ok))


def episode(model, spec, config, forced_deployment=None, deterministic=False, max_steps=None,
            prediction_mode='joint', prefix=None, request=None):
    started=time.perf_counter()
    seed_all(spec['seed']+9173)
    model.eval()
    entry=entry_for(spec['region'])
    ctx=public_context(entry,spec['direction'],config['available_drones'],config['minimum_search'])
    feasible=deployment_mask(config['available_drones'],config['minimum_search'])
    device=next(model.parameters()).device
    with torch.no_grad():
        selector_logits=model.select(torch.as_tensor(ctx,device=device),torch.as_tensor(feasible,device=device))
        dist=torch.distributions.Categorical(logits=selector_logits)
        chosen=int(selector_logits.argmax()) if deterministic else int(dist.sample())
        if forced_deployment is not None:
            chosen=next(i for i,d in enumerate(DEPLOYMENTS) if d.key==forced_deployment)
            if not feasible[chosen]:
                raise ValueError('Forced deployment violates the public fleet budget')
        selector_logp=float(dist.log_prob(torch.tensor(chosen,device=device)))
        selector_value=float(model.selector_value(torch.as_tensor(ctx,device=device)).squeeze())
    d=DEPLOYMENTS[chosen]
    observer=Observer(model if prediction_mode!='none' else None)
    env=World(**spec,deployment=d.key,observer=observer,max_steps=max_steps,
              fleet_cost=config['fleet_cost_per_step'])
    records={k:[] for k in ['context','candidates','action_mask','teacher','gain','geometry','support',
                            'state','chosen','logp','old_logits','value','endpoints','commands']}
    while not env.done:
        b=policy_inputs(env)
        state=critic_state(env)
        with torch.no_grad():
            logits,_=model.policy(tensor_observation(b,device),prediction_mode)
            distribution=torch.distributions.Categorical(logits=logits)
            action=logits.argmax(-1) if deterministic else distribution.sample()
            logp=distribution.log_prob(action)
            value=float(model.value(torch.as_tensor(state,device=device)))
        for key in ('context','candidates','action_mask','teacher','gain','geometry','support'):
            records[key].append(b[key])
        for key,value_ in [('state',state),('chosen',action.cpu().numpy()),('logp',logp.cpu().numpy()),
                           ('old_logits',logits.cpu().numpy()),('value',value),('endpoints',env.k*6)]:
            records[key].append(value_)
        command=np.zeros((d.n,2))
        command[d.ns:]=OFFSETS[action.cpu().numpy()]
        records['commands'].append(command)
        env.step(command)
        env.previous_actions=np.zeros(d.nr,int) if env.history[-1]['protected'] else action.cpu().numpy().copy()
    arrays={key:np.asarray(value) for key,value in records.items()}
    arrays.update(design=b['design'],rf_values=np.asarray(observer.values),rf_masks=np.asarray(observer.masks),
                  rf_intents=np.asarray(observer.intents),links=np.asarray(env.channel_available),
                  nodes=np.asarray(env.node_log),positions=env.city.points[np.asarray(env.node_log)],
                  rewards=np.array([env.settled[k]['reward'] for k in range(env.k)],np.float32),
                  reward_terms=np.stack([env.settled[k]['reward_terms'] for k in range(env.k)]),
                  aux_endpoints=np.rint(np.asarray(observer.prediction_times)*2).astype(int),
                  aux_geometry=np.asarray(observer.prediction_geometry),
                  aux_support=np.asarray(observer.prediction_support),
                  aux_probabilities=np.asarray(observer.predictions),
                  public_context=ctx,deployment_mask=feasible,
                  selector_choice=np.array(chosen),selector_logp=np.array(selector_logp),
                  selector_old_logits=selector_logits.cpu().numpy(),selector_value=np.array(selector_value))
    arrays['labels'],arrays['label_valid']=timing_labels(arrays['links'],arrays['endpoints'].clip(1),d.ns)
    # No prediction exists at the initial boundary, even if its offline target does.
    arrays['label_valid'][arrays['endpoints']==0]=False
    if len(arrays['aux_endpoints']):
        arrays['aux_labels'],arrays['aux_valid']=timing_labels(arrays['links'],arrays['aux_endpoints'],d.ns)
    else:
        arrays['aux_labels']=np.zeros((0,d.nr,d.n),np.int64)
        arrays['aux_valid']=np.zeros((0,d.nr,d.n),bool)
    audit=trajectory_audit(env,arrays)
    if not audit['passed']:
        raise ValueError('Trajectory audit failed: '+str(audit))
    row=dict(env.summary,spec=spec,audit=audit,prediction_mode=prediction_mode,
             deterministic_policy=bool(deterministic),
             selector_on_policy=forced_deployment is None,selected_deployment=d.key,
             full_mission=max_steps is None,smoke_only=max_steps is not None or bool(request and request.get('purpose')!='training'),
             prediction_calls=len(observer.prediction_times),prediction_cadence_s=1,control_cadence_s=3,
             selector_return=float(arrays['rewards'].sum()/2400.),wall_s=time.perf_counter()-started)
    if request is not None:row['request']=request
    if prefix is not None:
        prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
        path=prefix.with_suffix('.npz');temporary=prefix.with_suffix('.pending.npz')
        np.savez_compressed(temporary,**arrays);temporary.replace(path)
        row['array_sha256']=sha(path)
        atomic_json(prefix.with_suffix('.json'),row)
    return arrays,row


def observation_at(arrays,t):
    end=int(arrays['endpoints'][t])
    nr,n=arrays['support'][t].shape
    result={k:arrays[k][t] for k in ('context','candidates','action_mask','teacher','gain','geometry','support')}
    result['design']=arrays['design']
    for key,last,dtype in [('values',48,np.float32),('masks',48,bool),('intents',9,np.float32)]:
        a=np.asarray(arrays['rf_'+key][max(0,end-40):end],dtype=dtype)
        if len(a)<40:
            a=np.concatenate([np.zeros((40-len(a),nr,n,last),dtype=dtype),a])
        result[key]=a.transpose(1,2,0,3).copy()
    return result


def legacy_episode(spec,config,max_steps=None,prefix=None,request=None):
    """Frozen best176 controller in the same private N=8 physics engine.

    Old ANY-event heads stay separate from v6 NEXT-event metrics. This arm is
    used for matched mission comparison, not a fabricated predictor AP gain.
    """
    from . import LEGACY
    from solo.model import load_controller
    from solo.features import geometry,observe_physics
    from solo.world import live_history,supported
    path=LEGACY/'results/transformer/train/transformer_main/checkpoints/s0000176.pt'
    if config['available_drones']<8 or config['minimum_search']>3:
        raise ValueError('Frozen v5 fleet is not feasible under this budget')
    seed_all(spec['seed']+9173)
    model,_=load_controller(path,'cpu');model.eval()
    observer=Observer()
    env=World(**spec,observer=observer,max_steps=max_steps,fleet_cost=config['fleet_cost_per_step'])
    commands=[];predictions=[];supports=[];milliseconds=[]
    tick=time.perf_counter()
    while not env.done:
        c,f,m,t,gain=observe_physics(env)
        history=live_history(observer)
        support=supported(env.last_x[3:],env.last_mask[3:],len(observer.values))
        with torch.no_grad():
            logits,p=model.policy(torch.from_numpy(c),torch.from_numpy(f),torch.from_numpy(m),
                    torch.from_numpy(t),torch.from_numpy(gain),*[torch.from_numpy(h) for h in history],
                    torch.from_numpy(geometry(env)),torch.from_numpy(support))
            action=logits.argmax(-1).numpy()
        predictions.append(p.numpy());supports.append(support);milliseconds.append(model.last_prediction_ms)
        command=np.zeros((8,2));command[3:]=OFFSETS[action];commands.append(command)
        env.step(command)
        env.previous_actions=np.zeros(5,int) if env.history[-1]['protected'] else action.copy()
    arrays=dict(nodes=np.asarray(env.node_log),commands=np.asarray(commands),
        reward_terms=np.stack([env.settled[k]['reward_terms'] for k in range(env.k)]),
        rewards=np.array([env.settled[k]['reward'] for k in range(env.k)],np.float32),
        links=np.asarray(env.channel_available),rf_values=np.asarray(observer.values),
        rf_masks=np.asarray(observer.masks),rf_intents=np.asarray(observer.intents),
        v5_any_event_prediction=np.asarray(predictions),v5_prediction_supported=np.asarray(supports),
        aux_endpoints=np.zeros(0,int),aux_probabilities=np.zeros((0,5,8,4),np.float32),
        aux_support=np.zeros((0,5,8),bool),aux_valid=np.zeros((0,5,8),bool),aux_labels=np.zeros((0,5,8),int))
    audit=trajectory_audit(env,arrays)
    if not audit['passed']:raise ValueError('Frozen v5 bridge trajectory failed safety/packet audit')
    row=dict(env.summary,deployment='legacy_v5',spec=spec,audit=audit,
             deterministic_policy=True,
             selector_on_policy=False,selected_deployment='legacy_v5',
             full_mission=max_steps is None,smoke_only=max_steps is not None or bool(request and request.get('purpose')!='training'),
             selector_return=float(arrays['rewards'].sum()/2400.),wall_s=time.perf_counter()-tick,
             legacy_checkpoint_sha256=sha(path),forecast_calls=len(predictions),
             forecast_p95_ms=float(np.percentile(milliseconds,95)),
             prediction_scope='v5 ANY-event outputs retained separately; no v6 NEXT-event AP assigned')
    if request is not None:row['request']=request
    if prefix is not None:
        prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
        temporary=prefix.with_suffix('.pending.npz');path=prefix.with_suffix('.npz')
        np.savez_compressed(temporary,**arrays);temporary.replace(path)
        row['array_sha256']=sha(path);atomic_json(prefix.with_suffix('.json'),row)
    return arrays,row
