from __future__ import annotations
from pathlib import Path
import json,time
import numpy as np
import torch
from solo.contracts import ROOT,HISTORY,SCHEMA,sha,atomic_json,atomic_npz,ensure_map,seed_all
from solo.model import load_controller
from solo.targets import prediction_targets
from cooperative.continuity_world import training_state
from solo.world import World,live_history,supported
from solo.features import observe_physics,geometry
from cooperative.continuity_control import observe
from cooperative.candidates import indices_to_correction
from cooperative.refinement_analysis import trajectory_audit,NETWORK_COLUMNS
from cooperative.continuity_report import independent_reward_audit

_STOP=None
_MODEL=None
_CHECKPOINT=None
_SIGNATURE=None


def initialize_worker(snapshot,signature,stop):
    global _STOP,_MODEL,_CHECKPOINT,_SIGNATURE
    torch.set_num_threads(1)
    _STOP=stop;_MODEL,_CHECKPOINT=load_controller(snapshot,'cpu');_SIGNATURE=signature


def load_completed(prefix,spec,checkpoint_sha,signature):
    prefix=Path(prefix);meta=prefix.with_suffix('.json');array=prefix.with_suffix('.npz')
    if not meta.exists() or not array.exists():return None
    row=json.loads(meta.read_text(encoding='utf-8'))
    if (row.get('schema')!=SCHEMA or row.get('checkpoint_sha256')!=checkpoint_sha
        or row.get('execution_signature')!=signature or any(row.get(k)!=v for k,v in spec.items())
        or row.get('array_sha256')!=sha(array)):
        raise ValueError('Existing rollout differs from frozen request: '+str(prefix))
    return row


def episode(spec,snapshot,prefix,signature,deterministic=False,stop=None,model=None,max_steps=None):
    started=time.perf_counter();prefix=Path(prefix);snapshot=Path(snapshot)
    cksha=sha(snapshot)
    old=load_completed(prefix,spec,cksha,signature)
    if old:return old
    plan=json.loads((ROOT/'configs/transformer_plan.json').read_text(encoding='utf-8'))
    entry=next(x for x in plan['maps'] if x['key']==spec['region'])
    ensure_map(entry,stop)
    seed_all(spec['seed']+9173)
    if model is None:model,ck=load_controller(snapshot,'cpu')
    else:ck=torch.load(snapshot,map_location='cpu',weights_only=False)
    model.eval()
    env=World(**spec,max_steps=max_steps)
    arrays={k:[] for k in ['context','candidates','mask','teacher','state','chosen','logp','value','old_logits',
      'geometry','gain','prediction','prediction_supported',
      'nominal_search_targets','phase_before_action','requested_corrections','applied_corrections','command_switches','relay_motion']}
    previous=np.zeros(5,int);decision_ms=[];prediction_ms=[]
    while not env.done:
        if stop is not None and stop.is_set():raise InterruptedError('Stop requested; incomplete episode is not committed')
        tick=time.perf_counter()
        histories=live_history(env.recorder);g=geometry(env)
        c,f,m,teacher,gain=observe_physics(env)
        support=supported(env.last_x[3:8],env.last_mask[3:8],len(env.recorder.values))
        with torch.no_grad():
            logits,prediction=model.policy(torch.from_numpy(c),torch.from_numpy(f),torch.from_numpy(m),torch.from_numpy(teacher),torch.from_numpy(gain),
                *[torch.from_numpy(v) for v in histories],torch.from_numpy(g),torch.from_numpy(support))
            dist=torch.distributions.Categorical(logits=logits)
            action=logits.argmax(-1) if deterministic else dist.sample()
            logp=dist.log_prob(action);state=training_state(env,False)
            value=float(model.value(torch.from_numpy(state)))
        decision_ms.append((time.perf_counter()-tick)*1000)
        prediction_ms.append(model.last_prediction_ms)
        for key,item in [('context',c),('candidates',f),('mask',m),('teacher',teacher),('state',state),
                        ('chosen',action.numpy()),('logp',logp.numpy()),('value',value),('old_logits',logits.numpy()),
                        ('prediction',prediction.numpy()),('prediction_supported',support),('geometry',g),('gain',gain)]:arrays[key].append(item)
        arrays['nominal_search_targets'].append(env.proposal['nominal'][:3].copy())
        arrays['phase_before_action'].append(env.proposal['phase'])
        a=action.numpy();command=indices_to_correction(a)
        env.step(command)
        arrays['requested_corrections'].append(command[3:].copy())
        arrays['applied_corrections'].append(env.history[-1]['correction'][3:].copy())
        arrays['command_switches'].append(a!=previous);previous=a.copy()
        arrays['relay_motion'].append(env.city.points[env.node_log[-1]][3:]-env.city.points[env.node_log[-2]][3:])
        env.previous_actions=np.zeros(5,int) if env.history[-1]['protected'] else a.copy()
    row=dict(env.summary,controller='transformer_'+ck['mode'],mode=ck['mode'],schema=SCHEMA,**spec,
             deterministic=deterministic,checkpoint_sha256=cksha,execution_signature=signature,
             source_version='standalone-transformer-ppo-v5',wall_s=time.perf_counter()-started,
             command_switch_count=int(np.asarray(arrays['command_switches']).sum()),
             relay_stationary_fraction=float((np.linalg.norm(arrays['relay_motion'],axis=-1)<1e-9).mean()),
             decision_p95_ms=float(np.percentile(decision_ms,95)))
    data={key:np.asarray(value) for key,value in arrays.items()}
    data.update(rf_values=np.asarray(env.recorder.values),rf_masks=np.asarray(env.recorder.masks),rf_intents=np.asarray(env.recorder.intents))
    g=int(env.packets.ic[0]);birth=env.packets.born[:g]
    bounds=np.searchsorted(birth,np.arange(env.k+1)*3-1e-9)
    delivered=env.packets.status[:g]==1
    weighted=delivered*np.minimum(1.,.25/np.maximum(env.packets.delays[:g],.005))
    cumulative=np.column_stack([np.arange(g+1),np.r_[0,np.cumsum(delivered)],np.r_[0,np.cumsum(weighted)]])
    data.update(packet_cohorts=cumulative[bounds[1:]]-cumulative[bounds[:-1]],
      reward_terms=np.stack([env.settled[k]['reward_terms'] for k in range(env.k)]),
      nodes=np.asarray(env.node_log),positions_xy=env.city.points[np.asarray(env.node_log)].astype(np.float32),
      actions=data['chosen'],C=np.asarray(env.stats),links=np.asarray(env.channel_available),
      packet_frames=np.asarray(env.packet_frames),rewards=np.asarray([env.settled[k]['reward'] for k in range(env.k)],np.float32))
    data['prediction_targets'],data['prediction_valid']=prediction_targets(data['links'],env.k)
    audit=trajectory_audit(data,env.city,row);audit.update(independent_reward_audit(data,env.city,row))
    if not audit['passed']:raise ValueError('Continuous safety/graph/packet audit failed')
    row['audit']=audit
    row['forecast_p95_ms']=float(np.percentile(prediction_ms,95)) if prediction_ms else 0.
    row['lightgbm_used']=False
    row['gru_used']=False
    atomic_npz(prefix.with_suffix('.npz'),**data)
    row['array_sha256']=sha(prefix.with_suffix('.npz'))
    atomic_json(prefix.with_suffix('.json'),row)
    return row


def worker(task):
    snapshot,prefix,spec,signature,deterministic=task
    if signature!=_SIGNATURE:raise ValueError('Worker received a different execution contract')
    if _STOP is not None and _STOP.is_set():return dict(stopped=True,spec=spec)
    try:return episode(spec,snapshot,prefix,signature,deterministic,_STOP,_MODEL)
    except InterruptedError:return dict(stopped=True,spec=spec)
