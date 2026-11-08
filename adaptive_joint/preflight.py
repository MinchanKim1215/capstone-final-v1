"""Real pre-training acceptance gates. Never performs the 256-update campaign."""
from pathlib import Path
import copy
import io
import json
import time
import unittest
import numpy as np
import torch
from . import ROOT,LEGACY,ARTIFACTS
from .contracts import (DEPLOYMENTS,atomic_json,digest,legacy_hashes,read_json,seed_all,sha)
from .experiment import (assert_protocol,collect,load_checkpoint,save_checkpoint)
from .learning import assert_behavior,update
from .model import Controller,optimizer,tensor_observation
from .data import Episodes
from .rollout import episode,observation_at


def unit_tests():
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),top_level_dir=str(ROOT.parent))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    return dict(passed=result.wasSuccessful(),tests=result.testsRun,output=stream.getvalue())


def preserved():
    return read_json(ARTIFACTS/'legacy_before.json')==legacy_hashes()


def nominal_parity(region='ansan_bono',seed=16213):
    """Unchanged v5 engine vs generalized engine, actual complete zero-action flights."""
    from cooperative.world import ClosedLoopWorld
    from .world import World
    from .observation import Observer
    for path in [LEGACY/'cache/usable'/f'{region}_east_west_motion.npz']:
        if not path.exists():raise ValueError('Legacy read-only baseline fixture is absent: '+str(path))
    # Visibility already exists from the completed v5 campaign. Do not run a
    # fallback that would regenerate legacy assets if this fixture is missing.
    from .contracts import entry_for
    entry=entry_for(region)
    visibility=LEGACY/'cache/usable'/f"{region}_{entry['sha256'][:12]}_visibility.npz"
    if not visibility.exists():raise ValueError('Legacy visibility fixture missing; parity must not modify original caches')
    spec=dict(region=region,seed=seed,kind='intermittent',direction='east_west')
    old=ClosedLoopWorld(**spec,forecast=False)
    while not old.done:old.step(np.zeros((8,2)))
    new=World(**spec,observer=Observer(),fleet_cost=0.)
    while not new.done:new.step(np.zeros((8,2)))
    arrays_equal=(np.array_equal(old.node_log,new.node_log) and
                  np.array_equal(old.channel_available,new.channel_available) and
                  np.array_equal(old.packets.status,new.packets.status) and
                  np.array_equal(old.packets.delays,new.packets.delays) and
                  np.array_equal(old.last_x,new.last_x) and np.array_equal(old.last_mask,new.last_mask))
    metrics=['duration_s','search_connected_mean','pdr','outage_total_s','outage_max_s','beacon_tx','beacon_rx']
    equal=all(old.summary[k]==new.summary[k] for k in metrics)
    return dict(passed=bool(arrays_equal and equal),complete=old.summary['mission_completed'] and new.summary['mission_completed'],
                metrics={k:[old.summary[k],new.summary[k]] for k in metrics},arrays_bit_equal=bool(arrays_equal))


def run(protocol,full=True):
    assert_protocol(protocol)
    c=dict(protocol['config']);c.update(epochs=1,minibatch=16)
    folder=ARTIFACTS/'preflight'/digest(protocol)[:16]
    folder.mkdir(parents=True,exist_ok=True)
    checks={}
    def record(name,value):
        checks[name]=value;atomic_json(folder/'checks.json',checks)
        print(json.dumps(dict(stage='preflight',check=name,result=value),ensure_ascii=False),flush=True)
    tests=unit_tests();record('unit_tests',tests)
    if not tests['passed']:raise AssertionError(tests['output'])
    assets=read_json(ARTIFACTS/'assets_manifest.json')
    assets_ok=assets['ready'] and assets['protocol_sha256']==digest(protocol)
    if assets_ok:
        assets_ok=all(Path(r['path']).exists() and sha(r['path'])==r['sha256'] for r in assets['rows'])
    record('all_nominal_assets',dict(passed=assets_ok,regions=assets['regions'],trajectories=assets['trajectories'],
                                   incomplete_nominal=assets['incomplete_nominal']))
    if not assets_ok:raise ValueError('Run assets for this frozen protocol before preflight')
    from .contracts import plan,public_context,PUBLIC_CONTEXT_DIM
    contexts=[public_context(m,'east_west') for m in plan()['maps'] if m['joint_split'] in ('train','validation')]
    record('all_development_map_inputs',dict(passed=all(v.shape==(PUBLIC_CONTEXT_DIM,) and np.isfinite(v).all() for v in contexts),maps=len(contexts),dimension=PUBLIC_CONTEXT_DIM))
    seed_all(c['seed']);model=Controller();opt=optimizer(model,c)
    initial=save_checkpoint(folder/'initial',model,opt,c,0,protocol,[],purpose='preflight_only')
    specs=[dict(region='ansan_bono',seed=16213,kind='continuous',direction='east_west') for _ in DEPLOYMENTS]
    episodes=collect(initial,protocol,specs,folder/'short_four',[d.key for d in DEPLOYMENTS],False,24)
    record('four_deployments',dict(passed=all(r['audit']['passed'] for _,r in episodes),
           details=[dict(deployment=r['deployment'],n=r['total_drones'],search=r['search_drones'],
                         calls=r['prediction_calls'],audit=r['audit']) for _,r in episodes]))
    north_specs=[dict(s,direction='north_south') for s in specs]
    north=collect(initial,protocol,north_specs,folder/'north_south_four',[d.key for d in DEPLOYMENTS],False,12)
    record('north_south_four',dict(passed=all(r['audit']['passed'] for _,r in north),
                                 deployments=[r['deployment'] for _,r in north]))
    episodes.extend(north)
    # Genuine adaptive rollouts under two different public resource constraints.
    for i,(available,search) in enumerate([(6,3),(10,6)]):
        config=dict(c,available_drones=available,minimum_search=search)
        a,r=episode(model,dict(region='ansan_bono',seed=16214+i,kind='continuous',direction='east_west'),
                    config,max_steps=24)
        episodes.append((a,r))
        if r['total_drones']>available or r['search_drones']<search:raise AssertionError('Budget violated')
    record('resource_constraints',dict(passed=True,selected=[r['deployment'] for _,r in episodes[-2:]]))
    # Future RF and labels must not alter the saved input at an earlier boundary.
    a,_=episodes[0];t=9;before=observation_at(a,t);poison=copy.deepcopy(a)
    poison['rf_values'][int(a['endpoints'][t]):]=31000
    poison['labels'][:]=0;poison['label_valid'][:]=~poison['label_valid']
    after=observation_at(poison,t)
    causal=all(np.array_equal(before[k],after[k]) for k in before)
    record('no_future_input',dict(passed=causal))
    if not causal:raise AssertionError('Future input leaked')
    data=Episodes(episodes,c)
    cpu_error=assert_behavior(model,data,'cpu',c.get('prediction_mode','joint'))
    device=c['device']
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA preflight is required by this profile')
    model.to(device);opt=optimizer(model,c)
    stats=update(model,opt,episodes,c,1,device)
    proof=stats['gradient_proof']
    passed=proof['encoder_from_policy']>0 and proof['timing_from_policy']>0
    passed=passed and stats['weight_change_l2']['actor']>0 and stats['weight_change_l2']['selector']>0
    record('real_joint_backward',dict(passed=passed,cpu_behavior_error=cpu_error,statistics=stats))
    if not passed:raise AssertionError('No true PPO gradient into the time predictor or selector')
    saved=save_checkpoint(folder/'updated',model,opt,c,1,protocol,[stats],purpose='preflight_only')
    loaded,_,_=load_checkpoint(saved,protocol,device,True)
    equal=all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    record('checkpoint_round_trip',dict(passed=equal))
    if not equal:raise AssertionError('Checkpoint changed model tensors')
    from .replay import compare
    replay=compare(saved,protocol,folder/'short_four',device,max_times=6)
    counts={name:m['next_onset_lt5s']['samples'] for name,m in replay['models'].items()}
    same=len(set(counts.values()))==1 and min(counts.values())>0
    record('matched_prediction_replay',dict(passed=same,common_samples=counts,scope='SMOKE ONLY'))
    if not same:raise AssertionError('Prediction comparison used different observation populations')
    fresh_specs=[dict(region='ansan_bono',seed=17220+i,kind='continuous',direction='east_west') for i in range(4)]
    fresh=collect(saved,protocol,fresh_specs,folder/'resume_rollouts',None,False,24)
    model_a,opt_a,_=load_checkpoint(saved,protocol,device,True)
    stats_a=update(model_a,opt_a,fresh,c,2,device)
    state_a={k:v.detach().cpu().clone() for k,v in model_a.state_dict().items()}
    model_b,opt_b,_=load_checkpoint(saved,protocol,device,True)
    stats_b=update(model_b,opt_b,fresh,c,2,device)
    exact=all(torch.equal(v,model_b.state_dict()[k].detach().cpu()) for k,v in state_a.items())
    record('resume_bit_exact',dict(passed=exact,statistics_equal=stats_a==stats_b))
    if not exact:raise AssertionError('Resumed Adam/PPO update is not deterministic')
    frozen,_,_=load_checkpoint(saved,protocol,device,True)
    fixed_config=dict(c,prediction_mode='fixed')
    fixed_stats=update(frozen,optimizer(frozen,fixed_config),fresh,fixed_config,2,device)
    unchanged=fixed_stats['weight_change_l2']['forecaster.backbone']==0 and fixed_stats['weight_change_l2']['forecaster.timing']==0
    record('fixed_predictor_arm',dict(passed=unchanged))
    if not unchanged:raise AssertionError('Frozen forecast control arm changed')
    from .rollout import legacy_episode
    _,legacy=legacy_episode(specs[0],c,max_steps=12)
    record('frozen_best176_bridge',dict(passed=legacy['audit']['passed'],checkpoint_sha256=legacy['legacy_checkpoint_sha256']))
    if full:
        parity=nominal_parity();record('complete_v5_physics_parity',parity)
        if not parity['passed'] or not parity['complete']:raise AssertionError('Generalization changed legacy eight-agent physics')
        full_rows=collect(initial,protocol,specs,folder/'full_four',[d.key for d in DEPLOYMENTS],True,None,load_arrays=False)
        record('complete_four_missions',dict(passed=all(r['audit']['passed'] and r['mission_completed'] for r in full_rows),
                   smoke_only=True,details=[dict(deployment=r['deployment'],completed=r['mission_completed'],duration_s=r['duration_s'],audit=r['audit']) for r in full_rows]))
    original_ok=preserved();record('original_preserved',dict(passed=original_ok))
    ready=full and original_ok and all(v.get('passed',False) for v in checks.values())
    result=dict(ready=ready,protocol_sha256=digest(protocol),checks_path=str((folder/'checks.json').resolve()),
                purpose='Pre-training acceptance only. No campaign or performance superiority claim.',
                long_training_started=False,sealed_test_opened=False,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    atomic_json(ARTIFACTS/'readiness.json',result)
    return result
