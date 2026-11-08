"""통신 우선 v3 연구: training-only pilot → independent warm-start PPO → validation → fresh test."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse,json,hashlib,random
from pathlib import Path
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from cooperative.refine import atomic_json,SOURCE_FILES as BASE_SOURCES
from cooperative.io import validated_result
from cooperative.locking import TrainingLock
from cooperative.continuity_run import episode
from cooperative.continuity_world import REWARD_SPEC
from cooperative.continuity_control import CONTROL_SPEC
from cooperative.policy import ActorCritic
from cooperative.train import save,update
from cooperative.analyze import summarize,key
from cooperative.contracts import verify_forecaster_files
from swarm_sim.regions import ROOT,sha256_file

STUDY=ROOT/'results/continuity'
MODELS=ROOT/'model/continuity'
VERSION='communications-priority-v3'
SOURCE_FILES=(*BASE_SOURCES,'cooperative/continuity_world.py','cooperative/continuity_control.py',
              'cooperative/continuity_run.py','cooperative/continuity.py')
SEEDS=[28701,28702,28703]

def fingerprint():return {p:sha256_file(ROOT/p) for p in SOURCE_FILES}
def spec_seed(split,s):
    token=VERSION+':'+split+':'+':'.join(str(s[k]) for k in ['region','direction','kind','seed'])
    return dict(s,seed=int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%(2**31-3000004))

def candidates(profile):
    return {
        'nominal':dict(controller='nominal',checkpoint=None,profile='v2'),
        'v2_mpc':dict(controller='mpc',checkpoint=None,profile='v2'),
        'v2_ppo':dict(controller='ppo',checkpoint='model/refinement/risk_ensemble.pt',profile='v2'),
        'v2_no_risk_ppo':dict(controller='ppo',checkpoint='model/refinement/no_risk_ensemble.pt',profile='v2'),
        'priority_mpc':dict(controller='priority_mpc',checkpoint=None,profile=profile),
        'priority_no_risk_mpc':dict(controller='priority_mpc_no_risk',checkpoint=None,profile=profile),
        # The pilot persistence arm uses the linear score with hysteresis. Keep that
        # exact controller, rather than silently combining it with the winning PPO teacher.
        'persistent_mpc':dict(controller='persistent_mpc',checkpoint=None,profile='persistent'),
        'priority_ppo':dict(controller='priority_ppo',checkpoint='model/continuity/ensemble.pt',profile=profile),
        'priority_ppo_guarded':dict(controller='priority_ppo_guarded',checkpoint='model/continuity/ensemble.pt',profile=profile),
    }

def protocol():
    pilot=json.loads((STUDY/'pilot_selection.json').read_text(encoding='utf-8'))
    if pilot['source_sha256']!=fingerprint():raise ValueError('Pilot source changed; use a separate study')
    base=json.loads((ROOT/'configs/cooperative_evaluation.json').read_text(encoding='utf-8'))
    plan=dict(format=VERSION,source_sha256=fingerprint(),reward=REWARD_SPEC,control=CONTROL_SPEC,
              teacher_profile=pilot['teacher_profile'],pilot_selection_sha256=sha256_file(STUDY/'pilot_selection.json'),
              training_seeds=SEEDS,training_iterations=4,training_batch_episodes=8,
              training_warmstart='v2 risk actor per seed; fresh critic and Adam; 32 new full missions each',
              parents={str(seed):dict(path=f'model/refinement/risk_s{27101+i}/last.pt',
                                     sha256=sha256_file(ROOT/f'model/refinement/risk_s{27101+i}/last.pt')) for i,seed in enumerate(SEEDS)},
              validation=[spec_seed('validation',s) for s in base['validation']],
              test=[spec_seed('test',s) for s in base['test'][::2]],
              forecaster_files=verify_forecaster_files(),test_used_for_selection=False,
              test_scope='new seeds on existing eight test maps previously evaluated in v1/v2; six spatial groups',
              formal_counts=dict(training=96,validation=252,test=288),pilot_full_missions=48,
              selection='complete and safe; PDR >= nominal-.005; max duration <=1.15; maximize map-macro CTR, then timely PDR, then relay travel',
              candidates=candidates(pilot['teacher_profile']))
    path=ROOT/'configs/continuity_plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen v3 study changed')
    if not path.exists():atomic_json(path,plan)
    return plan

def job(spec,candidate,tag,phase,sources,record=False,sampling=None):
    name=f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
    return dict(spec=spec,candidate=candidate,prefix=str(STUDY/phase/tag/name),sources=sources,record=record,sampling=sampling)

def rollout_job(item):
    torch.set_num_threads(1)
    if item['sampling'] is not None:torch.manual_seed(item['sampling'])
    candidate=item['candidate'];ck=str(ROOT/candidate['checkpoint']) if candidate['checkpoint'] else None
    sha=sha256_file(ck) if ck else None
    previous=validated_result(item['prefix'],item['spec'],sha,rollout=item['record'])
    if (previous and previous.get('continuity_sources')==item['sources'] and
        previous.get('controller')==candidate['controller'] and previous.get('teacher_profile')==candidate['profile']):return previous
    return episode(item['spec'],candidate['controller'],ck,item['prefix'],record=item['record'],
                   profile=candidate['profile'] if candidate['profile']!='v2' else 'linear',sources=item['sources'])

def collect(tasks,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=[]
        for future in as_completed([pool.submit(rollout_job,x) for x in tasks]):
            result=future.result();results.append(result)
            print(json.dumps(dict(stage='episode',completed=len(results),total=len(tasks),region=result['region'],
                                  controller=result['controller'],CTR=result['search_connected_mean'],PDR=result['pdr'],
                                  complete=result['mission_completed'])),flush=True)
    return results

def pilot(workers):
    STUDY.mkdir(parents=True,exist_ok=True);sources=fingerprint()
    specs=[dict(region=region,direction=direction,kind=kind,seed=290101+ri*100+di*10+ki)
           for ri,region in enumerate(['teheran','gangnam','hongdae'])
           for di,direction in enumerate(['east_west','north_south'])
           for ki,kind in enumerate(['continuous','intermittent'])]
    manifest={'v2_mpc':dict(controller='mpc',checkpoint=None,profile='v2'),
              **{p:dict(controller='priority_mpc',checkpoint=None,profile=p) for p in ['linear','probability','persistent']}}
    atomic_json(STUDY/'pilot_plan.json',dict(source_sha256=sources,specs=specs,candidates=manifest,training_regions_only=True))
    tasks=[job(s,c,tag,'pilot',sources) for tag,c in manifest.items() for s in specs]
    rows=collect(tasks,workers)
    groups={tag:[r for r in rows if r['controller']==c['controller'] and r['teacher_profile']==c['profile']] for tag,c in manifest.items()}
    summaries={k:summarize(v) for k,v in groups.items()}
    allowed=[tag for tag in ['linear','probability'] if summaries[tag]['complete_count']==12 and summaries[tag]['max_duration_ratio']<=1.15]
    if not allowed:raise ValueError('Training pilot has no feasible priority profile')
    best=max(allowed,key=lambda tag:(summaries[tag]['macro']['search_connected_mean'],summaries[tag]['macro']['timely_pdr']))
    result=dict(format='training-only-teacher-selection-v3',teacher_profile=best,summaries=summaries,
                no_validation_or_test_data_used=True,training_specs=specs,source_sha256=sources)
    path=STUDY/'pilot_selection.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['teacher_profile']!=best or old['source_sha256']!=sources:raise ValueError('Pilot selection changed')
        return old
    atomic_json(path,result);return result

def training(plan,workers):
    maps=json.loads((ROOT/'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']
    keys=sorted(m['key'] for m in maps if m['split']=='train')
    for seed in SEEDS:
        torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        folder=MODELS/f's{seed}';folder.mkdir(parents=True,exist_ok=True)
        net=ActorCritic();optim=torch.optim.Adam(net.parameters(),lr=3e-4,eps=1e-5)
        config=dict(use_risk=True,continuity_version=VERSION,continuity_sources=plan['source_sha256'],
                    teacher_profile=plan['teacher_profile'],reward_spec=REWARD_SPEC,parent=plan['parents'][str(seed)])
        history=[];start=0
        with closing(TrainingLock(folder/'training.lock')):
            if (folder/'last.pt').exists():
                ck=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
                if ck['seed']!=seed or any(ck.get(k)!=v for k,v in config.items()):raise ValueError('Warm-start resume contract changed')
                net.load_state_dict(ck['model']);optim.load_state_dict(ck['optimizer']);history=ck['history'];start=ck['iteration']
                torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
            else:
                parent=torch.load(ROOT/config['parent']['path'],map_location='cpu',weights_only=False)
                net.actor.load_state_dict({k[len('actor.'):]:v for k,v in parent['model'].items() if k.startswith('actor.')})
                save(folder/'initial.pt',net,optim,0,history,seed,**config)
            schedule=np.random.default_rng(seed).permutation(keys).tolist()
            for iteration in range(start,4):
                checkpoint=folder/f'iteration_{iteration:02d}.pt'
                if not checkpoint.exists():save(checkpoint,net,optim,iteration,history,seed,**config)
                tasks=[];paths=[]
                for j in range(8):
                    k=iteration*8+j
                    spec=dict(region=schedule[k%len(schedule)],direction=['east_west','north_south'][(k//3)%2],
                              kind=['continuous','intermittent','clear'][k%3],seed=seed+50000+k*113)
                    item=job(spec,dict(controller='priority_sample',checkpoint=checkpoint.relative_to(ROOT).as_posix(),profile=plan['teacher_profile']),
                             f's{seed}','training',plan['source_sha256'],True,seed*100000+k)
                    tasks.append(item);paths.append(Path(item['prefix']).with_suffix('.npz'))
                results=collect(tasks,workers)
                stats=update(net,optim,paths,seed+iteration,epochs=4,target_kl=.02)
                stats.update(iteration=iteration+1,episodes=8,complete_count=sum(r['mission_completed'] for r in results),
                             CTR=float(np.mean([r['search_connected_mean'] for r in results])),PDR=float(np.mean([r['pdr'] for r in results])))
                history.append(stats);save(folder/'last.pt',net,optim,iteration+1,history,seed,**config)
                save(folder/f'iteration_{iteration+1:02d}.pt',net,optim,iteration+1,history,seed,**config)
                print(json.dumps(dict(stage='update',seed=seed,**stats)),flush=True)
        atomic_json(STUDY/'training'/f's{seed}'/'history.json',history)
    destination=MODELS/'ensemble.pt';members=[MODELS/f's{s}/last.pt' for s in SEEDS]
    shas=[sha256_file(p) for p in members]
    if destination.exists():
        existing=torch.load(destination,map_location='cpu',weights_only=False)
        if existing['member_sha256']!=shas:raise ValueError('Ensemble input changed')
    else:
        weights=[torch.load(p,map_location='cpu',weights_only=False) for p in members]
        torch.save(dict(format='cooperative-ensemble-v3',ensemble_models=[p['model'] for p in weights],member_sha256=shas,
                        use_risk=True,teacher_profile=plan['teacher_profile'],continuity_version=VERSION,
                        continuity_sources=plan['source_sha256'],reward_spec=REWARD_SPEC,guard_regret=.15),destination)

def evaluate(plan,phase,workers):
    specs=plan[phase];manifest=plan['candidates']
    tasks=[job(s,c,tag,phase,plan['source_sha256']) for tag,c in manifest.items() for s in specs]
    rows=collect(tasks,workers);groups={}
    for tag,c in manifest.items():
        sha=sha256_file(ROOT/c['checkpoint']) if c['checkpoint'] else None
        groups[tag]=sorted([r for r in rows if r['controller']==c['controller'] and r.get('checkpoint_sha256')==sha and r['teacher_profile']==c['profile']],key=key)
        if len(groups[tag])!=len(specs):raise ValueError('Missing or ambiguous method rows: '+tag)
        atomic_json(STUDY/phase/f'{tag}_complete.json',dict(results=groups[tag],summary=summarize(groups[tag]),candidate=c))
    return groups

def validate(plan,workers):
    groups=evaluate(plan,'validation',workers);summaries={k:summarize(v) for k,v in groups.items()}
    baseline=summaries['nominal']['macro']['pdr']
    feasible={k:v['complete_count']==28 and v['episodes']==28 and v['safety_violations']==0 and v['max_duration_ratio']<=1.15 and v['macro']['pdr']>=baseline-.005 for k,v in summaries.items()}
    choices=[k for k,v in feasible.items() if v]
    if not choices:raise ValueError('No feasible validation policy')
    selected=min(choices,key=lambda k:(-summaries[k]['macro']['search_connected_mean'],-summaries[k]['macro']['timely_pdr'],summaries[k]['macro']['relay_travel_m']))
    freeze=dict(format='continuity-validation-selection-v3',selected=selected,selected_policy=plan['candidates'][selected],
                candidates=plan['candidates'],summaries=summaries,feasible=feasible,source_sha256=plan['source_sha256'],
                test_specs=plan['test'],test_used_for_selection=False,
                checkpoint_sha256={k:sha256_file(ROOT/c['checkpoint']) for k,c in plan['candidates'].items() if c['checkpoint']})
    path=STUDY/'selection.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        if previous['selected']!=selected or previous['checkpoint_sha256']!=freeze['checkpoint_sha256']:raise ValueError('Frozen validation choice changed')
        return previous
    atomic_json(path,freeze);return freeze

def verify_selection(plan):
    freeze=json.loads((STUDY/'selection.json').read_text(encoding='utf-8'))
    if (freeze['source_sha256']!=plan['source_sha256'] or freeze['candidates']!=plan['candidates'] or
        freeze['test_specs']!=plan['test'] or freeze['test_used_for_selection'] or
        freeze['selected_policy']!=freeze['candidates'][freeze['selected']] or not freeze['feasible'][freeze['selected']]):raise ValueError('Invalid frozen v3 selection')
    for tag,sha in freeze['checkpoint_sha256'].items():
        if sha256_file(ROOT/freeze['candidates'][tag]['checkpoint'])!=sha:raise ValueError('V3 frozen checkpoint changed')
    return freeze

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['all','pilot','train','validation','test','report'],default='all');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.workers<1:p.error('workers>=1 required')
    STUDY.mkdir(parents=True,exist_ok=True)
    with closing(TrainingLock(STUDY/'study.lock')):
        if a.stage=='pilot' or (a.stage=='all' and not (STUDY/'pilot_selection.json').exists()):pilot(a.workers)
        if a.stage=='pilot':return
        plan=protocol()
        for stage in ['train','validation','test','report']:
            if a.stage not in ('all',stage):continue
            atomic_json(STUDY/'progress.json',dict(stage=stage))
            print(json.dumps(dict(stage=stage)),flush=True)
            if stage=='train':training(plan,a.workers)
            elif stage=='validation':validate(plan,a.workers)
            elif stage=='test':verify_selection(plan);evaluate(plan,'test',a.workers)
            else:
                from cooperative.continuity_report import analyze,build
                analyze();print(build(),flush=True)
        atomic_json(STUDY/'progress.json',dict(stage='complete'))

if __name__=='__main__':main()
