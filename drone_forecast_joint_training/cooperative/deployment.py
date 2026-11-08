"""v3 결과는 개발 자료로 재분류하고, 강건한 기본값을 새 확인 난수에서 검사한다."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.setdefault(name,'1')
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import closing
import numpy as np
import torch
from cooperative.continuity import protocol as v3_protocol,SOURCE_FILES as V3_SOURCES
from cooperative.deployment_run import episode
from cooperative.deployment_policy import FORMAT
from cooperative.io import validated_result
from cooperative.analyze import key,summarize
from cooperative.refine import atomic_json
from cooperative.locking import TrainingLock
from swarm_sim.regions import ROOT,sha256_file

STUDY=ROOT/'results/deployment';MODELS=ROOT/'model/deployment'
VERSION='communications-deployment-v3.1'
SOURCES=(*V3_SOURCES,'cooperative/deployment_run.py','cooperative/deployment_policy.py','cooperative/deployment.py')
BLENDS={'consensus_025':.25,'consensus_050':.5,'consensus_075':.75}

def fingerprint():return {p:sha256_file(ROOT/p) for p in SOURCES}

def plan():
    base=v3_protocol();source=fingerprint()
    blend={k:dict(controller='consensus_ppo',checkpoint=f'model/deployment/{k}.pt',profile=base['teacher_profile']) for k in BLENDS}
    confirmation=[]
    for spec in base['test']:
        token=VERSION+':confirmation:'+':'.join(str(spec[k]) for k in ['region','direction','kind','seed'])
        confirmation.append(dict(spec,seed=int(hashlib.sha256(token.encode()).hexdigest()[:8],16)%(2**31-3000004)))
    data=dict(format=VERSION,source_sha256=source,base_source_sha256=base['source_sha256'],
        parent_v3_plan_sha256=sha256_file(ROOT/'configs/continuity_plan.json'),
        previous_v3_test_reclassified_as_development=True,
        reason='v3 validation winner guard lost to v2 on v3 test; require new untouched confirmation',
        development={p:base[p] for p in ['validation','test']},blend_candidates=blend,
        reference_candidates=base['candidates'],confirmation=confirmation,
        old_policy=dict(path='model/refinement/risk_ensemble.pt',sha256=sha256_file(ROOT/'model/refinement/risk_ensemble.pt')),
        priority_policy=dict(path='model/continuity/ensemble.pt',sha256=sha256_file(ROOT/'model/continuity/ensemble.pt')),
        teacher_profile=base['teacher_profile'],mixture_weights=BLENDS,
        selection='complete,safe,max duration<=1.15,PDR >= v2 PPO-.001 in each development phase; maximize worst phase CTR gain vs v2 PPO, then mean gain, timely PDR',
        promotion_gate='frozen candidate must have confirmation map-macro CTR >= v2 PPO and PDR >= v2 PPO-.0005, all complete/safe/max duration<=1.15; otherwise retain v2 PPO',
        confirmation_used_for_candidate_selection=False)
    path=ROOT/'configs/deployment_plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=data:raise ValueError('Frozen deployment study changed')
    if not path.exists():atomic_json(path,data)
    MODELS.mkdir(parents=True,exist_ok=True)
    for tag,alpha in BLENDS.items():
        path=ROOT/blend[tag]['checkpoint']
        expected=dict(format=FORMAT,alpha=alpha,teacher_profile=data['teacher_profile'],use_risk=True,
            old_policy=data['old_policy'],priority_policy=data['priority_policy'],deployment_sources=source)
        if path.exists():
            if torch.load(path,map_location='cpu',weights_only=False)!=expected:raise ValueError('Consensus descriptor changed')
        else:torch.save(expected,path)
    return data

def job(spec,candidate,tag,phase,sources):
    stem=f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
    return dict(spec=spec,candidate=candidate,tag=tag,prefix=str(STUDY/phase/tag/stem),sources=sources)

def rollout(item):
    torch.set_num_threads(1)
    c=item['candidate'];ck=ROOT/c['checkpoint'] if c['checkpoint'] else None
    sha=sha256_file(ck) if ck else None
    old=validated_result(item['prefix'],item['spec'],sha)
    if old and old.get('deployment_sources')==item['sources'] and old['controller']==c['controller'] and old['teacher_profile']==c['profile']:return old
    return episode(item['spec'],c['controller'],str(ck) if ck else None,item['prefix'],
        profile=c['profile'] if c['profile']!='v2' else 'linear',sources=item['sources'])

def collect(tasks,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out=[]
        for f in as_completed([pool.submit(rollout,j) for j in tasks]):
            r=f.result();out.append(r)
            print(json.dumps(dict(completed=len(out),total=len(tasks),controller=r['controller'],region=r['region'],CTR=r['search_connected_mean'])),flush=True)
    return out

def load_groups(p,phase,candidates,specs):
    groups={}
    for tag,c in candidates.items():
        rows=[]
        sha=sha256_file(ROOT/c['checkpoint']) if c['checkpoint'] else None
        for s in specs:
            j=job(s,c,tag,phase,p['source_sha256']);r=validated_result(j['prefix'],s,sha)
            if not r or r.get('deployment_sources')!=p['source_sha256'] or r['controller']!=c['controller'] or r['teacher_profile']!=c['profile']:
                raise ValueError('Missing or changed deployment episode: '+j['prefix'])
            rows.append(r)
        groups[tag]=sorted(rows,key=key)
        atomic_json(STUDY/phase/f'{tag}_complete.json',dict(candidate=c,results=groups[tag],summary=summarize(groups[tag])))
    return groups

def develop(p,workers):
    summaries={};all_candidates={**p['reference_candidates'],**p['blend_candidates']}
    for phase,specs in p['development'].items():
        target='development_'+phase
        collect([job(s,c,t,target,p['source_sha256']) for t,c in p['blend_candidates'].items() for s in specs],workers)
        groups=load_groups(p,target,p['blend_candidates'],specs)
        for tag,c in p['reference_candidates'].items():
            raw=json.loads((ROOT/'results/continuity'/phase/f'{tag}_complete.json').read_text(encoding='utf-8'))
            if {key(r) for r in raw['results']}!={key(s) for s in specs}:raise ValueError('v3 development scenario mismatch')
            sha=sha256_file(ROOT/c['checkpoint']) if c['checkpoint'] else None
            for r in raw['results']:
                s={k:r[k] for k in ['region','direction','kind','seed']};stem=f'{s["region"]}_{s["direction"]}_{s["kind"]}_{s["seed"]}'
                actual=validated_result(ROOT/'results/continuity'/phase/tag/stem,s,sha)
                if not actual or actual.get('continuity_sources')!=p['base_source_sha256']:raise ValueError('v3 development raw data changed')
            groups[tag]=raw['results']
        summaries[phase]={tag:summarize(rows) for tag,rows in groups.items()}
    feasible={};score={}
    for tag in all_candidates:
        feasible[tag]=all(summaries[phase][tag]['complete_count']==len(p['development'][phase])
            and summaries[phase][tag]['safety_violations']==0 and summaries[phase][tag]['max_duration_ratio']<=1.15
            and summaries[phase][tag]['macro']['pdr']>=summaries[phase]['v2_ppo']['macro']['pdr']-.001 for phase in summaries)
        deltas=[summaries[phase][tag]['macro']['search_connected_mean']-summaries[phase]['v2_ppo']['macro']['search_connected_mean'] for phase in summaries]
        score[tag]=[min(deltas),float(np.mean(deltas)),float(np.mean([summaries[phase][tag]['macro']['timely_pdr'] for phase in summaries]))]
    selected=max([k for k in all_candidates if feasible[k]],key=lambda k:tuple(score[k]))
    confirmation={k:all_candidates[k] for k in ['nominal','v2_mpc','v2_ppo','v2_no_risk_ppo','priority_ppo']}
    confirmation[selected]=all_candidates[selected]
    if len(confirmation)<6:confirmation['priority_ppo_guarded']=all_candidates['priority_ppo_guarded']
    if len(confirmation)!=6:raise ValueError('Six prespecified confirmation comparisons required')
    freeze=dict(format=VERSION,selected_candidate=selected,selected_policy=all_candidates[selected],
        candidates=all_candidates,development_summaries=summaries,feasible=feasible,robust_score=score,
        confirmation_candidates=confirmation,confirmation_specs=p['confirmation'],source_sha256=p['source_sha256'],
        checkpoint_sha256={k:sha256_file(ROOT/c['checkpoint']) for k,c in all_candidates.items() if c['checkpoint']},
        confirmation_used_for_selection=False,promotion_gate=p['promotion_gate'])
    path=STUDY/'selection.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=freeze:raise ValueError('Deployment selection changed')
    atomic_json(path,freeze)
    print(json.dumps(dict(frozen=selected,score=score[selected],confirmation_cases=192)),flush=True)
    return freeze

def verify_selection(p):
    f=json.loads((STUDY/'selection.json').read_text(encoding='utf-8'))
    if (f['source_sha256']!=p['source_sha256'] or f['confirmation_specs']!=p['confirmation'] or f['confirmation_used_for_selection']
        or f['selected_policy']!=f['candidates'][f['selected_candidate']] or not f['feasible'][f['selected_candidate']]):raise ValueError('Invalid deployment freeze')
    for tag,sha in f['checkpoint_sha256'].items():
        if sha256_file(ROOT/f['candidates'][tag]['checkpoint'])!=sha:raise ValueError('Deployment checkpoint changed')
    return f

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--stage',choices=['all','development','confirmation','report'],default='all');a.add_argument('--workers',type=int,default=8);args=a.parse_args()
    if args.workers<1:a.error('workers>=1')
    STUDY.mkdir(parents=True,exist_ok=True)
    with closing(TrainingLock(STUDY/'study.lock')):
        p=plan()
        if args.stage in ['all','development']:
            atomic_json(STUDY/'progress.json',dict(stage='development'));develop(p,args.workers)
        if args.stage in ['all','confirmation']:
            atomic_json(STUDY/'progress.json',dict(stage='confirmation'))
            f=verify_selection(p)
            collect([job(s,c,t,'confirmation',p['source_sha256']) for t,c in f['confirmation_candidates'].items() for s in p['confirmation']],args.workers)
            load_groups(p,'confirmation',f['confirmation_candidates'],p['confirmation'])
        if args.stage in ['all','report']:
            atomic_json(STUDY/'progress.json',dict(stage='report'))
            from cooperative.deployment_report import analyze,build
            analyze();print(build(),flush=True)
        atomic_json(STUDY/'progress.json',dict(stage='complete'))

if __name__=='__main__':main()
