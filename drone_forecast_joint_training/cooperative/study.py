"""기기 경로와 무관한 v2 전체 연구 재현. 완료된 원본은 무결성 검사 후 재사용."""

import argparse
import json
import os
from contextlib import closing

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')

from cooperative import refine
from cooperative.analyze import summarize,key
from cooperative.locking import TrainingLock
from swarm_sim.regions import ROOT, sha256_file

METHODS = ['nominal','reactive','mpc','mpc_no_risk','legacy_ppo',
           'risk_ensemble','risk_ensemble_guarded','no_risk_ensemble','no_risk_ensemble_guarded']


def execution_plan():
    return dict(training_modes=['risk','no_risk'],training_seeds=[27101,27102,27103],episodes_per_run=64,
                seed_diagnostic_scenarios_per_policy=14,selection_scenarios_per_method=28,
                deployable_learned_candidates=['risk_ensemble','risk_ensemble_guarded','no_risk_ensemble','no_risk_ensemble_guarded'],
                independent_seed_policies_are_diagnostics_not_best_seed_selection=True,
                test_scenarios_per_method=32,test_strata='8 regions x 2 directions x 2 jammer kinds',
                test_subset_rule='first predeclared seed in every region/direction/jammer stratum',
                test_subset_fixed_before_v2_validation=True,formal_test_methods=9)


def candidates():
    result={name:dict(controller=name,checkpoint=None,category='baseline')
            for name in ['nominal','reactive','mpc','mpc_no_risk']}
    result['legacy_ppo']=dict(controller='ppo',checkpoint='model/cooperative/best_learned.pt',category='historical_reference')
    for mode in ['risk','no_risk']:
        for seed in [27101,27102,27103]:
            name=f'{mode}_s{seed}'
            result[name]=dict(controller='ppo',checkpoint=f'model/refinement/{name}/last.pt',category='seed_diagnostic')
        result[f'{mode}_ensemble']=dict(controller='ppo',checkpoint=f'model/refinement/{mode}_ensemble.pt',category='learned')
    for mode in ['risk','no_risk']:
        result[f'{mode}_ensemble_guarded']=dict(controller='ppo_guarded',checkpoint=f'model/refinement/{mode}_ensemble.pt',category='learned')
    return result


def job(spec,candidate,tag,split,plan):
    name=f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
    ck=str(ROOT/candidate['checkpoint']) if candidate['checkpoint'] else None
    return (spec,candidate['controller'],ck,str(refine.STUDY/split/tag/name),plan['source_sha256'],None)


def signature(candidate):
    ck=candidate['checkpoint']
    return candidate['controller'],sha256_file(ROOT/ck) if ck else None


def validate_freeze(freeze,plan):
    if freeze.get('format')!='refinement-validation-selection-v2':
        raise ValueError('v2 검증 선정 형식 필요')
    if freeze['source_sha256']!=plan['source_sha256'] or freeze['test_used_for_selection']:
        raise ValueError('선정 시점의 코드·시험 사용 계약 불일치')
    if freeze['test_specs']!=plan['test'][::2] or freeze['candidates']!=candidates():
        raise ValueError('동결된 조건·후보와 다른 재현 실행')
    selected=freeze['selected']
    if selected not in freeze['candidates'] or freeze['selected_policy']!=freeze['candidates'][selected]:
        raise ValueError('선정 이름과 실행 제어기 불일치')
    expected_tags={tag for tag,c in freeze['candidates'].items() if c['checkpoint']}
    if set(freeze['checkpoint_sha256'])!=expected_tags:
        raise ValueError('모든 학습 후보의 동결 가중치 지문 필요')
    if (freeze['best_learned_risk'] not in ['risk_ensemble','risk_ensemble_guarded']
            or freeze['best_learned_no_risk'] not in ['no_risk_ensemble','no_risk_ensemble_guarded']
            or not freeze['diagnostic_candidates_excluded_from_selection']
            or not freeze['legacy_policy_excluded_from_updated_default']):
        raise ValueError('독립 학습·위험 대조군·선정 후보 계약 불일치')
    for tag,expected in freeze['checkpoint_sha256'].items():
        if sha256_file(ROOT/freeze['candidates'][tag]['checkpoint'])!=expected:
            raise ValueError('선정 이후 가중치 변경: '+tag)
    if not all(freeze['feasible'].get(tag) for tag in [selected,freeze['best_learned_risk'],freeze['best_learned_no_risk']]):
        raise ValueError('선정 정책의 검증 합격 기록 필요')
    from cooperative.contracts import verify_forecaster_files
    verify_forecaster_files()
    return freeze


def training(workers):
    plan=refine.protocol()
    for seed in plan['training_seeds']:
        for mode in plan['training_modes']:
            refine.train(argparse.Namespace(mode=mode,seed=seed,iterations=8,batch_episodes=8,workers=workers,resume=True))
    for mode in plan['training_modes']:
        destination=refine.MODELS/f'{mode}_ensemble.pt'
        members=[str(refine.MODELS/f'{mode}_s{seed}/last.pt') for seed in plan['training_seeds']]
        if destination.exists():
            import torch
            ck=torch.load(destination,map_location='cpu',weights_only=False)
            if ck.get('member_sha256')!=[sha256_file(path) for path in members]:
                raise ValueError('완료된 앙상블의 학습 원본 변경. 별도 연구 복사본 필요')
        else:
            refine.ensemble(argparse.Namespace(checkpoints=members,out=str(destination)))


def validation(workers):
    plan=refine.protocol();manifest=candidates()
    refine.atomic_json(refine.STUDY/'candidate_manifest.json',manifest)
    tasks=[]
    diagnostic=[s for s in plan['validation'] if s['direction']=='east_west']
    lookup={signature(c):tag for tag,c in manifest.items()}
    if len(lookup)!=len(manifest):raise ValueError('중복 후보')
    for tag,candidate in manifest.items():
        specs=diagnostic if candidate['category']=='seed_diagnostic' else plan['validation']
        tasks.extend(job(spec,candidate,tag,'validation',plan) for spec in specs)
    results=refine.collect(tasks,workers);groups={tag:[] for tag in manifest}
    for row in results:groups[(lookup[(row['controller'],row.get('checkpoint_sha256'))])].append(row)
    for rows in groups.values():rows.sort(key=key)
    summaries={tag:summarize(rows) for tag,rows in groups.items()}
    for tag,rows in groups.items():
        refine.atomic_json(refine.STUDY/'validation'/f'{tag}_complete.json',dict(results=rows,summary=summaries[tag],**manifest[tag]))
    base=summaries['nominal']['macro']['pdr']
    def feasible(tag):
        row=summaries[tag]
        return (row['episodes']==28 and row['complete_count']==28 and row['safety_violations']==0
                and row['max_duration_ratio']<=1.15 and row['macro']['pdr']>=base-.005)
    def rank(tag):
        m=summaries[tag]['macro']
        return (-m['search_connected_mean'],-m['timely_pdr'],m['relay_travel_m'])
    deploy=[tag for tag,c in manifest.items() if c['category'] in ['baseline','learned'] and feasible(tag)]
    risk=[tag for tag in ['risk_ensemble','risk_ensemble_guarded'] if feasible(tag)]
    no_risk=[tag for tag in ['no_risk_ensemble','no_risk_ensemble_guarded'] if feasible(tag)]
    if not deploy or not risk or not no_risk:
        raise RuntimeError('검증 합격 후보 부족. 시험을 사용해 제한을 완화하지 말고 별도 연구 필요')
    selected=min(deploy,key=rank)
    freeze=dict(format='refinement-validation-selection-v2',selected=selected,selected_policy=manifest[selected],
                best_learned_risk=min(risk,key=rank),best_learned_no_risk=min(no_risk,key=rank),
                summaries=summaries,candidates=manifest,feasible={tag:feasible(tag) for tag in manifest},
                diagnostic_candidates_excluded_from_selection=True,legacy_policy_excluded_from_updated_default=True,
                source_sha256=plan['source_sha256'],test_used_for_selection=False,test_specs=plan['test'][::2],
                test_subset_rule='first predeclared seed in every region/direction/jammer stratum',
                checkpoint_sha256={tag:signature(c)[1] for tag,c in manifest.items() if c['checkpoint']})
    path=refine.STUDY/'selection.json'
    if path.exists():
        previous=validate_freeze(json.loads(path.read_text(encoding='utf-8')),plan)
        if previous['feasible']!=freeze['feasible']:
            raise ValueError('동결된 검증 합격 여부와 재계산이 다름')
        for name in ('selected','best_learned_risk','best_learned_no_risk'):
            # 완료 순서에 따른 부동소수점 합산 차이만 있는 동률은 원래 동결 선택 유지.
            equivalent=all(abs(a-b)<=tol for a,b,tol in zip(rank(previous[name]),rank(freeze[name]),(1e-12,1e-12,1e-8)))
            if previous[name]!=freeze[name] and not equivalent:
                raise ValueError('동결된 선정과 재계산이 다름. 원본 보존 후 별도 조사 필요')
        return previous
    refine.atomic_json(path,freeze)
    return freeze


def testing(workers):
    plan=refine.protocol()
    path=refine.STUDY/'selection.json'
    if not path.exists():raise FileNotFoundError('시험 이전에 검증 선정을 먼저 수행해야 합니다.')
    freeze=validate_freeze(json.loads(path.read_text(encoding='utf-8')),plan)
    manifest={tag:freeze['candidates'][tag] for tag in METHODS}
    lookup={signature(c):tag for tag,c in manifest.items()};groups={tag:[] for tag in manifest}
    tasks=[job(spec,c,tag,'test',plan) for tag,c in manifest.items() for spec in freeze['test_specs']]
    for row in refine.collect(tasks,workers):
        groups[lookup[(row['controller'],row.get('checkpoint_sha256'))]].append(row)
    for tag,rows in groups.items():
        rows.sort(key=key)
        refine.atomic_json(refine.STUDY/'test'/f'{tag}_complete.json',dict(results=rows,summary=summarize(rows),**manifest[tag]))
    return freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','train','validation','test','report'],default='all')
    parser.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 1))
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers는 1 이상 필요')
    refine.protocol();refine.STUDY.mkdir(parents=True,exist_ok=True)
    from cooperative.contracts import verify_forecaster_files
    verify_forecaster_files()
    path=ROOT/'configs/refinement_execution.json';execution=execution_plan()
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=execution:
        raise ValueError('사전 실행 계획과 다른 예산·평가 구성')
    if not path.exists():refine.atomic_json(path,execution)
    with closing(TrainingLock(refine.STUDY/'study.lock')):
        if args.stage in ['all','train']:training(args.workers)
        if args.stage in ['all','validation']:validation(args.workers)
        if args.stage in ['all','test']:testing(args.workers)
        if args.stage in ['all','report']:
            from cooperative.refinement_analysis import analyze
            from cooperative.refinement_report import build
            analyze();print(build())


if __name__=='__main__':main()
