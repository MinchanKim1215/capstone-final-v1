"""Finite predeclared comparison: select, calibrate, freeze, confirm once."""
from pathlib import Path
import subprocess
import sys
import time
from . import ARTIFACTS,WORKSPACE
from .contracts import read,write,sha,now,verify_prior
from .evaluate import evaluate
from .blend import create
from .calibration import fit,apply,freeze,verify_freeze,audit_cohort


def command(module,args,tag):
    folder=ARTIFACTS/'completion';folder.mkdir(exist_ok=True)
    with (folder/(tag+'.stdout.log')).open('w',encoding='utf-8') as out,(folder/(tag+'.stderr.log')).open('w',encoding='utf-8') as err:
        result=subprocess.run([sys.executable,'-u','-B','-m',module,*args],cwd=WORKSPACE,stdout=out,stderr=err)
    if result.returncode:raise RuntimeError(tag+' failed; inspect the saved completion log')


def stage(name,**extra):
    write(ARTIFACTS/'completion_status.json',dict(state='running',stage=name,updated_at=now(),**extra))


def main():
    verify_prior();folder=ARTIFACTS/'completion';folder.mkdir(exist_ok=True)
    while not (ARTIFACTS/'benchmarks/balanced_selection_results.json').exists():
        stage('waiting_for_balanced_selection')
        time.sleep(10)
    results=read(ARTIFACTS/'benchmarks/balanced_selection_results.json')
    dataset=Path(results['dataset_path']);paths=[Path(p) for p in results['results']]
    summaries=[read(p) for p in paths]
    for s in summaries:audit_cohort(s)
    trees=[(s,p) for s,p in zip(summaries,paths) if s['model_manifest']!='baseline' and read(s['model_manifest'])['family']=='lightgbm']
    neural=[(s,p) for s,p in zip(summaries,paths) if s['model_manifest']!='baseline' and read(s['model_manifest'])['family']!='lightgbm']
    tree,tree_path=max(trees,key=lambda x:x[0]['selection_score'])
    network,network_path=max(neural,key=lambda x:x[0]['selection_score'])
    stage('selection_only_blending',tree=tree['model_tag'],neural=network['model_tag'])
    for weight in (.25,.5,.75):
        path=create(str(tree_path),str(network_path),weight,'balanced_blend_'+str(int(weight*100)))
        paths.append(Path(path));summaries.append(read(path))
    selected_index=max(range(len(summaries)),key=lambda i:summaries[i]['selection_score'])
    selected=summaries[selected_index];selected_path=paths[selected_index]
    decision=dict(selected_summary=str(selected_path),selected_model=selected['model_manifest'],selected_tag=selected['model_tag'],
        criterion='Maximum arithmetic mean of common 5s AP and common any3to7s AP on 45 balanced selection flights',
        candidates=[dict(tag=s['model_tag'],score=s['selection_score'],summary=str(p)) for s,p in zip(summaries,paths)],
        calibration_outcomes_used=False,confirmation_outcomes_used=False,created_at=now())
    write(folder/'selection_decision.json',decision)
    model=str(Path(selected['model_manifest']).resolve()) if selected['model_manifest']!='baseline' else 'baseline'
    if model!=selected['model_manifest']:raise ValueError('Freeze requires the exact absolute evaluated model path')
    stage('calibration',selected_tag=selected['model_tag'])
    selected_cal=evaluate(model,'calibration','cuda',dataset)
    baseline_cal=evaluate('baseline','calibration','cuda',dataset)
    selected_policy=fit(selected_cal);baseline_policy=fit(baseline_cal)
    selected_calibrated=apply(selected_cal,selected_policy);baseline_calibrated=apply(baseline_cal,baseline_policy)
    stage('freezing',selected_tag=selected['model_tag'])
    frozen=freeze(selected_path,selected_policy,baseline_policy);verify_freeze()
    command('forecast_rebuild.prepare_data',['--workers','3','--include-confirmation'],'prepare_confirmation')
    if model!='baseline' and read(model).get('use_plan',False):
        command('forecast_rebuild.prepare_plans',['--workers','2','--confirmation'],'prepare_confirmation_plans')
    data=read(ARTIFACTS/'dataset.json')
    rows=[r for r in data['rows'] if r['role']=='train' or r['source_kind']=='fresh_v7']
    confirmation=folder/'balanced_confirmation_dataset.json'
    write(confirmation,dict(rows=rows,cohort='One fresh flight per development map and RF condition; disjoint geography folds',
        freeze_sha256=sha(frozen),confirmation_opened=True,final_test_opened=False,created_at=now()))
    stage('independent_confirmation',selected_tag=selected['model_tag'])
    selected_confirm=evaluate(model,'confirmation','cuda',confirmation)
    baseline_confirm=evaluate('baseline','confirmation','cuda',confirmation)
    selected_confirm_cal=apply(selected_confirm,selected_policy);baseline_confirm_cal=apply(baseline_confirm,baseline_policy)
    verify_freeze();pins=verify_prior()
    value=dict(created_at=now(),selection_decision=str(folder/'selection_decision.json'),freeze_path=str(frozen),freeze_sha256=sha(frozen),
        selected_tag=selected['model_tag'],selected_model=model,selected_selection=str(selected_path),
        selected_calibration=str(selected_cal),baseline_calibration=str(baseline_cal),
        selected_calibrated_calibration=str(selected_calibrated),baseline_calibrated_calibration=str(baseline_calibrated),
        selected_confirmation=str(selected_confirm),baseline_confirmation=str(baseline_confirm),
        selected_calibrated_confirmation=str(selected_confirm_cal),baseline_calibrated_confirmation=str(baseline_confirm_cal),
        selected_alarm=str(selected_policy),baseline_alarm=str(baseline_policy),prior_pins=pins,
        production_ready=False,hardware_validated=False,original_final_test_opened=False,
        scope='Prediction-only comparison on identical fixed-policy flights. No new closed-loop connectivity claim.')
    write(ARTIFACTS/'completed_comparison.json',value)
    write(ARTIFACTS/'completion_status.json',dict(state='completed',stage='comparison_complete',selected_tag=selected['model_tag'],updated_at=now()))
    print(ARTIFACTS/'completed_comparison.json',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ARTIFACTS/'completion_status.json',dict(state='failed',error=repr(error),updated_at=now()))
        raise
