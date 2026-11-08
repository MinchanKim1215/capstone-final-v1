"""Finite authorized sequence; no changes to predecessor implementations."""
from pathlib import Path
import os
import subprocess
import sys
import time
from . import ARTIFACTS,WORKSPACE
from .contracts import read,write,sha,now,prepare,OLD_DEVELOPMENT,OLD_NEURAL,OLD_TREE
from .selection import prepare_plan,select
from .evaluate import evaluate,freeze,model_pin,verify_freeze
from .calibration import fit,apply


def stage(name,**extra):write(ARTIFACTS/'status.json',dict(state='running',stage=name,updated_at=now(),original_final_test_opened=False,**extra))


def start(module,args,tag):
    folder=ARTIFACTS/'logs';folder.mkdir(exist_ok=True)
    out=(folder/(tag+'.stdout.log')).open('w',encoding='utf-8');err=(folder/(tag+'.stderr.log')).open('w',encoding='utf-8')
    job=subprocess.Popen([sys.executable,'-u','-B','-m',module,*args],cwd=WORKSPACE,stdout=out,stderr=err,
        env=dict(os.environ,PYTHONIOENCODING='utf-8'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return job,out,err


def finish(job,tag):
    process,out,err=job;code=process.wait();out.close();err.close()
    if code:raise RuntimeError(tag+' failed; inspect the saved log')


def run(module,args,tag):finish(start(module,args,tag),tag)


def main():
    prepare();prepare_plan();stage('stage1_same_data_continuation')
    while not (ARTIFACTS/'models/same_data_20000/model.json').exists():
        status=ARTIFACTS/'models/same_data_20000/status.json'
        if status.exists() and read(status)['state']=='failed':raise RuntimeError('Same-data continuation failed')
        time.sleep(5)
    stage('stage2_fresh_experience_collection')
    collector=start('forecast_extension.collect',[],'fresh_collection')
    preparer=start('forecast_extension.stream_prepare',[],'stream_preparation')
    # GPU prediction is independent of collecting fixed-policy CPU flights.
    stage1=[]
    for model in (str(OLD_NEURAL),str(ARTIFACTS/'models/same_data_20000/model.json'),str(OLD_TREE)):
        stage1.append(str(evaluate(model,'selection',OLD_DEVELOPMENT)))
    write(ARTIFACTS/'stage1_comparison.json',dict(summaries=stage1,created_at=now(),scope='Same 45 old selection flights; no new data yet used in these models'))
    finish(collector,'fresh_collection');finish(preparer,'stream_preparation')
    stage('stage2_finalize_data_and_observation_augmentation')
    run('forecast_extension.data',[],'prepare_data')
    run('forecast_extension.build_samples',[],'new_samples')
    run('forecast_extension.build_samples',['--merged'],'merged_tree_samples')
    stage('stage2_new_experience_training')
    neural=start('forecast_extension.train',['--tag','new_data_20000','--new-samples','forecast_extension/artifacts/samples/new_only'],'new_data_training')
    run('forecast_extension.train_tree',['--tag','new_data_tree'],'new_tree_energy')
    run('forecast_extension.train_tree',['--tag','new_data_tree_no_energy','--no-energy'],'new_tree_no_energy')
    finish(neural,'new_data_training')
    stage('stage3_selection_and_precision_recall_false_alarm_comparison')
    decision=select();stage('stage3_fresh_calibration',selected_tag=decision['selected_tag']);items=[]
    for summary_path in decision['summaries']:
        selection=read(summary_path);model=selection['model_manifest'];cal=evaluate(model,'calibration')
        policy=fit(cal);calibrated=apply(cal,policy)
        items.append(dict(tag=selection['model_tag'],pin=model_pin(model),selection_summary=summary_path,selection_sha256=sha(summary_path),
            calibration_summary=str(cal),calibration_sha256=sha(cal),policy=str(policy),policy_sha256=sha(policy),calibrated_calibration=str(calibrated)))
    frozen=freeze(items,decision['selected_model']);verify_freeze();stage('stage3_prepare_fresh_confirmation')
    run('forecast_extension.data',['--confirmation'],'prepare_confirmation')
    stage('stage3_fresh_flight_confirmation',selected_tag=decision['selected_tag']);outcomes=[]
    for item in items:
        summary=evaluate(item['pin']['path'],'confirmation',ARTIFACTS/'dataset_with_confirmation.json')
        calibrated=apply(summary,item['policy']);outcomes.append(dict(tag=item['tag'],raw_summary=str(summary),calibrated_summary=str(calibrated)))
    verify_freeze()
    write(ARTIFACTS/'completed_comparison.json',dict(created_at=now(),selected_tag=decision['selected_tag'],selected_model=decision['selected_model'],
        selection_decision=str(ARTIFACTS/'selection_decision.json'),stage1_comparison=str(ARTIFACTS/'stage1_comparison.json'),freeze=str(frozen),freeze_sha256=sha(frozen),
        outcomes=outcomes,production_ready=False,hardware_validated=False,original_final_test_opened=False,
        scope='Fresh-flight evaluation on known geographies, all candidates fixed before confirmation, no final-test map access'))
    stage('comparison_complete_pending_audit',selected_tag=decision['selected_tag'])
    print(ARTIFACTS/'completed_comparison.json',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',error=repr(error),updated_at=now(),original_final_test_opened=False));raise
