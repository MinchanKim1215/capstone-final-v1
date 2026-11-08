"""Resume after C/tree training with the final exact threshold arithmetic."""
import time
from . import ARTIFACTS
from .contracts import read,write,sha,now
from .campaign import stage,run
from .selection import select
from .evaluate import evaluate,freeze,model_pin,verify_freeze
from .calibration import fit,apply


def main():
    required=['new_data_20000','new_data_tree','new_data_tree_no_energy']
    while any(not (ARTIFACTS/'models'/tag/'model.json').exists() for tag in required):
        for tag in required:
            path=ARTIFACTS/'models'/tag/'status.json'
            if path.exists() and read(path)['state']=='failed':raise RuntimeError('Training failed: '+tag)
        time.sleep(5)
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
        outcomes=outcomes,production_ready=False,hardware_validated=False,original_final_test_opened=False,runner='finish.py',runner_sha256=sha(__file__),
        scope='Fresh-flight evaluation on known geographies, all candidates fixed before confirmation, no final-test map access'))
    stage('comparison_complete_pending_audit',selected_tag=decision['selected_tag'])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',error=repr(error),updated_at=now(),original_final_test_opened=False));raise
