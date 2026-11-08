"""Finite post-comparison integration, audit, robustness and report workflow."""
import time
from . import ARTIFACTS
from .contracts import read,write,now
from .campaign import run


def stage(name):write(ARTIFACTS/'post_status.json',dict(state='running',stage=name,updated_at=now()))


def main():
    stage('waiting_for_frozen_fresh_flight_comparison')
    while not (ARTIFACTS/'completed_comparison.json').exists():
        status=ARTIFACTS/'status.json'
        if status.exists() and read(status)['state']=='failed':raise RuntimeError('Main experiment failed; inspect its saved log')
        if (ARTIFACTS/'dataset.json').exists() and not (ARTIFACTS/'models/new_data_20000/model.json').exists() and not (ARTIFACTS/'reuse_complete.json').exists():
            run('forecast_extension.reuse',[],'reuse_verified_predictions')
            write(ARTIFACTS/'reuse_complete.json',dict(completed_at=now(),scope='Identical old-selection inputs and frozen model pins only'))
        time.sleep(10)
    stage('whole_flight_runtime_equivalence');run('forecast_extension.check_runtime',[],'runtime_check')
    stage('raw_model_and_sample_provenance_audit');run('forecast_extension.audit',[],'audit')
    stage('paired_spatial_analysis');run('forecast_extension.analyze',[],'analysis')
    stage('measurement_robustness_diagnostic');run('forecast_extension.stress',[],'stress')
    stage('figures_and_report');run('forecast_extension.figures',[],'figures');run('forecast_extension.report',[],'report')
    write(ARTIFACTS/'post_status.json',dict(state='completed',stage='audits_and_report_complete',updated_at=now()))
    result=read(ARTIFACTS/'final_results.json');selected=result['models'][result['selected_tag']]['calibrated']
    write(ARTIFACTS/'status.json',dict(state='completed',stage='three_steps_and_audit_completed',updated_at=now(),selected_tag=result['selected_tag'],
        simulation_goal_met=selected['trusted_simulation_alarm_enabled'],production_ready=False,hardware_validated=False,original_final_test_opened=False))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ARTIFACTS/'post_status.json',dict(state='failed',error=repr(error),updated_at=now()));raise
