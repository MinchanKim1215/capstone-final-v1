"""Finite integration/stress/provenance diagnostics without model reselection."""
import time
from . import ARTIFACTS
from .contracts import read,write,now
from .finish_study import command
from .calibration import verify_freeze


def status(stage,state='running'):
    write(ARTIFACTS/'post_checks_status.json',dict(stage=stage,state=state,updated_at=now()))


def main():
    status('waiting_for_freeze')
    while not (ARTIFACTS/'model_alarm_freeze.json').exists():time.sleep(10)
    frozen=verify_freeze();status('selected_runtime_equivalence')
    runtime_path=ARTIFACTS/'selected_runtime_checks.json'
    runtime=read(runtime_path) if runtime_path.exists() else {}
    if runtime.get('model_tag')!=frozen['selected_tag'] or runtime.get('runtime_adapter')!='runtime_stable.StableLocalForecast' or runtime.get('tested_link_predictions',0)<1000:
        command('forecast_rebuild.check_selected_runtime',[],'selected_runtime')
    status('selection_measurement_stress')
    command('forecast_rebuild.stress',['--summary',frozen['selection_summary'],'--policy',frozen['selected_alarm']],'measurement_stress')
    status('waiting_for_confirmation')
    while not (ARTIFACTS/'completed_comparison.json').exists():time.sleep(10)
    status('paired_spatial_analysis')
    command('forecast_rebuild.analyze_results',[],'analyze_results')
    status('preservation_and_source_audit')
    command('forecast_rebuild.audit',[],'audit')
    verify_freeze();status('completed','completed')


if __name__=='__main__':
    try:main()
    except BaseException:
        status('inspect_preserved_error_log','failed');raise
