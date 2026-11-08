"""Export, test, audit, analyze and report only after all frozen comparisons."""
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now,status
from .evaluate import verify_freeze
from .runtime import dependencies
from .campaign import run


def export():
    frozen=verify_freeze();tasks={}
    for task,decision in frozen['decisions'].items():
        item=next(i for i in frozen['models'] if i['task']==task and i['pin']['path']==decision['model'])
        tasks[task]=dict(model=decision['model'],tag=decision['tag'],pin=item['pin'],policy=item['policy'],policy_sha256=item['policy_sha256'])
    path=ARTIFACTS/'research_candidates.json'
    value=dict(tasks=tasks,freeze_sha256=sha(ARTIFACTS/'freeze.json'),runtime_sha256=sha(ROOT/'runtime.py'),runtime_dependencies=dependencies(),
        alarm_start_adapter_sha256=sha(ROOT/'runtime_aligned.py'),probability_start_s=20.,alarm_start_s=21.,
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,created_at=now())
    if path.exists():
        previous=read(path)
        if any(previous[k]!=v for k,v in value.items() if k!='created_at'):raise ValueError('Exported model bundle changed')
    else:write(path,value)


def main():
    if not (ARTIFACTS/'completed_comparison.json').exists():raise ValueError('Complete the frozen comparison before finalization')
    export();status('final_whole_flight_runtime_check');run('forecast_service.check_runtime','runtime')
    status('final_provenance_and_label_audit');run('forecast_service.audit','audit')
    status('final_cluster_analysis');run('forecast_service.analyze','analysis')
    status('final_figures_and_report');run('forecast_service.figures','figures');run('forecast_service.report','report')
    write(ARTIFACTS/'status.json',dict(state='completed',stage='all_four_steps_completed',updated_at=now(),production_ready=False,hardware_validated=False,original_final_test_opened=False))


if __name__=='__main__':main()
