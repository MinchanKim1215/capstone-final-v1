"""Export exact frozen choices, verify actual APIs and evidence, then report."""
from . import ROOT,ARTIFACTS,WORKSPACE
from .contracts import read,write,sha,now,status,V10
from .evaluate import verify_freeze


def export():
    frozen=verify_freeze();tasks={}
    for task,decision in frozen['decisions'].items():
        item=next(i for i in frozen['candidates'] if i['candidate']==decision['candidate'])
        tasks[task]=dict(candidate=decision['candidate'],policy=item['policy'],policy_sha256=item['policy_sha256'])
    dependencies=[WORKSPACE/'forecast_rebuild/runtime_stable.py',WORKSPACE/'forecast_rebuild/runtime.py',
        WORKSPACE/'forecast_rebuild/features.py',WORKSPACE/'forecast_rebuild/online_planning.py',
        WORKSPACE/'forecast_service/features.py',WORKSPACE/'forecast_service/inference.py']
    parent=read(V10/'research_candidates.json')
    files={str(p):sha(p) for p in dependencies}
    for entry in parent['tasks'].values():files.update(entry['pin']['files'])
    value=dict(tasks=tasks,freeze_sha256=sha(ARTIFACTS/'freeze.json'),runtime_sha256=sha(ROOT/'runtime.py'),runtime_dependencies=files,
        probability_start_s=20.,alarm_start_s=21.,primary_observer='first relay only',
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,created_at=now())
    path=ARTIFACTS/'research_candidates.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Export cannot silently change')
    else:write(path,value)


def main():
    status('export_and_runtime_verification');export()
    from .check_runtime import main as runtime
    previous=ARTIFACTS/'runtime_checks.json'
    cached=read(previous) if previous.exists() else None
    if not (cached and cached['passed'] and cached['export_sha256']==sha(ARTIFACTS/'research_candidates.json')
            and cached['runtime_sha256']==sha(ROOT/'runtime.py') and cached['checker_sha256']==sha(ROOT/'check_runtime.py')
            and cached['freeze_sha256']==sha(ARTIFACTS/'freeze.json')):runtime()
    status('integrity_and_causal_audit')
    from .audit import main as audit
    audit()
    from .analyze import main as analyze
    from .figures import main as figures
    from .report import main as report
    analyze();figures();report();verify_freeze()
    write(ARTIFACTS/'status.json',dict(state='completed',stage='training_confirmation_audit_report_completed',updated_at=now(),
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,original_final_test_opened=False))


if __name__=='__main__':main()
