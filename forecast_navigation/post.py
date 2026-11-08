"""Export only frozen decisions; verify actual execution before completion."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,status,PARENT
from .evaluate import verify_freeze


def export():
    frozen=verify_freeze();tasks={}
    for task,decision in frozen['decisions'].items():
        item=next(i for i in frozen['candidates'] if i['candidate']==decision['candidate'])
        tasks[task]=dict(candidate=decision['candidate'],policy=item['policy'],policy_sha256=item['policy_sha256'])
    parent=read(PARENT/'research_candidates.json');dependencies=dict(parent['runtime_dependencies'])
    for name in ('runtime.py','features.py','codec.py','inference.py','model.py','evaluate.py','contracts.py'):
        path=PARENT.parent/name;dependencies[str(path)]=sha(path)
    for path in (PARENT/'freeze.json',PARENT/'research_candidates.json'):
        dependencies[str(path)]=sha(path)
    value=dict(created_at=now(),tasks=tasks,dependencies=dependencies,freeze_sha256=sha(ARTIFACTS/'freeze.json'),runtime_sha256=sha(ROOT/'runtime.py'),
        primary_observer='First relay only',probability_start_s=20.,alarm_start_s=21.,payload_bytes=128,
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False)
    path=ARTIFACTS/'research_candidates.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Export cannot change silently')
    else:write(path,value)


def main():
    status('export_and_actual_runtime');export()
    from .check_runtime import main as runtime
    path=ARTIFACTS/'runtime_checks.json';cached=read(path) if path.exists() else None
    if not(cached and cached['passed'] and cached['export_sha256']==sha(ARTIFACTS/'research_candidates.json')
           and cached['runtime_sha256']==sha(ROOT/'runtime.py') and cached['checker_sha256']==sha(ROOT/'check_runtime.py')
           and cached['freeze_sha256']==sha(ARTIFACTS/'freeze.json')):runtime()
    status('integrity_audit')
    from .audit import main as audit
    from .analyze import main as analyze
    from .figures import main as figures
    from .report import main as report
    audit();analyze();figures();report();verify_freeze()
    write(ARTIFACTS/'status.json',dict(state='completed',stage='controlled_training_confirmation_runtime_audit_report_completed',updated_at=now(),
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,original_final_test_opened=False))


if __name__=='__main__':main()
