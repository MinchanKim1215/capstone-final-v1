"""Completion requires actual runtime, integrity, matched results and reports."""
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now,status
from .evaluate import verify_freeze

def export():
    f=verify_freeze();item=next(i for i in f['candidates'] if i['candidate']==f['decision']['candidate']);parent=read(PARENT/'research_candidate.json');deps=dict(parent['dependencies'])
    for p in (PARENT/'freeze.json',PARENT/'research_candidate.json',PARENT/'delivery_manifest.json',PARENT.parent/'runtime.py'):deps[str(p)]=sha(p)
    for name in ('features.py','model.py','inference.py'):deps[str(ROOT/name)]=sha(ROOT/name)
    value=dict(candidate=item['candidate'],policy=item['policy'],policy_sha256=item['policy_sha256'],dependencies=deps,freeze_sha256=sha(ARTIFACTS/'freeze.json'),runtime_sha256=sha(ROOT/'runtime.py'),
        created_at=now(),primary='First-relay service recall atcalibrationFA4/h',production_ready=False,hardware_validated=False,automatic_PPO_replacement=False)
    path=ARTIFACTS/'research_candidate.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Export changed')
    else:write(path,value)

def main():
    export();status('temporal_actual_runtime')
    from .check_runtime import main as runtime
    from .audit import main as audit
    from .analyze import main as analyze
    from .report import main as report
    from .figures import main as figures
    path=ARTIFACTS/'runtime_checks.json';cached=read(path) if path.exists() else None
    if not(cached and cached['passed'] and cached['export_sha256']==sha(ARTIFACTS/'research_candidate.json') and cached['runtime_sha256']==sha(ROOT/'runtime.py') and cached['checker_sha256']==sha(ROOT/'check_runtime.py')):runtime()
    status('temporal_integrity_audit');audit();analyze();report();figures();verify_freeze()
    write(ARTIFACTS/'status.json',dict(state='completed',stage='temporal_training_confirmation_runtime_audit_report_completed',updated_at=now(),production_ready=False,hardware_validated=False,original_final_test_opened=False,automatic_PPO_replacement=False))

if __name__=='__main__':main()
