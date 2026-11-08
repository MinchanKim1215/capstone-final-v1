"""Finish actual execution proof and evidence before marking completion."""
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now,status
from .evaluate import verify_freeze


def export():
    f=verify_freeze();item=next(i for i in f['candidates'] if i['candidate']==f['decision']['candidate']);parent=read(PARENT/'research_candidates.json');deps=dict(parent['dependencies'])
    for p in (PARENT/'freeze.json',PARENT/'research_candidates.json',PARENT.parent/'runtime.py'):
        deps[str(p)]=sha(p)
    for name in ('features.py','embedding.py','model.py','inference.py'):
        p=ROOT/name;deps[str(p)]=sha(p)
    value=dict(candidate=item['candidate'],policy=item['policy'],policy_sha256=item['policy_sha256'],dependencies=deps,
        freeze_sha256=sha(ARTIFACTS/'freeze.json'),runtime_sha256=sha(ROOT/'runtime.py'),created_at=now(),
        primary='First-relay service only',production_ready=False,hardware_validated=False,automatic_PPO_replacement=False)
    path=ARTIFACTS/'research_candidate.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Export changed')
    else:write(path,value)


def main():
    status('actual_frozen_runtime_verification');export();path=ARTIFACTS/'runtime_checks.json';cached=read(path) if path.exists() else None
    from .check_runtime import main as runtime
    if not(cached and cached['passed'] and cached['export_sha256']==sha(ARTIFACTS/'research_candidate.json') and cached['runtime_sha256']==sha(ROOT/'runtime.py')
           and cached['checker_sha256']==sha(ROOT/'check_runtime.py') and cached['freeze_sha256']==sha(ARTIFACTS/'freeze.json')):runtime()
    status('event_recall_integrity_audit')
    from .audit import main as audit
    from .analyze import main as analyze
    from .figures import main as figures
    from .report import main as report
    from .operating_points import analyze as operating_points
    audit();analyze();operating_points();figures();report();verify_freeze()
    write(ARTIFACTS/'status.json',dict(state='completed',stage='event_training_new_confirmation_runtime_audit_report_completed',updated_at=now(),
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,original_final_test_opened=False))


if __name__=='__main__':main()
