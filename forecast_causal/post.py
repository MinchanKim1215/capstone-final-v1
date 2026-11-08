"""Finish all four stages, including the real observable API and delivery audit."""
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now,status,milestone
from .evaluate import verify_freeze

def export():
    f=verify_freeze();item=next(i for i in f['candidates'] if i['candidate']==f['decision']['candidate']);parent=read(PARENT/'research_candidate.json');deps=dict(parent['dependencies'])
    for p in (PARENT/'freeze.json',PARENT/'research_candidate.json',PARENT/'delivery_manifest.json',PARENT.parent/'runtime.py',ARTIFACTS/'path_uncertainty.json'):
        deps[str(p)]=sha(p)
    for name in ('features.py','state.py','wire.py','physics.py','model.py','inference.py'):
        deps[str(ROOT/name)]=sha(ROOT/name)
    value=dict(candidate=item['candidate'],policy=item['policy'],policy_sha256=item['policy_sha256'],dependencies=deps,freeze_sha256=sha(ARTIFACTS/'freeze.json'),runtime_sha256=sha(ROOT/'runtime.py'),created_at=now(),
        primary='All-event first-alert recall at3<=lead<7s, calibrationFA4/h',production_ready=False,hardware_validated=False,automatic_PPO_replacement=False)
    path=ARTIFACTS/'research_candidate.json'
    if path.exists():
        if {k:v for k,v in read(path).items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Export changed')
    else:write(path,value)

def main():
    from .delivery import already_sealed
    if already_sealed():return
    export();status('stage4_actual_runtime')
    from .check_runtime import main as runtime
    from .audit import main as audit
    from .analyze import main as analyze
    from .report import main as report
    from .figures import main as figures
    path=ARTIFACTS/'runtime_checks.json';cached=read(path) if path.exists() else None
    if not(cached and cached['passed'] and cached['export_sha256']==sha(ARTIFACTS/'research_candidate.json') and cached['runtime_sha256']==sha(ROOT/'runtime.py') and cached['checker_sha256']==sha(ROOT/'check_runtime.py')):runtime()
    status('stage4_integrity_audit');audit();status('stage4_results_and_visual_QA');analyze();report();figures();verify_freeze()
    # Visual inspection and delivery checking must be completed before milestone4.
    status('stage4_awaiting_visual_delivery_check',next_action='보고서 수치와 도표를 실제로 검토하고 납품 검사 후4단계 완료를 기록한다.')

if __name__=='__main__':main()
