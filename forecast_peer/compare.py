"""Prospective model choice and frozen policies before any fresh outcomes."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status,CORE
from .evaluate import evaluate,verify_freeze
from .calibration import fit,apply
from .inference import pin

def specification():
    base=dict(task='service',tag='frozen_v13',family='baseline');retuned=dict(task='service',tag='retuned_v13',family='baseline');candidates=[base,retuned]
    for variant in ('global_event','peer_bce','peer_event'):
        c=dict(task='service',tag='v15_'+variant,family='new',model=str(ARTIFACTS/'models'/('v15_'+variant)/'model.json'));candidates.append(c)
        if variant!='peer_bce':candidates.append(dict(task='service',tag='v15_'+variant+'_blend50',family='blend',members=[base,c]))
    v=dict(candidates=candidates,reference='frozen_v13',threshold_reference='retuned_v13',selection='Highest exact FA4 selection score among all except fixed V13; retuned V13 is eligible. No confirmation selection.')
    path=ARTIFACTS/'comparison_spec.json'
    if path.exists() and read(path)!=v:raise ValueError('Comparison changed')
    if not path.exists():write(path,v)
    return v

def main():
    verify();spec=specification();freeze=ARTIFACTS/'freeze.json'
    if (ARTIFACTS/'completed_comparison.json').exists():verify_freeze();return
    if not freeze.exists():
        status('temporal_selection');paths=[evaluate(c,'selection') for c in spec['candidates']]
        selected=max([read(p) for p in paths if read(p)['model_tag']!='frozen_v13'],key=lambda s:tuple(s['selection_score']))
        decision=dict(candidate=selected['candidate'],selection_event_score=selected['selection_event_score'],selection_summary=next(str(p) for p in paths if read(p)['candidate']==selected['candidate']),confirmation_used=False,created_at=now())
        write(ARTIFACTS/'selection_decision.json',decision);items=[];status('temporal_calibration')
        for candidate,sel in zip(spec['candidates'],paths):
            cal=evaluate(candidate,'calibration');policy=fit(cal);calibrated=apply(cal,policy)
            items.append(dict(candidate=candidate,pin=pin(candidate['model']) if candidate['family']=='new' else None,selection_summary=str(sel),selection_summary_sha256=sha(sel),calibration_summary=str(cal),
                calibration_summary_sha256=sha(cal),policy=str(policy),policy_sha256=sha(policy),calibrated_calibration=str(calibrated)))
        write(freeze,dict(created_at=now(),sources={name:sha(ROOT/name) for name in CORE},candidates=items,decision=decision,data_manifest_sha256=sha(ARTIFACTS/'data_manifest.json'),
            training_plan_sha256=sha(ARTIFACTS/'training_plan.json'),comparison_spec_sha256=sha(ARTIFACTS/'comparison_spec.json'),original_final_test_opened=False,model_reselection_forbidden=True,production_ready=False))
    f=verify_freeze()
    from .confirmation import main as collect
    collect();status('temporal_frozen_confirmation');outcomes=[]
    for item in f['candidates']:
        raw=evaluate(item['candidate'],'confirmation');calibrated=apply(raw,item['policy']);outcomes.append(dict(candidate=item['candidate'],raw_summary=str(raw),calibrated_summary=str(calibrated)))
    verify_freeze();write(ARTIFACTS/'completed_comparison.json',dict(created_at=now(),decision=f['decision'],outcomes=outcomes,freeze_sha256=sha(freeze),original_final_test_opened=False,production_ready=False))

if __name__=='__main__':main()
