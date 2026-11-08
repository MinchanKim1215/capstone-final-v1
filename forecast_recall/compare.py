"""Recall-oriented choices before own new confirmation; threshold-only control."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status
from .evaluate import evaluate,verify_freeze
from .calibration import fit,apply
from .inference import pin


def specification():
    baseline=dict(task='service',tag='frozen_parent',family='baseline');threshold=dict(task='service',tag='threshold_only_parent',family='baseline')
    candidates=[baseline,threshold]
    for variant in ('bce','event','memory_event'):
        c=dict(task='service',tag='v13_'+variant,family='new',model=str(ARTIFACTS/'models'/('v13_'+variant)/'model.json'));candidates.append(c)
        if variant!='bce':candidates.append(dict(task='service',tag='v13_'+variant+'_blend50',family='blend',members=[baseline,c]))
    p=ARTIFACTS/'comparison_spec.json';v=dict(candidates=candidates,reference='frozen_parent',threshold_reference='threshold_only_parent',
        selection='Frozen parent reference only; every other candidate selected by exact event-recall/FA score from full selection45. No confirmation tuning.',primary='Service first-relay event recall',original_goals_unchanged=True)
    if p.exists():
        if read(p)!=v:raise ValueError('Comparison specification changed')
    else:write(p,v)
    return v


def main():
    verify();spec=specification();freeze=ARTIFACTS/'freeze.json'
    if (ARTIFACTS/'completed_comparison.json').exists():verify_freeze();return
    if not freeze.exists():
        status('event_recall_selection');paths=[evaluate(c,'selection') for c in spec['candidates']]
        summaries=[read(p) for p in paths if read(p)['model_tag']!='frozen_parent'];selected=max(summaries,key=lambda s:tuple(s['selection_score']))
        decision=dict(candidate=selected['candidate'],selection_event_score=selected['selection_event_score'],selection_summary=next(str(p) for p in paths if read(p)['candidate']==selected['candidate']),confirmation_used=False,created_at=now())
        write(ARTIFACTS/'selection_decision.json',decision);items=[];status('event_recall_calibration')
        for candidate,sel in zip(spec['candidates'],paths):
            cal=evaluate(candidate,'calibration');policy=fit(cal);calibrated=apply(cal,policy)
            items.append(dict(candidate=candidate,pin=pin(candidate['model']) if candidate['family']=='new' else None,
                selection_summary=str(sel),selection_summary_sha256=sha(sel),calibration_summary=str(cal),calibration_summary_sha256=sha(cal),
                policy=str(policy),policy_sha256=sha(policy),calibrated_calibration=str(calibrated)))
        names=['__init__.py','contracts.py','data.py','features.py','embedding.py','model.py','train.py','episodes.py','metrics.py','inference.py','evaluate.py','calibration.py','references.py','confirmation.py','runtime.py','compare.py']
        write(freeze,dict(created_at=now(),sources={name:sha(ROOT/name) for name in names},candidates=items,decision=decision,
            data_manifest_sha256=sha(ARTIFACTS/'data_manifest.json'),training_plan_sha256=sha(ARTIFACTS/'training_plan.json'),comparison_spec_sha256=sha(ARTIFACTS/'comparison_spec.json'),
            original_final_test_opened=False,model_reselection_forbidden=True,production_ready=False,hardware_validated=False))
    f=verify_freeze()
    from .confirmation import main as collect
    collect();status('event_recall_frozen_new_confirmation');outcomes=[]
    for item in f['candidates']:
        raw=evaluate(item['candidate'],'confirmation');calibrated=apply(raw,item['policy']);outcomes.append(dict(candidate=item['candidate'],raw_summary=str(raw),calibrated_summary=str(calibrated)))
    verify_freeze();write(ARTIFACTS/'completed_comparison.json',dict(created_at=now(),decision=f['decision'],outcomes=outcomes,freeze_sha256=sha(freeze),original_final_test_opened=False,production_ready=False))


if __name__=='__main__':main()
