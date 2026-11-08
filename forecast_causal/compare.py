"""Select/calibrate all12profiles before genuinely new42fixed-policy flights."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status,milestone
from .evaluate import evaluate,verify_freeze
from .calibration import fit,apply
from .inference import pin

CORE=['__init__.py','contracts.py','instrumented_packet.py','recorder.py','wire.py','replay.py','state.py','features.py','physics.py','path_uncertainty.py','data.py','inputs.py',
      'model.py','tree_train.py','neural_train.py','train.py','metrics.py','inference.py','evaluate.py','calibration.py','compare.py','legacy_data.py','references.py','confirmation.py','runtime.py']

def specification():
    old=dict(task='service',tag='frozen_v13',family='baseline');retuned=dict(task='service',tag='retuned_v13',family='baseline');candidates=[old,retuned];experts={}
    for variant in ('tree_base','tree_network','tree_physics','tree_both','gru_base','gru_network'):
        c=dict(task='service',tag='v16_'+variant,family='new',model=str(ARTIFACTS/'models'/('v16_'+variant)/'model.json'));candidates.append(c);experts[variant]=c
    for variant in ('tree_physics','tree_both','gru_network'):
        candidates.append(dict(task='service',tag='v16_old_'+variant+'50',family='blend',members=[old,experts[variant]],weights=[.5,.5]))
    candidates.append(dict(task='service',tag='v16_old50_tree_both25_gru_network25',family='blend',members=[old,experts['tree_both'],experts['gru_network']],weights=[.5,.25,.25]))
    value=dict(candidates=candidates,reference='frozen_v13',threshold_reference='retuned_v13',selection='Exact eventrecallFA4/min50 onselection45; currentfixedreferenceonly,retunedoldeligible. No model or threshold reselection onconfirmation.',primary_FA_budget=4,secondary_FA_budgets=[3,5],precision_maintenance_is_adoption_requirement=True)
    path=ARTIFACTS/'comparison_spec.json'
    if path.exists() and read(path)!=value:raise ValueError('Predeclared candidate list changed')
    if not path.exists():write(path,value)
    return value

def main():
    verify();spec=specification();freeze=ARTIFACTS/'freeze.json'
    if (ARTIFACTS/'completed_comparison.json').exists():verify_freeze();return
    if not freeze.exists():
        status('stage3_selection',next_action='선택45비행에서 모델을 정하고 보정45비행의 주·보조 문턱을 모두 고정한다.');paths=[evaluate(c,'selection') for c in spec['candidates']]
        selected=max([read(p) for p in paths if read(p)['model_tag']!='frozen_v13'],key=lambda s:tuple(s['selection_score']))
        decision=dict(candidate=selected['candidate'],selection_event_score=selected['selection_event_score'],selection_summary=next(str(p) for p in paths if read(p)['candidate']==selected['candidate']),confirmation_used=False,created_at=now());write(ARTIFACTS/'selection_decision.json',decision)
        items=[];status('stage3_calibration')
        for candidate,selection in zip(spec['candidates'],paths):
            cal=evaluate(candidate,'calibration');policy=fit(cal);calibrated=apply(cal,policy)
            items.append(dict(candidate=candidate,pin=pin(candidate['model']) if candidate['family']=='new' else None,selection_summary=str(selection),selection_summary_sha256=sha(selection),calibration_summary=str(cal),calibration_summary_sha256=sha(cal),policy=str(policy),policy_sha256=sha(policy),calibrated_calibration=str(calibrated)))
        binds=('data_manifest','training_plan','comparison_spec','protocol','experiment_plan','path_uncertainty','cause_rulebook','adoption_rule')
        write(freeze,dict(created_at=now(),sources={k:sha(ROOT/k) for k in CORE},candidates=items,decision=decision,**{k+'_sha256':sha(ARTIFACTS/(k+'.json')) for k in binds},original_final_test_opened=False,model_reselection_forbidden=True,production_ready=False))
        milestone(3,'artifacts/trained_models.json·selection_decision.json·freeze.json; 12개 후보·주FA4/보조FA3,5문턱을 확인 전 고정','새42비행을 생성해 모든 후보를 같은 관측·정답·전체 사건 분모에서 비교하고 실제 실행API와 보존 감사를 마친다.')
    frozen=verify_freeze()
    from .confirmation import main as collect
    collect();status('stage4_frozen_confirmation');outcomes=[]
    for item in frozen['candidates']:
        raw=evaluate(item['candidate'],'confirmation');calibrated=apply(raw,item['policy']);outcomes.append(dict(candidate=item['candidate'],raw_summary=str(raw),calibrated_summary=str(calibrated)))
    verify_freeze();write(ARTIFACTS/'completed_comparison.json',dict(created_at=now(),decision=frozen['decision'],outcomes=outcomes,freeze_sha256=sha(freeze),new_confirmation_flights=42,original_final_test_opened=False,production_ready=False))

if __name__=='__main__':main()
