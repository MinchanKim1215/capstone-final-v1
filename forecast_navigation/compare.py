"""Prospective V12 choices; retain both operational and stronger rank anchors."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status,PARENT
from .inference import parent_choices,pin
from .evaluate import evaluate,verify_freeze
from .calibration import fit,apply
from forecast_graph.compare import choose


def specification():
    p=verify();parents=parent_choices();candidates=[];references={}
    for task,items in parents.items():
        selected=items['selected'];champion=items['champion']
        baseline=dict(task=task,tag='v11_selected',family='parent',parent_kind='selected',parent_candidate=selected['candidate'],parent_policy=selected['policy'])
        current=[baseline]
        if champion['candidate']!=selected['candidate']:
            anchor=dict(task=task,tag='v11_ranking_champion',family='parent',parent_kind='champion',parent_candidate=champion['candidate'],parent_policy=champion['policy'])
            current.append(anchor)
        else:anchor=baseline
        references[task]=anchor['tag']
        for profile in p['profiles']:
            candidate=dict(task=task,tag='v12_'+profile,family='new',profile=profile,model=str(ARTIFACTS/'models'/('v12_'+profile)/'model.json'))
            current.extend([candidate,dict(task=task,tag='v12_'+profile+'_blend50',family='blend',members=[anchor,candidate])])
        candidates.extend(current)
    value=dict(created_at=now(),candidates=candidates,references=references,
        rationale='Retain both frozen operational and ranking choices; primary strong anchor comes from V11 pre-confirmation selection ranking, never the V11 confirmation scores.',
        supervised_confirmation_used=False)
    path=ARTIFACTS/'comparison_spec.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Comparison specification changed')
        return old
    write(path,value);return value


def main():
    spec=specification()
    if (ARTIFACTS/'completed_comparison.json').exists():verify_freeze();return
    frozen_path=ARTIFACTS/'freeze.json'
    if not frozen_path.exists():
        selection={task:[] for task in ('link','service')};status('navigation_candidate_selection')
        for candidate in spec['candidates']:selection[candidate['task']].append(evaluate(candidate,'selection'))
        #Same pure selection rule; the imported function only reads summaries
        #and calls its own sweep, so use the equivalent own sweep below.
        from .calibration import sweep
        decisions={}
        for task,paths in selection.items():
            summaries=[read(p) for p in paths];sweeps=[sweep(p) for p in paths]
            full=[i for i,s in enumerate(sweeps) if s['full']];partial=[i for i,s in enumerate(sweeps) if s['partial']]
            if full:index=max(full,key=lambda i:(sweeps[i]['full']['event_recall_3s'],summaries[i]['selection_score']));rule='full_goals_then_recall'
            elif partial:index=max(partial,key=lambda i:(sweeps[i]['partial']['event_recall_3s'],summaries[i]['selection_score']));rule='precision90_FA1_partial_then_recall'
            else:index=max(range(len(paths)),key=lambda i:summaries[i]['selection_score']);rule='mean_AP_fallback'
            decisions[task]=dict(candidate=summaries[index]['candidate'],rule=rule,ranking_champion=max(summaries,key=lambda s:s['selection_score'])['model_tag'],confirmation_used=False,created_at=now())
        write(ARTIFACTS/'selection_decisions.json',decisions);status('navigation_probability_calibration');items=[]
        for task,paths in selection.items():
            for selected in paths:
                s=read(selected);candidate=s['candidate'];cal=evaluate(candidate,'calibration');policy=fit(cal);calibrated=apply(cal,policy)
                items.append(dict(candidate=candidate,pin=pin(candidate['model']) if candidate['family']=='new' else None,
                    selection_summary=str(selected),selection_summary_sha256=sha(selected),calibration_summary=str(cal),calibration_summary_sha256=sha(cal),
                    policy=str(policy),policy_sha256=sha(policy),calibrated_calibration=str(calibrated)))
        sources=['__init__.py','contracts.py','preview.py','wire.py','collect.py','data.py','episodes.py','train.py','inference.py','evaluate.py','calibration.py','compare.py','confirmation.py','runtime.py']
        write(frozen_path,dict(created_at=now(),sources={name:sha(ROOT/name) for name in sources},candidates=items,decisions=decisions,references=spec['references'],
            data_manifest_sha256=sha(ARTIFACTS/'data_manifest.json'),training_plan_sha256=sha(ARTIFACTS/'training_plan.json'),comparison_spec_sha256=sha(ARTIFACTS/'comparison_spec.json'),
            original_final_test_opened=False,production_ready=False,hardware_validated=False,model_reselection_forbidden=True))
    frozen=verify_freeze()
    from .confirmation import main as collect
    collect();status('navigation_frozen_fresh_confirmation');outcomes=[]
    for item in frozen['candidates']:
        raw=evaluate(item['candidate'],'confirmation');cal=apply(raw,item['policy']);outcomes.append(dict(candidate=item['candidate'],raw_summary=str(raw),calibrated_summary=str(cal)))
    verify_freeze();write(ARTIFACTS/'completed_comparison.json',dict(created_at=now(),decisions=frozen['decisions'],references=frozen['references'],outcomes=outcomes,
        freeze_sha256=sha(frozen_path),original_final_test_opened=False,production_ready=False))


if __name__=='__main__':main()

