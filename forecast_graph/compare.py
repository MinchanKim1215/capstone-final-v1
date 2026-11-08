"""Select/calibrate every candidate before immutable fresh-flight confirmation."""
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,status,V10
from .inference import pin
from .evaluate import evaluate,verify_freeze
from .calibration import sweep,fit,apply


def candidates():
    models=read(ARTIFACTS/'trained_models.json')['candidates']+read(ARTIFACTS/'tree_status.json')['models']
    result=[]
    for task in ('link','service'):
        old=dict(task=task,tag='previous_selected',family='previous')
        current=[dict(task=task,tag=read(m)['tag'],family='new',model=m) for m in models]
        hybrid=next(c for c in current if c['tag']=='v11_hybrid_shared')
        result.extend([old]+current+[dict(task=task,tag='v11_hybrid_blend50',family='blend',members=[old,hybrid])])
    return result


def choose(paths):
    summaries=[read(p) for p in paths];sw=[sweep(p) for p in paths]
    full=[i for i,s in enumerate(sw) if s['full']];partial=[i for i,s in enumerate(sw) if s['partial']]
    if full:index=max(full,key=lambda i:(sw[i]['full']['event_recall_3s'],summaries[i]['selection_score']));rule='full_goals_then_recall'
    elif partial:index=max(partial,key=lambda i:(sw[i]['partial']['event_recall_3s'],summaries[i]['selection_score']));rule='precision90_FA1_partial_then_recall'
    else:index=max(range(len(paths)),key=lambda i:summaries[i]['selection_score']);rule='mean_AP_fallback'
    return dict(candidate=summaries[index]['candidate'],rule=rule,ranking_champion=max(summaries,key=lambda s:s['selection_score'])['model_tag'],
        confirmation_used=False,created_at=now())


def main():
    verify()
    if (ARTIFACTS/'completed_comparison.json').exists():verify_freeze();return
    path=ARTIFACTS/'freeze.json'
    if not path.exists():
        all_candidates=candidates();selection={task:[] for task in ('link','service')}
        comparison_plan=dict(candidates=all_candidates,blend_weight=.5,primary_physical_observer='First relay, same flights for all models',
            model_selection='Selection-only alarm goals, then AP fallback; confirmation cannot choose models',created_at=now())
        if not (ARTIFACTS/'comparison_plan.json').exists():write(ARTIFACTS/'comparison_plan.json',comparison_plan)
        status('selection_comparison')
        for candidate in all_candidates:selection[candidate['task']].append(evaluate(candidate,'selection'))
        decisions={task:choose(paths) for task,paths in selection.items()};write(ARTIFACTS/'selection_decisions.json',decisions)
        status('probability_and_alarm_calibration');items=[]
        for task,paths in selection.items():
            for selected in paths:
                summary=read(selected);candidate=summary['candidate'];cal=evaluate(candidate,'calibration');policy=fit(cal);calibrated=apply(cal,policy)
                items.append(dict(candidate=candidate,pin=pin(candidate['model']) if candidate['family']=='new' else None,
                    selection_summary=str(selected),selection_summary_sha256=sha(selected),calibration_summary=str(cal),calibration_summary_sha256=sha(cal),
                    policy=str(policy),policy_sha256=sha(policy),calibrated_calibration=str(calibrated)))
        sources=['__init__.py','contracts.py','codec.py','trace.py','features.py','data.py','model.py','train.py','tree.py','inference.py','episodes.py','evaluate.py','calibration.py','compare.py','confirmation.py','runtime.py']
        write(path,dict(created_at=now(),sources={name:sha(ROOT/name) for name in sources},candidates=items,decisions=decisions,
            data_manifest_sha256=sha(ARTIFACTS/'data_manifest.json'),data_plan_sha256=sha(ARTIFACTS/'data_plan.json'),
            training_plan_sha256=sha(ARTIFACTS/'training_plan.json'),comparison_plan_sha256=sha(ARTIFACTS/'comparison_plan.json'),
            tree_control_plan_sha256=sha(ARTIFACTS/'tree_control_plan.json'),
            original_final_test_opened=False,model_reselection_forbidden=True,production_ready=False,hardware_validated=False))
    frozen=verify_freeze()
    from .confirmation import main as collect
    collect();status('frozen_fresh_confirmation');outcomes=[]
    for item in frozen['candidates']:
        raw=evaluate(item['candidate'],'confirmation');cal=apply(raw,item['policy'])
        outcomes.append(dict(candidate=item['candidate'],raw_summary=str(raw),calibrated_summary=str(cal)))
    verify_freeze();write(ARTIFACTS/'completed_comparison.json',dict(created_at=now(),decisions=frozen['decisions'],outcomes=outcomes,
        freeze_sha256=sha(path),original_final_test_opened=False,production_ready=False,hardware_validated=False))
    status('comparison_complete_pending_audit')


if __name__=='__main__':main()
