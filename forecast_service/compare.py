"""Selection -> calibration -> immutable freeze -> new42-flight confirmation."""
from pathlib import Path
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now,status,verify,V9
from .inference import pin
from .evaluate import evaluate,verify_freeze
from .calibration import sweep,fit,apply


def candidates():
    trained=read(ARTIFACTS/'trained_models.json')['candidates'];old=read(V9/'freeze.json')
    from forecast_extension.contracts import OLD_TREE
    proxy=ARTIFACTS/'models/v10_service_v9proxy/model.json'
    if not proxy.exists():write(proxy,dict(family='service_proxy',task='service',tag='v10_service_v9proxy',link_model=old['selected_model'],use_energy=False,use_plan=False,created_at=now()))
    return [dict(task='link',model=str(OLD_TREE)),dict(task='link',model=old['selected_model']),dict(task='service',model=str(proxy))]+trained


def choose(paths):
    summaries=[read(path) for path in paths];sweeps=[sweep(path) for path in paths]
    full=[i for i,s in enumerate(sweeps) if s['full']];partial=[i for i,s in enumerate(sweeps) if s['partial']]
    if full:index=max(full,key=lambda i:(sweeps[i]['full']['event_recall_3s'],summaries[i]['selection_score']));rule='full_goal_then_recall'
    elif partial:index=max(partial,key=lambda i:(sweeps[i]['partial']['event_recall_3s'],summaries[i]['selection_score']));rule='precision90_FA1_partial_then_recall'
    else:index=max(range(len(paths)),key=lambda i:summaries[i]['selection_score']);rule='mean_AP_fallback'
    return dict(model=summaries[index]['model_manifest'],tag=summaries[index]['model_tag'],rule=rule,
        ranking_champion=max(summaries,key=lambda s:s['selection_score'])['model_tag'],created_at=now(),confirmation_used=False)


def main():
    verify()
    if (ARTIFACTS/'completed_comparison.json').exists():verify_freeze();return
    freeze_path=ARTIFACTS/'freeze.json'
    if not freeze_path.exists():
        status('stage3_selection_comparison');models=candidates();selection={task:[] for task in ('link','service')}
        for model in models:selection[model['task']].append(evaluate(model['task'],model['model']))
        decisions={task:choose(paths) for task,paths in selection.items()};write(ARTIFACTS/'selection_decisions.json',decisions)
        status('stage3_calibration');items=[]
        for task,paths in selection.items():
            for path in paths:
                summary=read(path);model=summary['model_manifest'];cal=evaluate(task,model,'calibration');policy=fit(cal);calibrated=apply(cal,policy)
                items.append(dict(task=task,tag=summary['model_tag'],pin=pin(model),selection_summary=str(path),selection_summary_sha256=sha(path),
                    calibration_summary=str(cal),calibration_summary_sha256=sha(cal),policy=str(policy),policy_sha256=sha(policy),calibrated_calibration=str(calibrated)))
        sources=['__init__.py','contracts.py','plan.py','replay.py','features.py','data.py','train.py','inference.py','feature_cache.py','evaluate.py','calibration.py','confirmation.py','compare.py','runtime.py']
        write(freeze_path,dict(created_at=now(),models=items,decisions=decisions,sources={name:sha(ROOT/name) for name in sources},
            experiment_plan_sha256=sha(ARTIFACTS/'experiment_plan.json'),trained_catalog_sha256=sha(ARTIFACTS/'trained_models.json'),
            original_final_test_opened=False,production_ready=False,hardware_validated=False,model_reselection_forbidden=True))
    frozen=verify_freeze()
    from .confirmation import main as collect
    collect();status('stage4_frozen_confirmation_evaluation');outcomes=[]
    for item in frozen['models']:
        raw=evaluate(item['task'],item['pin']['path'],'confirmation');calibrated=apply(raw,item['policy'])
        outcomes.append(dict(task=item['task'],tag=item['tag'],raw_summary=str(raw),calibrated_summary=str(calibrated)))
    verify_freeze();write(ARTIFACTS/'completed_comparison.json',dict(created_at=now(),decisions=frozen['decisions'],outcomes=outcomes,
        freeze_sha256=sha(freeze_path),original_final_test_opened=False,production_ready=False,hardware_validated=False))
    status('comparison_complete_pending_audit')


if __name__=='__main__':main()
