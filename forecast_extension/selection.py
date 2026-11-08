"""Selection-only ranking and precision-constrained event operating points."""
from pathlib import Path
import numpy as np
import torch
from . import ARTIFACTS
from .contracts import read,write,sha,now,OLD_NEURAL,OLD_TREE
from .evaluate import evaluate
from .metrics import alert_counts,combine_alerts
from forecast_rebuild.calibration import records,alarm_support,meets


def prepare_plan():
    value=dict(created_at=now(),base_candidates=['baseline','v8_tree','v8_neural','same_data_20000','new_data_20000','same_data_final20000','new_data_final20000','new_data_tree','new_data_tree_no_energy'],
        fixed_endpoint_control='Include both regimes at exactly step20000 as well as best checkpoints selected on the old probe; this separates endpoint exposure from early checkpoint choice.',
        blends=dict(tree_weights=[.25,.5,.75],members='Highest mean AP tree and neural on old+fresh selection only'),
        primary_selection='If any candidate meets all selection goals, maximize event recall. Otherwise maximize recall among precision>=.90, false_or_late/hour<=1, minimum50 first-alert candidates. If none exist, maximize mean common AP5/AP37.',
        ranking_champion='Also record highest AP mean independently of the operational choice',
        fresh_confirmation_models='All fixed base and three blend candidates, once after a single freeze; no reselection afterwards',
        scope='Fresh flight realizations on known maps; original32 final-test maps remain sealed',
        caution='One continuation per data regime; batch sampling differs with the expanded data. Matched update budget is not proof of a pure causal data effect.')
    path=ARTIFACTS/'comparison_plan.json'
    if path.exists():
        old=read(path)
        if 'fixed_endpoint_control' not in old:
            if (ARTIFACTS/'freeze.json').exists() or (ARTIFACTS/'selection_decision.json').exists() or (ARTIFACTS/'confirmation_access.json').exists():
                raise ValueError('Do not revise a comparison after selection/confirmation')
            write(ARTIFACTS/'comparison_plan_initial.json',old)
            value['revision_reason']='Pre-selection addition of matched step20000 endpoints, before fresh calibration/confirmation evaluation.'
            write(path,value);return path
        for key in value:
            if key!='created_at' and old[key]!=value[key]:raise ValueError('Comparison plan changed')
        return path
    write(path,value);return path


def candidate_models():
    aliases=[]
    for source_tag,tag in [('same_data_20000','same_data_final20000'),('new_data_20000','new_data_final20000')]:
        parent=ARTIFACTS/'models'/source_tag/'model.json';metadata=read(parent);checkpoint=parent.parent/'s0020000.pt'
        if not checkpoint.exists():raise ValueError('The matched final checkpoint is missing')
        path=ARTIFACTS/'models'/tag/'model.json'
        if not path.exists():
            endpoint_metrics=torch.load(checkpoint,map_location='cpu',weights_only=False)['selection']
            value=dict(metadata,tag=tag,checkpoint=str(checkpoint),sha256=sha(checkpoint),selected_step=20000,
                       selection=endpoint_metrics,selection_score=endpoint_metrics['score'],
                       endpoint_control=True,parent_manifest=str(parent),parent_manifest_sha256=sha(parent),created_at=now())
            write(path,value)
        elif read(path)['sha256']!=sha(checkpoint):raise ValueError('An endpoint control changed')
        aliases.append(str(path))
    base=['baseline',str(OLD_TREE),str(OLD_NEURAL)]+[str(ARTIFACTS/'models'/tag/'model.json') for tag in
        ('same_data_20000','new_data_20000','new_data_tree','new_data_tree_no_energy')]
    return base+aliases


def blend(tree,neural,weight):
    folder=ARTIFACTS/'models'/('v9_blend_'+str(int(weight*100)));path=folder/'model.json'
    members=[dict(path=tree['model_manifest'],model_sha256=sha(tree['model_manifest']),weight=weight),
             dict(path=neural['model_manifest'],model_sha256=sha(neural['model_manifest']),weight=1-weight)]
    value=dict(family='ensemble',tag=folder.name,members=members,use_energy=bool(read(members[0]['path'])['use_energy'] or read(members[1]['path'])['use_energy']),
               use_plan=True,created_at=now(),calibration_used=False,confirmation_used=False)
    if path.exists():
        old=read(path)
        if old['members']!=members:raise ValueError('A blend was overwritten')
    else:write(path,value)
    return str(path)


def sweep(summary_path):
    summary=read(summary_path)
    if summary['scope']!='selection':raise ValueError('Do not optimize operating points on calibration or confirmation')
    path=Path(summary_path).parent/'selection_alarm_sweep.json'
    if path.exists():
        old=read(path)
        if old['source_sha256']==sha(summary_path):return old
        raise ValueError('Selection evidence changed')
    data=records(summary);energy=summary['model_manifest']!='baseline' and read(summary['model_manifest']).get('use_energy',False)
    pooled=np.concatenate([p[...,5][ep.valid&alarm_support(ep,energy)] for ep,p in data])
    thresholds=np.unique(np.r_[np.linspace(.01,.99,50),np.quantile(pooled,np.linspace(.5,.999,35))]);trials=[]
    for threshold in thresholds:
        counts=combine_alerts([alert_counts(ep.links,ep.ends,p[...,5],alarm_support(ep,energy),ep.record['n_search'],float(threshold)) for ep,p in data])
        trials.append(dict(threshold=float(threshold),**counts))
    full=[r for r in trials if meets(r)]
    partial=[r for r in trials if r['first_alerts']>=50 and r['precision'] is not None and r['precision']>=.9 and r['false_or_late_per_drone_hour']<=1]
    key=lambda r:(r['event_recall_3s'],r['precision'],-r['false_or_late_per_drone_hour'])
    result=dict(model_tag=summary['model_tag'],source_summary=str(summary_path),source_sha256=sha(summary_path),created_at=now(),
                full_goal=max(full,key=key) if full else None,partial_precision90=max(partial,key=key) if partial else None,trials=trials,
                calibration_used=False,confirmation_used=False)
    write(path,result);return result


def select():
    prepare_plan();paths=[evaluate(model) for model in candidate_models()];summaries=[read(path) for path in paths]
    trees=[s for s in summaries if s['model_manifest']!='baseline' and read(s['model_manifest'])['family']=='lightgbm']
    neural=[s for s in summaries if s['model_manifest']!='baseline' and read(s['model_manifest'])['family']=='residual_transformer']
    tree=max(trees,key=lambda s:s['selection_score']);network=max(neural,key=lambda s:s['selection_score'])
    for weight in (.25,.5,.75):
        path=evaluate(blend(tree,network,weight));paths.append(path);summaries.append(read(path))
    sweeps=[sweep(path) for path in paths]
    full=[i for i,s in enumerate(sweeps) if s['full_goal']];partial=[i for i,s in enumerate(sweeps) if s['partial_precision90']]
    if full:index=max(full,key=lambda i:(sweeps[i]['full_goal']['event_recall_3s'],summaries[i]['selection_score']));rule='all_goals_then_event_recall'
    elif partial:index=max(partial,key=lambda i:(sweeps[i]['partial_precision90']['event_recall_3s'],summaries[i]['selection_score']));rule='precision90_FA1_partial_then_event_recall'
    else:index=max(range(len(summaries)),key=lambda i:summaries[i]['selection_score']);rule='AP_mean_fallback'
    winner=summaries[index];rank=max(summaries,key=lambda s:s['selection_score'])
    result=dict(selected_model=winner['model_manifest'],selected_tag=winner['model_tag'],selection_summary=str(paths[index]),rule=rule,
        ranking_champion=rank['model_tag'],rank_champion_score=rank['selection_score'],
        summaries=[str(p) for p in paths],sweeps=sweeps,created_at=now(),calibration_outcomes_used=False,confirmation_outcomes_used=False)
    write(ARTIFACTS/'selection_decision.json',result);return result


if __name__=='__main__':print(select())

