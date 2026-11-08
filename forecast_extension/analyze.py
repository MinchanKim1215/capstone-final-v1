"""Matched comparisons and geography-cluster uncertainty on fresh flights."""
from . import ARTIFACTS
from .contracts import read,write,sha,now,OLD_TREE
from forecast_rebuild.analyze_results import paired_uncertainty,macro_group
from forecast_rebuild.dataset import Episode
from forecast_rebuild.diagnose import observability,combine as combine_observability
from .activity import diagnostic
import math


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    return value


def main():
    completed=read(ARTIFACTS/'completed_comparison.json');models={}
    for outcome in completed['outcomes']:
        models[outcome['tag']]=dict(raw=read(outcome['raw_summary']),calibrated=read(outcome['calibrated_summary']))
    old=models[read(OLD_TREE)['tag']];winner=models[completed['selected_tag']]
    uncertainty=paired_uncertainty(old['raw'],winner['raw'],old['calibrated'],winner['calibrated'])
    records={r['key']:r for r in read(winner['raw']['dataset_path'])['rows']}
    observability_rows=[observability(Episode(records[case['key']])) for case in winner['raw']['cases']]
    groups={}
    for condition in ('clear','intermittent','continuous'):
        groups[condition]={tag:macro_group([case for case in model['raw']['cases'] if case['kind']==condition]) for tag,model in models.items()}
    value=dict(created_at=now(),selected_tag=completed['selected_tag'],selected_model=completed['selected_model'],
        models=models,condition_groups=groups,paired_spatial_uncertainty_against_v8_tree=uncertainty,
        event_observability=combine_observability(observability_rows),stage1=read(ARTIFACTS/'stage1_comparison.json'),
        recent_data_diagnostic=diagnostic(models),
        selection=read(ARTIFACTS/'selection_decision.json'),scope='New flight realizations on known 14 confirmation maps, not new geographies',
        production_ready=False,hardware_validated=False,original_final_test_opened=False)
    write(ARTIFACTS/'final_results.json',finite_json(value));print(ARTIFACTS/'final_results.json')


if __name__=='__main__':main()

