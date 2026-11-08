"""Paired confirmation; isolate navigation inputs from continued fitting."""
from . import ARTIFACTS
from .contracts import read,write,sha,now
from forecast_graph.analyze import finite
from forecast_rebuild.analyze_results import paired_uncertainty,macro_group


def contrast(models,a,b,meaning):
    aa=models[a];bb=models[b]
    return dict(reference=a,candidate=b,meaning=meaning,
        pooled_ap5_difference=bb['raw']['next5_on_common_currently_up_links']['ap']-aa['raw']['next5_on_common_currently_up_links']['ap'],
        pooled_ap37_difference=bb['raw']['any3to7_on_common_support']['ap']-aa['raw']['any3to7_on_common_support']['ap'],
        uncertainty=None if a==b else paired_uncertainty(aa['raw'],bb['raw'],aa['calibrated'],bb['calibrated']))


def main():
    done=read(ARTIFACTS/'completed_comparison.json');tasks={}
    for task in ('link','service'):
        models={i['candidate']['tag']:dict(raw=read(i['raw_summary']),calibrated=read(i['calibrated_summary'])) for i in done['outcomes'] if i['candidate']['task']==task}
        selected=done['decisions'][task]['candidate']['tag'];reference=done['references'][task];ranking=done['decisions'][task]['ranking_champion']
        tasks[task]=dict(selected_tag=selected,reference_tag=reference,models=models,decision=done['decisions'][task],
            selected_vs_reference=contrast(models,reference,selected,'Frozen alarm-oriented choice versus predeclared stronger anchor on exactly the same new42 flights'),
            ranking_champion=ranking,ranking_vs_reference=contrast(models,reference,ranking,'Ranking champion determined from selection before confirmation; never reselected on confirmation'),
            contrasts=[contrast(models,'v12_original','v12_mission','Same architecture, initial weights, query rows and extra6000 updates per seed; only announced plan view differs'),
                contrast(models,'v12_original_blend50','v12_mission_blend50','Same frozen parent blend; isolates announced plan view')],
            condition_groups={kind:{tag:macro_group([c for c in value['raw']['cases'] if c['kind']==kind]) for tag,value in models.items()}
                for kind in ('clear','continuous','intermittent')})
    write(ARTIFACTS/'final_results.json',finite(dict(created_at=now(),tasks=tasks,known_maps=14,new_confirmation_flights=42,
        comparison_sha256=sha(ARTIFACTS/'completed_comparison.json'),gate_sha256=sha(ARTIFACTS/'gate_result.json'),
        production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,original_final_test_opened=False,
        warning='AP is ranking quality, not accuracy. Physical and service target denominators differ. Navigation RMSE reduction is not prediction gain.')))


if __name__=='__main__':main()
