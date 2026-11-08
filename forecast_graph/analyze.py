"""Paired, within-task confirmation comparisons and explicit ablation blocks."""
import math
from . import ARTIFACTS
from .contracts import read,write,sha,now
from forecast_rebuild.analyze_results import paired_uncertainty,macro_group


def finite(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
    if isinstance(x,list):return [finite(v) for v in x]
    return x


def main():
    done=read(ARTIFACTS/'completed_comparison.json');tasks={}
    for task in ('link','service'):
        models={i['candidate']['tag']:dict(raw=read(i['raw_summary']),calibrated=read(i['calibrated_summary'])) for i in done['outcomes'] if i['candidate']['task']==task}
        tag=done['decisions'][task]['candidate']['tag'];reference='previous_selected';old=models[reference];selected=models[tag]
        uncertainty=None if tag==reference else paired_uncertainty(old['raw'],selected['raw'],old['calibrated'],selected['calibrated'])
        contrasts=[]
        for a,b,meaning in [('v11_graph_local','v11_graph_shared','Received plan and neighbor-report information block; same graph architecture and fit budget'),
                ('v11_graph_shared','v11_hybrid_shared','Physics input plus learned residual/auxiliary objective block'),
                ('v11_tree_existing_inputs','v11_tree_shared','Added information alone; exact same queries, seeds, tree settings and old input columns')]:
            aa=models[a];bb=models[b]
            contrasts.append(dict(reference=a,candidate=b,meaning=meaning,
                pooled_ap5_difference=bb['raw']['next5_on_common_currently_up_links']['ap']-aa['raw']['next5_on_common_currently_up_links']['ap'],
                pooled_ap37_difference=bb['raw']['any3to7_on_common_support']['ap']-aa['raw']['any3to7_on_common_support']['ap'],
                uncertainty=paired_uncertainty(aa['raw'],bb['raw'],aa['calibrated'],bb['calibrated'])))
        tasks[task]=dict(selected_tag=tag,reference_tag=reference,models=models,decision=done['decisions'][task],
            selected_vs_reference_uncertainty=uncertainty,contrasts=contrasts,
            condition_groups={kind:{t:macro_group([c for c in v['raw']['cases'] if c['kind']==kind]) for t,v in models.items()}
                for kind in ('clear','continuous','intermittent')})
    write(ARTIFACTS/'final_results.json',finite(dict(created_at=now(),tasks=tasks,known_maps=14,new_confirmation_flights=42,
        comparison_sha256=sha(ARTIFACTS/'completed_comparison.json'),primary_observer='First relay for both separate tasks',
        original_final_test_opened=False,production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,
        warning='Different physical/service targets and old all-relay aggregates must not be directly subtracted.')))


if __name__=='__main__':main()

