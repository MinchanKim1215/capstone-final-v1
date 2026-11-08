"""Keep different truths separate; paired spatial uncertainty within a task."""
import math
from . import ARTIFACTS
from .contracts import read,write,now,V9
from forecast_rebuild.analyze_results import paired_uncertainty,macro_group


def finite(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list):return [finite(v) for v in value]
    return value


def main():
    done=read(ARTIFACTS/'completed_comparison.json');tasks={}
    for task in ('link','service'):
        models={o['tag']:dict(raw=read(o['raw_summary']),calibrated=read(o['calibrated_summary'])) for o in done['outcomes'] if o['task']==task}
        selected=done['decisions'][task]['tag'];reference=read(read(V9/'freeze.json')['selected_model'])['tag'] if task=='link' else 'v10_service_v9proxy'
        a=models[reference];b=models[selected]
        uncertainty=paired_uncertainty(a['raw'],b['raw'],a['calibrated'],b['calibrated'])
        groups={kind:{tag:macro_group([c for c in value['raw']['cases'] if c['kind']==kind]) for tag,value in models.items()} for kind in ('clear','continuous','intermittent')}
        ablations={}
        pairs=[('v10_link_uniform','v10_link_targeted')] if task=='link' else [('v10_service_uniform','v10_service_targeted'),('v10_service_targeted','v10_service_targeted_topology')]
        for left,right in pairs:
            a=models[left];b=models[right]
            ablations[left+'__'+right]=dict(reference=left,candidate=right,
                pooled_ap5_difference=b['raw']['next5_on_common_currently_up_links']['ap']-a['raw']['next5_on_common_currently_up_links']['ap'],
                pooled_ap37_difference=b['raw']['any3to7_on_common_support']['ap']-a['raw']['any3to7_on_common_support']['ap'],
                uncertainty=paired_uncertainty(a['raw'],b['raw'],a['calibrated'],b['calibrated']))
        tasks[task]=dict(selected_tag=selected,reference_tag=reference,models=models,groups=groups,
            selected_vs_reference_uncertainty=uncertainty,one_change_ablations=ablations,decision=done['decisions'][task])
    write(ARTIFACTS/'final_results.json',finite(dict(created_at=now(),tasks=tasks,known_maps=14,new_confirmation_flights=42,
        original_final_test_opened=False,production_ready=False,hardware_validated=False,
        warning='Physical-link and global packet-service targets have different events, denominators and units. No across-task percentage improvement is valid.')))
    print(ARTIFACTS/'final_results.json')


if __name__=='__main__':main()
