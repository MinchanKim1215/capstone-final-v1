"""Primary event recall gains; separate threshold tuning and new learning."""
from . import ARTIFACTS,PARENT
from .contracts import read,write,sha,now
from forecast_graph.analyze import finite
from forecast_rebuild.analyze_results import paired_uncertainty,macro_group
from forecast_extension.metrics import combine_alerts


def difference(models,a,b,meaning):
    old=models[a];new=models[b];oa=old['calibrated']['alarms'];na=new['calibrated']['alarms']
    return dict(reference=a,candidate=b,meaning=meaning,
        event_recall_difference=na['event_recall_3s']-oa['event_recall_3s'],precision_difference=None if na['precision'] is None or oa['precision'] is None else na['precision']-oa['precision'],
        FA_per_hour_difference=na['false_or_late_per_drone_hour']-oa['false_or_late_per_drone_hour'],
        ap5_difference=new['raw']['next5_on_common_currently_up_links']['ap']-old['raw']['next5_on_common_currently_up_links']['ap'],
        ap37_difference=new['raw']['any3to7_on_common_support']['ap']-old['raw']['any3to7_on_common_support']['ap'],
        uncertainty=None if a==b else paired_uncertainty(old['raw'],new['raw'],old['calibrated'],new['calibrated']))


def main():
    completed=read(ARTIFACTS/'completed_comparison.json');models={i['candidate']['tag']:dict(raw=read(i['raw_summary']),calibrated=read(i['calibrated_summary'])) for i in completed['outcomes']}
    selected=completed['decision']['candidate']['tag']
    alarm_groups={}
    for kind in ('clear','continuous','intermittent'):
        alarm_groups[kind]={}
        for tag,m in models.items():
            keys={c['key'] for c in m['raw']['cases'] if c['kind']==kind}
            alarm_groups[kind][tag]=combine_alerts([c for c in m['calibrated']['case_alarms'] if c['key'] in keys])
    result=dict(created_at=now(),task='First-relay service event recall',selected_tag=selected,reference_tag='frozen_parent',threshold_reference_tag='threshold_only_parent',models=models,
        decision=completed['decision'],selected_vs_frozen=difference(models,'frozen_parent',selected,'Same NEW42flights; old frozen risk and policy versus preselected new candidate'),
        selected_vs_threshold_only=difference(models,'threshold_only_parent',selected,'Same threshold fitting procedure and original metric; separates model gain from threshold changes'),
        contrasts=[difference(models,'v13_bce','v13_event','Same context, natural rows, initialization and4000updates; added equal-event and precision-surrogate losses'),
            difference(models,'v13_event','v13_memory_event','Same event objective and budget; added20s causal local context')],
        condition_groups={kind:{tag:macro_group([c for c in m['raw']['cases'] if c['kind']==kind]) for tag,m in models.items()} for kind in ('clear','continuous','intermittent')},
        condition_alarm_groups=alarm_groups,
        historical_V12=dict(event_hits=25,events=160,recall=.15625,precision=25/51,warning='Different V12 flights; never subtract directly from new-study result'),
        known_maps=14,new_confirmation_flights=42,original_final_test_opened=False,production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,
        comparison_sha256=sha(ARTIFACTS/'completed_comparison.json'))
    write(ARTIFACTS/'final_results.json',finite(result))


if __name__=='__main__':main()
