"""Matched fresh-flight outcomes; all candidates and geography uncertainty."""
from . import ARTIFACTS
from .contracts import read,write,sha,now
from forecast_rebuild.analyze_results import paired_uncertainty
from forecast_graph.analyze import finite
from forecast_extension.metrics import combine_alerts

def difference(models,a,b,meaning):
    old=models[a];new=models[b];x=old['calibrated']['alarms'];y=new['calibrated']['alarms']
    return dict(reference=a,candidate=b,meaning=meaning,event_recall_difference=y['event_recall_3s']-x['event_recall_3s'],
        precision_difference=None if x['precision'] is None or y['precision'] is None else y['precision']-x['precision'],FA_per_hour_difference=y['false_or_late_per_drone_hour']-x['false_or_late_per_drone_hour'],
        ap37_difference=new['raw']['any3to7_on_common_support']['ap']-old['raw']['any3to7_on_common_support']['ap'],uncertainty=None if a==b else paired_uncertainty(old['raw'],new['raw'],old['calibrated'],new['calibrated']))

def main():
    c=read(ARTIFACTS/'completed_comparison.json');models={i['candidate']['tag']:dict(raw=read(i['raw_summary']),calibrated=read(i['calibrated_summary'])) for i in c['outcomes']};selected=c['decision']['candidate']['tag'];groups={}
    for kind in ('clear','intermittent','continuous'):
        groups[kind]={}
        for tag,m in models.items():
            keys={x['key'] for x in m['raw']['cases'] if x['kind']==kind};groups[kind][tag]=combine_alerts([x for x in m['calibrated']['case_alarms'] if x['key'] in keys])
    value=dict(created_at=now(),selected_tag=selected,models=models,decision=c['decision'],known_maps=14,shared_confirmation_flights=42,new_V15_flights_after_freeze=0,
        selected_vs_frozen=difference(models,'frozen_v13',selected,'Existing user-selected V13FA5 threshold preserved, compared on same NEW42flights'),
        selected_vs_retuned=difference(models,'retuned_v13',selected,'Same calibrationFA4 policy; separates model gain from changed threshold'),
        contrasts=[difference(models,'v15_peer_bce','v15_peer_event','Same per-peer raw histories/weights/batches/budget; event and precision losses added'),
                   difference(models,'v15_global_event','v15_peer_event','Same frozenV13base/eventobjective/budget; pooled histories versus per-peer observed histories')],
        condition_alarms=groups,calibration_FA_budget=4.,historical_v13=dict(recall=55/174,precision=57/147,FA_per_hour=4.235958581738312,note='Different previous42flights; do not directly subtract'),
        comparison_sha256=sha(ARTIFACTS/'completed_comparison.json'),original_final_test_opened=False,production_ready=False,hardware_validated=False,automatic_PPO_replacement=False)
    write(ARTIFACTS/'final_results.json',finite(value))

if __name__=='__main__':main()
