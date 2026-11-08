"""Matched confirmation: factorial gains, event evidence and geography uncertainty."""
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,now
from .evaluate import case_prediction
from .causes import event_annotations
from forecast_rebuild.analyze_results import paired_uncertainty
from forecast_graph.analyze import finite
from forecast_extension.metrics import combine_alerts
from forecast_recall.episodes import cohort
from forecast_recall.diagnose import starts
from forecast_rebuild.calibration import transform

def difference(models,a,b,meaning):
    old=models[a];new=models[b];x=old['calibrated']['alarms'];y=new['calibrated']['alarms']
    return dict(reference=a,candidate=b,meaning=meaning,event_recall_difference=y['event_recall_3s']-x['event_recall_3s'],precision_difference=None if x['precision'] is None or y['precision'] is None else y['precision']-x['precision'],
        FA_per_hour_difference=y['false_or_late_per_drone_hour']-x['false_or_late_per_drone_hour'],ap37_difference=new['raw']['any3to7_on_common_support']['ap']-old['raw']['any3to7_on_common_support']['ap'],
        uncertainty=None if a==b else paired_uncertainty(old['raw'],new['raw'],old['calibrated'],new['calibrated']))

def cause_groups(models):
    items=read(ARTIFACTS/'confirmation_data_manifest.json')['records'];events=[];totals={};hits={tag:{} for tag in models};reasons={tag:{} for tag in models}
    for item in items:
        replay=read(item['replay_path'][:-4]+'.json');annotations=event_annotations(replay);ep=cohort(item);alert_times={};scores={};thresholds={};times=ep.ends*.5
        for tag,model in models.items():
            policy=read(model['calibrated']['policy_path']);p,_=case_prediction(model['raw']['candidate'],item,'confirmation');score=transform(p,policy['parameters'])[...,5]
            alert_times[tag]=np.array([t for t,a in starts(ep,score,ep.legacy_support,policy['threshold']) if a[0,0]])
            scores[tag]=score[:,0,0];thresholds[tag]=policy['threshold']
        for when,ev in annotations:
            flags=dict(all_events=True,**ev['flags']);outcomes={};diagnosis={}
            for tag,aa in alert_times.items():
                hit=bool(((aa>when-7)&(aa<=when-3)).any());outcomes[tag]=hit
                valid=(times>when-7)&(times<=when-3)&(times>=21)&ep.legacy_support[:,0,0]&np.isfinite(scores[tag]);high=bool((scores[tag][valid]>=thresholds[tag]).any())
                reason='timely_first_alert' if hit else 'no_qualified_input' if not valid.any() else 'risk_below_threshold' if not high else 'already_active_without_new_alert'
                reasons[tag][reason]=reasons[tag].get(reason,0)+1;diagnosis[tag]=dict(reason=reason,threshold=thresholds[tag],maximum_qualified_pre_event_score=float(scores[tag][valid].max()) if valid.any() else None)
            for flag,present in flags.items():
                if not present:continue
                totals[flag]=totals.get(flag,0)+1
                for tag,hit in outcomes.items():hits[tag][flag]=hits[tag].get(flag,0)+int(hit)
            events.append(dict(key=item['key'],event_time=when,flags=ev['flags'],timely_first_alert=outcomes,forecast_diagnosis=diagnosis))
    for tag,m in models.items():
        a=m['calibrated']['alarms']
        if totals['all_events']!=a['eligible_directed_link_events'] or hits[tag]['all_events']!=a['event_hits_3s']:raise ValueError('Cause grouping changed strict event denominator or hit counts')
    return dict(groups_overlap=True,instant_component_removal_not_full_causal_proof=True,all_events_retained=True,counts=totals,hits=hits,forecast_outcome_counts=reasons,events=events,rulebook_sha256=sha(ARTIFACTS/'cause_rulebook.json'))

def main():
    c=read(ARTIFACTS/'completed_comparison.json');models={i['candidate']['tag']:dict(raw=read(i['raw_summary']),calibrated=read(i['calibrated_summary'])) for i in c['outcomes']};selected=c['decision']['candidate']['tag'];conditions={}
    for kind in ('clear','intermittent','continuous'):
        conditions[kind]={}
        for tag,m in models.items():
            keys={i['key'] for i in m['raw']['cases'] if i['kind']==kind};conditions[kind][tag]=combine_alerts([i for i in m['calibrated']['case_alarms'] if i['key'] in keys])
    contrasts=[difference(models,a,b,why) for a,b,why in (
        ('v16_tree_base','v16_tree_network','Same tree budget; added actually observed packet state'),
        ('v16_tree_base','v16_tree_physics','Same tree budget; added public-map/path uncertainty features'),
        ('v16_tree_network','v16_tree_both','Physics increment with new measured state held present'),
        ('v16_tree_physics','v16_tree_both','Measured-state increment with physics held present'),
        ('v16_gru_base','v16_gru_network','Same frozenV13/base/RF memory/loss/budget; added measured-state GRU'))]
    old=models['frozen_v13']['calibrated']['alarms'];new=models[selected]['calibrated']['alarms'];costlimit=max(4.,old['false_or_late_per_drone_hour']);point=bool(new['event_recall_3s']>old['event_recall_3s'] and new['precision'] is not None and old['precision'] is not None and new['precision']>=old['precision'] and new['false_or_late_per_drone_hour']<=costlimit)
    primary=difference(models,'frozen_v13',selected,'Preserved user-selectedV13threshold. Same NEW42flights and complete event denominator.');ci=primary['uncertainty']['metrics']['calibrated_recall_difference']['percentile95'] if primary['uncertainty'] else [0,0]
    fits=[]
    for path in read(ARTIFACTS/'trained_models.json')['models']:
        ensemble=read(path)
        for member in ensemble['members']:
            m=read(member['path']);fits.append(dict(variant=m['variant'],seed=m['seed'],family=m['family'],updates=m.get('updates',m.get('iterations')),selected_step=m.get('selected_step'),selected_head_iterations={k:v['selected_iteration'] for k,v in m.get('heads',{}).items()},model_path=member['path'],sha256=sha(member['path'])))
    value=dict(created_at=now(),selected_tag=selected,models=models,decision=c['decision'],training_receipt=fits,known_maps=14,new_confirmation_flights_after_own_freeze=42,source_flights_including_development=377,
        selected_vs_frozen=primary,selected_vs_retuned=difference(models,'retuned_v13',selected,'Same FA4 calibration procedure; separates model from threshold gain'),contrasts=contrasts,condition_alarms=conditions,
        cause_analysis=cause_groups(models),adoption=dict(rule_sha256=sha(ARTIFACTS/'adoption_rule.json'),point_metrics_pass=point,recall_CI_excludes_zero=bool(ci[0]>0),observed_FA_cost_limit=costlimit,
            current_profile_changed=False,automatic_replacement=False,interpretation='Research candidate only; point gains require uncertainty and hardware review. Current profile preserved.'),
        primary_calibration_FA_budget=4.,secondary_budgets_prelocked=[3,5],comparison_sha256=sha(ARTIFACTS/'completed_comparison.json'),historical_v13=dict(recall=55/174,precision=57/147,FA_per_hour=4.235958581738312,note='Different old42flights; not subtracted here'),
        original_final_test_opened=False,production_ready=False,hardware_validated=False,automatic_PPO_replacement=False)
    write(ARTIFACTS/'final_results.json',finite(value))

if __name__=='__main__':main()
