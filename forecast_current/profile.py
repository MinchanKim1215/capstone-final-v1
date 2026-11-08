"""Record the explicit user preference and bind the already verified option."""
from pathlib import Path
from . import ROOT,PROFILE
from forecast_recall import ARTIFACTS as SOURCE
from forecast_recall.contracts import read,write,sha,now


def configure_detection_first():
    options=read(SOURCE/'research_operating_options.json');plan=read(SOURCE/'secondary_operating_points_plan.json')
    results=read(SOURCE/'secondary_operating_points.json');tag=options['selected_tag'];point=plan['items'][tag]['points']['5']
    verification=read(SOURCE/'runtime_options_checks.json')
    if not verification['passed'] or verification['budget']!=5 or verification['threshold']!=point['threshold']:
        raise ValueError('The selected research operating point is not the verified option')
    selected=results['models'][tag]['5']['confirmation'];previous=results['models']['frozen_parent']['5']['confirmation']
    files=[SOURCE/'delivery_manifest.json',SOURCE/'freeze.json',SOURCE/'research_candidate.json',SOURCE/'research_operating_options.json',
        SOURCE/'secondary_operating_points_plan.json',SOURCE/'secondary_operating_points.json',SOURCE/'runtime_options_checks.json',SOURCE.parent/'runtime_options.py']
    value=dict(profile_id='detection_first_v13_20261103',selected_at=now(),priority='advance_event_recall',
        user_choice='탐지율을 우선하고 약 4회/h 오경보 허용',scope='Research inference for first-relay service prediction',
        source_study='forecast_recall_v13',model_tag=tag,calibration_FA_budget_per_hour=5,threshold=point['threshold'],
        approximate_requested_FA_per_hour=4.,strict_FA_cap=False,
        benchmark=dict(known_maps=14,flights=42,selected=selected,matched_previous=previous,
            note='Historical matched confirmation result, not a new evaluation or a guarantee of future alarm rates'),
        input_contract='Same past-only telemetry and actually received128B beacons;20s warmup,1Hz forecast,3<=lead<7s event evaluation',
        future_comparison_priority='Prioritize event recall; report actual precision and false alarms around the user-requested4/h rate',
        dependencies={str(p):sha(p) for p in files},production_ready=False,hardware_validated=False,trusted_alarm_enabled=False,automatic_PPO_replacement=False,
        decision_timing='User selected the already evaluated research option after reviewing V13 results. No model retraining or historical result changes.')
    if PROFILE.exists():
        old=read(PROFILE)
        if {k:v for k,v in old.items() if k!='selected_at'}!={k:v for k,v in value.items() if k!='selected_at'}:
            raise ValueError('Current profile differs; preserve its history before changing it')
        return old
    write(PROFILE,value);return value


def load_profile():
    p=read(PROFILE)
    for name,expected in p['dependencies'].items():
        if sha(name)!=expected:raise ValueError('The selected profile dependency changed: '+name)
    options=read(SOURCE/'research_operating_options.json');plan=read(SOURCE/'secondary_operating_points_plan.json')
    budget=p['calibration_FA_budget_per_hour']
    if p['model_tag']!=options['selected_tag'] or budget not in options['available_budgets']:
        raise ValueError('Unverified model or operating point')
    if p['threshold']!=plan['items'][p['model_tag']]['points'][str(budget)]['threshold']:
        raise ValueError('Selected threshold is not the fixed calibration point')
    return p


if __name__=='__main__':
    value=configure_detection_first();load_profile()
    print('Selected',value['model_tag'],'threshold',value['threshold'],'calibration budget',value['calibration_FA_budget_per_hour'])
