"""Prospective recall-first study, authorized approximately four FA/h."""
from . import ROOT,ARTIFACTS,PARENT,NAV
from forecast_navigation.contracts import read,write,sha,digest,now,folds,GOALS,BASELINE
SEED=2026110314
CORE=['__init__.py','contracts.py','features.py','data.py','model.py','train.py','metrics.py','inference.py','evaluate.py','calibration.py','compare.py','confirmation.py','references.py','runtime.py','legacy_data.py']

def prepare():
    from forecast_recall.evaluate import verify_freeze
    verify_freeze()
    paths=[PARENT/'delivery_manifest.json',PARENT/'freeze.json',PARENT/'data_manifest.json',NAV/'delivery_manifest.json',BASELINE,ROOT.parent/'forecast_current/active_profile.json']
    p=dict(version='temporal-graph-v14-20261103',created_at=now(),seed=SEED,seeds=[SEED,SEED+10],
        dependencies={str(q):sha(q) for q in paths},user_priority='탐지율을 우선하고 약 4회/h 오경보 허용',
        calibration_FA_budget_per_hour=4.,strict_future_FA_cap=False,minimum_calibration_alerts=50,
        task='First relay packet-service events; unchanged first alerts, all-event denominator, strict3<=lead<7s',
        variants=['short_bce','short_event','long_event'],new_models=6,updates=2400,validate_every=400,
        batch_uniform=96,batch_events=16,learning_rate=.00015,
        initialization='Same per-seed V12mission graph checkpoint for all three variants; optimizer reset; all graph/service parameters trainable',
        training='Exactly V13 train queries, uniform natural rows and complete event bags; no previous confirmation fitting',
        history='Received-only node34 and directed edge4;4x1s or20x1s. Current9-horizon physics/global only. No new radio traffic or truth access.',
        losses=dict(natural_BCE=1.,equal_event_softmax=.5,soft_precision_target=.4,soft_precision_weight=1.,frozen_logit_anchor=.005),
        selection='Exact first-event recall atFA<=4/h on45selection flights; min50alerts first; 0-step eligible. Same threshold grid and criteria for all new candidates.',
        calibration='45calibration flights only: positive-slope Platt, maximum recall atFA<=4/h and>=50alerts; record absence explicitly.',
        confirmation='42new fixedV7 flights after source/model/policy freeze,14known maps,3interference conditions, unique V14 seeds',
        comparisons=['frozen_v13','retuned_v13','short_bce','short_event','long_event','short_event_blend50','long_event_blend50'],
        original_development_goals=GOALS,original_final_test_opened=False,hardware_validated=False,production_ready=False,automatic_PPO_replacement=False,
        research=[dict(url='https://arxiv.org/abs/2006.10637',use='Motivation for per-node/edge temporal memory; this discrete snapshot GRU is not an exact TGN reproduction or a guaranteed gain.'),
                  dict(url='https://proceedings.mlr.press/v151/rath22a.html',use='Recall/precision constrained early-warning motivation; own event loss and soft surrogate, no theorem claimed.')])
    path=ARTIFACTS/'protocol.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in p.items() if k!='created_at'}:raise ValueError('Prospective protocol changed')
        return old
    write(path,p);return p

def verify():
    p=read(ARTIFACTS/'protocol.json')
    for path,value in p['dependencies'].items():
        if sha(path)!=value:raise ValueError('Preserved dependency changed: '+path)
    return p

def status(stage,**kw):write(ARTIFACTS/'status.json',dict(state='running',stage=stage,updated_at=now(),production_ready=False,**kw))
