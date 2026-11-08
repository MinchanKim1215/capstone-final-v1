"""Register before any V14 confirmation predictive outcomes are inspected."""
from . import ROOT,ARTIFACTS,PARENT,GRAPH
from forecast_navigation.contracts import read,write,sha,digest,now,folds,BASELINE,GOALS
SEED=2026110315
CORE=['__init__.py','contracts.py','features.py','model.py','data.py','train.py','metrics.py','inference.py','evaluate.py','calibration.py','compare.py','confirmation.py','runtime.py']

def prepare():
    # V14 is still building input features. No confirmation model outcome read.
    outcome_files=list((GRAPH/'evaluation').glob('*/confirmation/summary.json'))
    if outcome_files:raise ValueError('Too late to register a shared-blind-confirmation study')
    paths=[PARENT/'delivery_manifest.json',PARENT/'freeze.json',PARENT/'data_manifest.json',GRAPH/'freeze.json',GRAPH/'data_manifest.json',ROOT.parent/'forecast_current/active_profile.json',BASELINE]
    value=dict(version='peer-raw-history-v15-20261103',created_at=now(),seed=SEED,seeds=[SEED,SEED+10],dependencies={str(x):sha(x) for x in paths},
        priority='Advance-event recall, user allows approximately4false/late alerts per drone-hour',
        variants=['global_event','peer_bce','peer_event'],new_models=6,updates=4000,validate_every=500,learning_rate=.0003,batch_uniform=256,batch_events=32,
        data='Exactly V13 training245flights/148949cachedqueries/125440naturalrows/2393completeeventbags,45selection,45calibration',
        base='Frozen already-trained V13 memory-event per-seed logits and V12mission481context. New residual final layer zero. No gradient into ancestors.',
        representation='1Hz20samples of119normalized observed RF/masks/currentintents/energy/energyfinite/peerrole per peer; query-conditioned shared64GRU attention. Control uses existing480globalaggregates with same64GRU/head.',
        losses=dict(natural_BCE=1.,event=.5,soft_precision_target=.4,soft_precision_weight=1.,residual_l2=.001),
        selection='Exact first-alert recall FA<=4/h,prefer>=50alerts,0-step eligible,full45selection before confirmation outcome access',
        calibration='Positive-slope Platt and fixed original full threshold sweep on45calibration only; maxrecall FA<=4/h/min50',
        confirmation=dict(source='The same42V14fixed-policy NEWflights on14knownmaps, planned before predictive outcomes',
            generation_timing='Raw flights already generated after V14 freeze, before this V15 registration. This is NOT a claim of another42newV15flights after V15freeze.',
            blindness='No V14confirmation predictions/results inspected for V15 design,training,selection or calibration. Root will keep all predictive outcomes unread until V15freeze.',
            independent_of_training=True,separate_from_original32finalmaps=True),
        comparisons=['frozen_v13','retuned_v13','global_event','peer_bce','peer_event','global_event_blend50','peer_event_blend50'],
        pre_registration_state=read(GRAPH/'status.json'),confirmation_prediction_files_at_registration=len(outcome_files),original_goals=GOALS,
        original_final_test_opened=False,production_ready=False,hardware_validated=False,automatic_PPO_replacement=False)
    path=ARTIFACTS/'protocol.json'
    if path.exists():raise ValueError('Already registered; use verify')
    write(path,value);return value

def verify():
    p=read(ARTIFACTS/'protocol.json')
    for path,value in p['dependencies'].items():
        if sha(path)!=value:raise ValueError('Preserved source changed: '+path)
    return p

def status(stage,**kw):write(ARTIFACTS/'status.json',dict(state='running',stage=stage,updated_at=now(),production_ready=False,**kw))
