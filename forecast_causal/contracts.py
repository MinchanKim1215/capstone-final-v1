"""User's four stages persist across turns; frozen ancestors are read-only."""
from . import ROOT,ARTIFACTS,WORKSPACE,PARENT,NAV
from forecast_navigation.contracts import read,write,sha,digest,now,folds,BASELINE,GOALS
import os
os.environ['NUMBA_CACHE_DIR']=str(ARTIFACTS/'numba_cache')
SEED=2026110316

def prepare():
    paths=[PARENT/'delivery_manifest.json',PARENT/'freeze.json',PARENT/'data_manifest.json',NAV/'delivery_manifest.json',
        WORKSPACE/'forecast_temporal/artifacts/delivery_manifest.json',WORKSPACE/'forecast_peer/artifacts/delivery_manifest.json',WORKSPACE/'forecast_current/active_profile.json',BASELINE]
    value=dict(version='cause-and-packet-state-v16',created_at=now(),seed=SEED,dependencies={str(p):sha(p) for p in paths},
        explicit_user_request='Persist and follow through: missed-event cause decomposition, observable precursor augmentation, cause-specific physics/LightGBM/GRU comparison, same recall/precision/FA validation.',
        priority='Increase3s+advance event recall at approximately4false/late alarms per drone-hour, maintain precision; do not remove hard-to-predict events.',
        stages=['selection45_event_cause_reconstruction','causal_measured_packet_state_and_received_beacons','controlled_physics_tree_GRU_experts','frozen_new42_confirmation_actual_runtime_integrity_report'],
        stage1='Only selection45flights diagnosed; same full packet/motion/RF/beacon replay. Multi-label failure evidence, no exclusive unsupported causal claims.',
        observability='Simulator truth/counterfactuals allowed for diagnosis or labels ONLY. Prediction uses current/past local queue and actual received128Bbeacons.',
        radio='Use unused reserved RF report slots10..15 and power slots10..15 within128B only for fleets<=10. Original payload must restore EXACTLY. Same receipt success/airtime.',
        data='Existing245train/45selection/45calibration; no V11-V15confirmation fitting. New confirmation42after all model/source/threshold freeze; original32unused.',
        tracking='PROJECT_FOLLOWUP.md and artifacts/status.json updated at every major stage with evidence and next action. Never claim launched jobs completed.',
        original_development_goals=GOALS,original_final_test_opened=False,automatic_PPO_replacement=False,hardware_validated=False,production_ready=False)
    target=ARTIFACTS/'protocol.json'
    if target.exists():
        old=read(target)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('Registered four-stage scope changed')
        return old
    write(target,value);return value

def verify():
    value=read(ARTIFACTS/'protocol.json')
    for p,v in value['dependencies'].items():
        if sha(p)!=v:raise ValueError('Preserved ancestor changed: '+p)
    return value

def status(stage,**kw):
    value=dict(state='running',stage=stage,updated_at=now(),production_ready=False,**kw);write(ARTIFACTS/'status.json',value)
    from .ledger import refresh
    refresh(value)

def milestone(stage,evidence,next_action):
    p=ARTIFACTS/'milestones.json';value=read(p) if p.exists() else {}
    value[str(stage)]=dict(completed_at=now(),evidence=evidence,next_action=next_action);write(p,value)
    from .ledger import refresh
    refresh(read(ARTIFACTS/'status.json'))
