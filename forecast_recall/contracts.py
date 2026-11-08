"""Finite event-oriented experiment; previous results never refitted on test."""
from . import ROOT,ARTIFACTS,PARENT
from forecast_navigation.contracts import read,write,sha,digest,now,folds,GOALS,BASELINE
SEED=2026110301


def prepare():
    from forecast_navigation.evaluate import verify_freeze
    verify_freeze()
    p=dict(created_at=now(),version='event-recall-v13-20261103',seed=SEED,seeds=[SEED,SEED+10],
        parent_delivery_sha256=sha(PARENT/'delivery_manifest.json'),parent_freeze_sha256=sha(PARENT/'freeze.json'),
        parent_data_sha256=sha(PARENT/'data_manifest.json'),baseline_sha256=sha(BASELINE),
        primary_task='First-relay service event recall. Physical predictor remains frozen; not a claimed new physical model.',
        variants=['bce','event','memory_event'],new_models=6,updates=4000,learning_rate=.0003,batch_uniform=256,batch_events=32,validate_every=500,
        frozen_representation='Per-seed V12mission GNN current service input481 and logits6; all graph parameters frozen',
        training=dict(flights=245,uniform_queries_per_flight=512,observer='First relay',
            event_queries='All qualified1Hz queries in3<=lead<7s for every service event; also event-10s..event+4s for hard negatives and alignment',
            sampling='Uniform natural stream BCE and precision counts use ONLY512uniform queries/flight, not event oversampling',
            event_loss='Equal-event smooth maximum warning loss over all qualified pre-event query bags',
            precision_loss='Soft predicted-positive counts constrain90% stepwise precision on natural rows; surrogate, exact event metrics remain final authority',
            weights=dict(event=.5,precision=4.),gradient_to_frozen_graph=False),
        memory=dict(period_s=1.,samples=20,source='Only each half-second local RF/intents/measured energy; no packet GT or future input',
            architecture='Shared480-dimensional causal aggregate,64GRU, residual service head'),
        selection='Every500: exact unchanged first-alert recall atFA<=1/h over full45selectionflights; preferoriginal full goals, then min50alerts, then >=10exploratoryalerts. No meanAP checkpoint.',
        calibration='Platt and same event-recall/FA policy selection on45calibration flights only; originalP90/R80/FA1/min50 goals unchanged',
        comparisons=['frozen_parent','threshold_only_parent','bce','event','memory_event','event_blend50','memory_event_blend50'],
        confirmation='42NEWindependent fixedV7 flights on known14maps after ownfreeze; V11/V12confirmation forbidden for fitting/reselection',
        metric='UNCHANGED persistent1s service health, same1Hz warmup/qualification/hysteresis, strict3<=lead<7s first alerts, all events denominator',
        goals=GOALS,original_final_test_opened=False,automatic_PPO_replacement=False,hardware_validated=False,production_ready=False,
        sources=[dict(title='Optimizing Early Warning Classifiers to Control False Alarms via a Minimum Precision Constraint',url='https://proceedings.mlr.press/v151/rath22a.html',
            use='General motivation only. Own event bag and surrogate implementation; not exact reproduction or claimed transfer guarantee.')])
    path=ARTIFACTS/'protocol.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in p.items() if k!='created_at'}:raise ValueError('Study plan changed')
        return old
    write(path,p);return p


def verify():
    p=read(ARTIFACTS/'protocol.json')
    for path,key in [(PARENT/'delivery_manifest.json','parent_delivery_sha256'),(PARENT/'freeze.json','parent_freeze_sha256'),
                     (PARENT/'data_manifest.json','parent_data_sha256'),(BASELINE,'baseline_sha256')]:
        if sha(path)!=p[key]:raise ValueError('Frozen predecessor changed')
    return p


def status(stage,**kwargs):write(ARTIFACTS/'status.json',dict(state='running',stage=stage,updated_at=now(),production_ready=False,**kwargs))
