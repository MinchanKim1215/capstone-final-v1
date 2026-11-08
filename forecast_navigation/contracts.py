"""V12: matched continuation only after the navigation-quality gate passes."""
from . import ROOT,ARTIFACTS
import forecast_graph
from forecast_graph.contracts import read,write,sha,digest,now,source_records,folds,GOALS,BASELINE
PARENT=forecast_graph.ARTIFACTS
SEED=2026110205


def prepare():
    from forecast_graph.evaluate import verify_freeze
    verify_freeze();gate=read(ARTIFACTS/'gate_result.json')
    if not gate['passed']:raise ValueError('Do not train an unqualified navigation revision')
    value=dict(created_at=now(),version='navigation-plan-v12-20261102',seed=SEED,seeds=[SEED,SEED+10],
        gate_sha256=sha(ARTIFACTS/'gate_result.json'),parent_freeze_sha256=sha(PARENT/'freeze.json'),
        parent_data_sha256=sha(PARENT/'data_manifest.json'),parent_training_sha256=sha(PARENT/'training_plan.json'),
        profiles=['original','mission'],
        initialization='Per-seed frozen V11 hybrid best weights; same initialization for both profiles; optimizer reset for both',
        updates=6000,batch_size=128,learning_rate=.0002,validate_every=500,
        data='Exactly the parent94080trainingqueries and fixed11520selectionprobe. No confirmation fitting.',
        packet=dict(bytes=128,period_s=1.,new_plan_bytes=[30,66],original_tail_knots_bytes=[114,126],extension_flag_byte=126,
            meaning='Mission-aware prospective path plus original tail knots. Both views reconstructed from the same received packet; no second transmission.',extra_airtime_s=0.),
        nav_input='Issuing controller CURRENT navigation state and public mission/map; held current correction. No RF/packet/actual future action input.',
        candidates=['frozen_v11_selected','continued_original','continued_mission','original_blend50','mission_blend50'],
        comparison='Equal-seed ensembles; original versusmission continuation isolates plan quality, not longer fit budget.',
        confirmation='42 new independent fixedV7 flights after this study freeze, known14maps, noV11confirmationreuse',
        primary_observer='First relay, same original physical/service targets and qualification',goals=GOALS,
        original_final_test_opened=False,automatic_PPO_replacement=False,production_ready=False,hardware_validated=False)
    path=ARTIFACTS/'protocol.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:raise ValueError('V12 plan cannot change silently')
        return old
    write(path,value);return value


def verify():
    p=read(ARTIFACTS/'protocol.json')
    for path,key in [(ARTIFACTS/'gate_result.json','gate_sha256'),(PARENT/'freeze.json','parent_freeze_sha256'),
                     (PARENT/'data_manifest.json','parent_data_sha256'),(PARENT/'training_plan.json','parent_training_sha256')]:
        if sha(path)!=p[key]:raise ValueError('V12 parent or gate changed')
    return p


def status(stage,**kwargs):write(ARTIFACTS/'status.json',dict(state='running',stage=stage,updated_at=now(),production_ready=False,**kwargs))

