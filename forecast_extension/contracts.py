"""Predeclared spatial boundaries and immutable predecessor references."""
from pathlib import Path
from . import ROOT,ARTIFACTS,WORKSPACE,VERSION
from forecast_rebuild.contracts import read,write,sha,digest,now,maps,folds,role,GOALS,BASELINE
from forecast_rebuild.calibration import verify_freeze as verify_parent_freeze
from forecast_rebuild import ARTIFACTS as PARENT_ARTIFACTS

SEED=2026110202
OLD_SAMPLES=PARENT_ARTIFACTS/'samples/expanded_planned'
OLD_DEVELOPMENT=PARENT_ARTIFACTS/'benchmarks/balanced_development_dataset.json'
INITIAL=PARENT_ARTIFACTS/'models/expanded_residual_planned_energy/latest.pt'
OLD_NEURAL=PARENT_ARTIFACTS/'models/expanded_residual_planned_energy/model.json'
OLD_TREE=PARENT_ARTIFACTS/'models/expanded_lgbm_planned_energy/model.json'


def prepare():
    parent=verify_parent_freeze();registry=maps()
    value=dict(version=VERSION,seed=SEED,created_at=now(),
        parent_freeze_path=str(PARENT_ARTIFACTS/'model_alarm_freeze.json'),parent_freeze_sha256=sha(PARENT_ARTIFACTS/'model_alarm_freeze.json'),
        immutable_inputs={str(path):sha(path) for path in (OLD_SAMPLES/'manifest.json',OLD_DEVELOPMENT,INITIAL,OLD_NEURAL,OLD_TREE)},
        train_regions=sorted(k for k,m in registry.items() if m['joint_split']=='train'),development_folds=folds(),
        sealed_test_regions=sorted(k for k,m in registry.items() if m['joint_split']=='test'),
        extension=dict(initial_step=5000,total_steps=20000,additional_updates=15000,batch_size=256,
                       warmup_updates=250,peak_lrs=[3e-6,1e-5,1e-4],cosine_floor=.1,validate_every=1000,
                       fixed_update_budget=True,checkpoint_selection='Best common AP5/AP37 mean on a fixed balanced selection probe; include step5000'),
        new_experience=dict(train_flights=245,selection_flights=45,calibration_flights=45,confirmation_flights=42,
            policy='Frozen v7 step64 for identical-observation predictor comparisons',
            training='One fresh seed/direction/deployment case per training map; types and four deployments stratified',
            development='All three interference conditions on every development map with one new seed and opposite v8 direction',
            augmented_train_flights=84,augmented_quota=1200,clean_quota_on_augmented_flight=1200,other_new_flight_quota=2400,
            gps_augmented_flights=63,gps_axis_sigmas_m=[1,2,4],sensor_only_augmented_flights=21,
            sensor_perturbations=['additional_2p5dB','missing'],
            note='These are simulation stress assumptions, not a selected hardware error distribution'),
        training_comparison='Same-data continuation and expanded-data continuation use the same step5000 weights/optimizer and the same 15000-update schedule.',
        model_selection='Old and new selection flights only. Calibration/confirmation geographies never enter supervised training.',
        final_confirmation='Fresh flight realizations on previously seen development confirmation maps. Not novel-geography performance.',
        previous_confirmation_reuse='Old v8 confirmation outcomes are excluded from optimization, model selection and reported new confirmation.',
        goals=GOALS,operating_points=['raw0.5','existing_advisory_rule','calibration_FA_budget1','precision90_FA1_partial'],
        production_ready=False,hardware_validated=False,original_final_test_opened=False)
    path=ARTIFACTS/'protocol.json'
    if path.exists():
        saved=read(path)
        for key in value:
            if key!='created_at' and saved[key]!=value[key]:raise ValueError('An existing V9 boundary changed: '+key)
        return saved
    write(path,value);return value


def verify_protocol():
    value=read(ARTIFACTS/'protocol.json');verify_parent_freeze()
    if sha(value['parent_freeze_path'])!=value['parent_freeze_sha256']:raise ValueError('V8 comparison changed')
    for path,expected in value['immutable_inputs'].items():
        if sha(path)!=expected:raise ValueError('Predecessor input changed: '+path)
    return value


def pin(names):return {name:sha(ROOT/name) for name in names}


if __name__=='__main__':print(prepare()['version'])

