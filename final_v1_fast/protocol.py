"""Freeze cohorts, clocks, acceptance rules and code before paired branches."""
from . import ROOT,WORKSPACE,ARTIFACTS,VERSION
from forecast_rebuild.contracts import read,write,sha,digest,now,folds,maps

ARMS={
 'p3_b1':dict(plan_s=3.,beacon_s=1.,emergency_s=None),
 'p1_b1':dict(plan_s=1.,beacon_s=1.,emergency_s=None),
 'p05_b1':dict(plan_s=.5,beacon_s=1.,emergency_s=None),
 'p05_b10':dict(plan_s=.5,beacon_s=.1,emergency_s=None),
 'p05_b10_e02':dict(plan_s=.5,beacon_s=.1,emergency_s=.2),
 'p05_b10_e01':dict(plan_s=.5,beacon_s=.1,emergency_s=.1),
 'p05_bhybrid':dict(plan_s=.5,beacon_s=.1,emergency_s=None,hybrid=True),
 'p05_bhybrid_e01':dict(plan_s=.5,beacon_s=.1,emergency_s=.1,hybrid=True),
}
CONFIG=dict(dt_s=.1,branch_duration_s=24.,forecast_s=8.,command_period_s=.1,
            command_latency_s=.1,stress_command_latency_s=.3,
            vmax_mps=5.,amax_mps2=2.5,separation_m=5.,
            minimum_target_hold_s=.5,emergency_minimum_gap_s=.2,
            margin_target_db=1.5,minimum_gain_db_seconds=1.,
            early_tolerance_db=.25,minimum_normal_margin_s=6.,
            fresh_report_s=2.,emergency_drop_db=1.5,emergency_drop_window_s=.2,
            beacon_slot_s=.005,command_slots_per_frame=3,
            sensor_processing_s=.1,planning_compute_s=.2,prediction_time_budget_ms=200.,
            model_choices='Frozen physics equations, no neural or LightGBM refit')


def status(stage,**kwargs):
    write(ARTIFACTS/'status.json',dict(version=VERSION,stage=stage,state=kwargs.pop('state','running'),
                                    updated_at=now(),**kwargs))


def prepare():
    sources=[]
    for spec in read(WORKSPACE/'final_v1/artifacts/protocol.json')['confirmation']:
        path=WORKSPACE/'final_v1/artifacts/confirm/v21'/f'{spec["region"]}_{spec["kind"]}.json'
        meta=read(path)
        if meta['spec']!=spec or sha(meta['archive_path'])!=meta['archive_sha256']:
            raise ValueError('Source Final V1 changed')
        sources.append(dict(path=str(path),sha256=sha(path),region=spec['region'],kind=spec['kind']))
    value=dict(version=VERSION,created_at=now(),arms=ARMS,config=CONFIG,sources=sources,
        preregistration_amendment='Before any new confirmation branch outcomes, add two hybrid beacon arms '
            'after development exposed all-node 10 Hz airtime. At 1 Hz all drones report; between those '
            'cycles the two endpoints of the predicted bottleneck link are requested to report at 10 Hz. '
            'Rate changes share the delayed lossy command packet; old rates remain until received. '
            'Original six-arm registration remains in protocol.json.',
        scope='Matched 24-second branches from existing Final V1 flights. Engineering diagnosis, '
              'not full-mission PPO retraining or an unseen-map deployment estimate.',
        initialization='All arms share source grid positions and zero initial velocities; queues initially empty. '
                       'Source onsets locate scenarios only; new fine-time events are counted independently.',
        inputs='Current shared navigation state and prospective mission plan; only successfully received '
               'RF reports with measurement and reception times. No future RF, jammer state or future recorded actions.',
        public_plan='Original navigation generator replayed exactly to branch time. The source command issued '
                    'at that time is held prospectively; no later source commands or positions enter the controller.',
        fine_shadow='OU bridges at 0.1 s conditional on the frozen 0.5 s exogenous process; '
                    'new stochastic fine-time simulation, never interpolation claimed as observations.',
        communication='One 5 ms beacon slot per drone per beacon cycle, three command forwarding slots per '
                      '100 ms. Those slots are removed from the same 30 packets/s data channel.',
        computation='A single planning worker occupies 200 ms per assessment, following the standalone '
                    'development timing check. Results cannot be transmitted before that delay. '
                    '10 Hz monitoring is cheap; overlapping full planning tasks are skipped and counted.',
        selection='Up to three distinct outage-window anchors and one non-outage anchor per source flight, '
                  'deterministically chosen before new branch outcomes. All cases and difficult failures retained.',
        acceptance='Zero safety violations; no pooled timely PDR loss above 0.25 percentage points; '
                   'paired map-bootstrap 95% upper bound of persistent-start difference below zero; '
                   'relay travel/reversals and computation latency disclosed. No automatic production promotion.',
        primary_metric='New >=1-second persistent network outages in matched diagnostic branch windows',
        no_operational_false_alarm_claim='Risk-conditioned windows cannot estimate whole-mission false alarms/hour.',
        protected_dependencies={str(p):sha(p) for p in [WORKSPACE/'final_v1/control.py',
             WORKSPACE/'final_v1/runner.py',WORKSPACE/'forecast_learning/artifacts/freeze.json',
             WORKSPACE/'adaptive_joint_expanded/world.py',WORKSPACE/'drone_forecast_joint_training/cooperative/packets.py']})
    path=ARTIFACTS/'protocol_v2.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:
            raise ValueError('Frozen protocol changed')
        return old
    write(path,value)
    return value


def freeze(cases):
    files=sorted(ROOT.glob('*.py'))
    value=dict(protocol_sha256=sha(ARTIFACTS/'protocol_v2.json'),cases_sha256=sha(ARTIFACTS/'cases.json'),
               source_sha256={p.name:sha(p) for p in files},cases=len(cases),created_at=now(),
               development_timing_sha256=sha(ARTIFACTS/'development_first.json'))
    path=ARTIFACTS/'freeze.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in value.items() if k!='created_at'}:
            raise ValueError('Frozen implementation changed after confirmation')
        return old
    write(path,value)
    return value
