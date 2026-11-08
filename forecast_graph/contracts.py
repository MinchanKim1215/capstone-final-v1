"""Preserve prior deliveries and predeclare data, protocol and study choices."""
from . import ROOT, ARTIFACTS, WORKSPACE, VERSION
from forecast_rebuild.contracts import read, write, sha, digest, now, folds, role, GOALS, BASELINE
from forecast_service import ARTIFACTS as V10
from forecast_extension import ARTIFACTS as V9

SEED = 2026110204


def prepare():
    from forecast_service.evaluate import verify_freeze
    verify_freeze()
    plan = dict(version=VERSION, created_at=now(), seed=SEED, seeds=[SEED, SEED + 10],
        parent_delivery_sha256=sha(V10 / 'delivery_manifest.json'),
        parent_freeze_sha256=sha(V10 / 'freeze.json'),
        baseline_policy_sha256=sha(BASELINE),
        records_sha256=sha(V9 / 'dataset.json'),
        training='245 original fresh V9 train flights; no past confirmation fitting',
        selection='45 V9 selection flights', calibration='45 V9 calibration flights',
        confirmation='42 new flights on existing14 confirmation maps, generated only after freeze',
        spatial_folds=folds(), original_final_test_opened=False,
        beacon=dict(bytes=128, bits=1024, period_s=1., reserved_slot_s=.005,
            receiver_rule='Only original successful 1024-bit beacons. Simultaneous payload snapshots use strictly previous received RF caches.',
            extra_airtime_s=0., max_nodes=16,
            note='Explicit alternative payload within existing128B model; not a hardware-certified MAC implementation.'),
        plan=dict(horizons_s=list(range(9)), xy_quantization_m=.25,
            committed_segment_s=3., refresh='Every issued3-second controller motion; sent at1Hz',
            beyond_commitment='Tentative graph path to the current issued target; future policy commands, actual future positions and pauses are unavailable.'),
        graph=dict(history_s=[0, 1, 2, 3], hidden=64, message_passing_steps=2, temporal='GRU',
            node_identity='Roles and physical relations, never map IDs or source seeds'),
        physics=dict(horizons_s=list(range(9)), branches=['received_plan', 'constant_velocity'],
            reference='Public GIS propagation plus received signal residual; correlated stochastic uncertainty, no hidden RF inputs',
            learned_correction='Future SINR margin auxiliary supervision and direct persistent-onset heads'),
        variants=['graph_local', 'graph_shared', 'hybrid_shared'],
        tree_control='Same shared/hybrid observations aggregated into a LightGBM control',
        per_flight_train_queries=384, train_observers='All relay observers balanced by deterministic query permutation',
        neural=dict(updates=6000, batch_size=128, lr=.0004, weight_decay=.0001,
            validate_every=500, checkpoint='Selection AP5/AP37 mean; no calibration or confirmation gradients'),
        candidate_definition='Equal-weight ensemble of two predeclared seeds per variant',
        tasks=dict(link='Unchanged persistent physical-link onset and3-to7 first-alert rules',
            service='Unchanged V10 rolling1s timelyPDR<90%, deadline250ms, persistence1s; designated first relay'),
        goals=GOALS, automatic_PPO_replacement=False, production_ready=False, hardware_validated=False)
    path=ARTIFACTS/'protocol.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!={k:v for k,v in plan.items() if k!='created_at'}:
            raise ValueError('Existing V11 plan cannot change silently')
        return old
    write(path, plan)
    return plan


def verify():
    p=read(ARTIFACTS/'protocol.json')
    for path,key in [(V10/'delivery_manifest.json','parent_delivery_sha256'),(V10/'freeze.json','parent_freeze_sha256'),
                     (BASELINE,'baseline_policy_sha256'),(V9/'dataset.json','records_sha256')]:
        if sha(path)!=p[key]:raise ValueError('Read-only parent changed: '+str(path))
    return p


def status(stage, **kwargs):
    write(ARTIFACTS/'status.json',dict(state='running',stage=stage,updated_at=now(),production_ready=False,**kwargs))


def source_records():
    return [r for r in read(V9/'dataset.json')['rows'] if r['source_kind']=='fresh_v9' and r['role'] in ('train','selection','calibration')]

