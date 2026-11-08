"""Study cohorts and locked comparisons, chosen before confirmation flights."""
from collections import Counter, defaultdict

from forecast_rebuild.contracts import digest, now, read, sha, write
from . import ARTIFACTS, ROOT, VERSION, WORKSPACE

REGISTRY = ARTIFACTS / 'map_registry.json'
PLAN = ARTIFACTS / 'protocol.json'
KINDS = ('clear', 'intermittent', 'continuous')
SEED = 2026110705
CONTROL_MODES = ('final_v1_3s', 'physics_3s', 'physics_05s', 'learned_05s', 'ppo_05s')
ALERT_CAP = 4.
SIM_SOURCES = ('__init__.py', 'navigation.py', 'context.py', 'runner.py', 'controller.py',
               'features.py', 'actor.py', 'protocol.py', 'rollouts.py')
UPSTREAM = ('additional_maps_20261030/expanded_plan.json', 'adaptive_joint_expanded/world.py',
            'adaptive_joint_expanded/artifacts/protocol.json',
            'adaptive_joint_expanded/artifacts/runs/adaptive_v7_expanded100_20261031/checkpoints/s0000064.pt',
            'drone_forecast_joint_training/cooperative/packets.py',
            'drone_forecast_joint_training/cooperative/motion.py',
            'final_v1/control.py', 'final_v1/runner.py', 'final_v1/artifacts/release.json',
            'forecast_learning/artifacts/freeze.json', 'final_v1_fast/motion.py',
            'final_v1_fast/controller.py', 'final_v1_fast/sensing.py')


def status(stage, **values):
    write(ARTIFACTS / 'status.json', dict(version=VERSION, stage=stage,
                                         state=values.pop('state', 'running'), updated_at=now(), **values))


def choose(entries, count, tag, strata):
    queues = defaultdict(list)
    for entry in entries:
        queues[strata(entry)].append(entry)
    for values in queues.values():
        values.sort(key=lambda item: digest([SEED, tag, item['key']]))
    chosen, groups = [], set()
    keys = sorted(queues, key=lambda key: digest([SEED, tag, key]))
    while len(chosen) < count:
        before = len(chosen)
        for key in keys:
            values = queues[key]
            while values and values[-1]['study_group'] in groups:
                values.pop()
            if values:
                value = values.pop()
                chosen.append(value)
                groups.add(value['study_group'])
            if len(chosen) == count:
                break
        if len(chosen) == before:
            raise ValueError(f'Not enough independent map groups for {tag}')
    return chosen


def make_spec(entry, cohort, kind):
    token = digest([SEED, cohort, entry['key']])
    return dict(region=entry['key'], direction=('east_west', 'north_south')[int(token[8:12], 16)%2],
                seed=int(token[:8],16)%(2**31-3000010), kind=kind)


def verify(plan):
    if sha(REGISTRY) != plan['registry_sha256']:
        raise ValueError('Map registry changed after cohort freeze')
    for name, expected in plan['simulation_sources'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen simulation source changed: {name}')
    for name, expected in plan['upstream_sources'].items():
        if sha(WORKSPACE / name) != expected:
            raise ValueError(f'Frozen prior dependency changed: {name}')


def prepare():
    if PLAN.exists():
        plan = read(PLAN)
        verify(plan)
        return plan
    entries = read(REGISTRY)['maps']
    train_new = [row for row in entries if row['study_role']=='train' and row['new_map']]
    train_old = [row for row in entries if row['study_role']=='train' and not row['new_map']]
    strata = lambda row: row.get('source_archive', row['key'].split('_',1)[0])
    old = choose(train_old, 12, 'historical_train', strata)
    used = {row['study_group'] for row in old}
    new = choose([row for row in train_new if row['study_group'] not in used], 48, 'new_train', strata)
    train = new + old
    calibration = choose([row for row in entries if row['study_role']=='calibration'],
                         15, 'calibration', strata)
    confirmation = choose([row for row in entries if row['study_role']=='confirmation'],
                          12, 'confirmation', strata)
    groups = [{row['study_group'] for row in cohort} for cohort in (train, calibration, confirmation)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Cohort geography crosses roles')
    policy_batches = []
    for update in (1,2):
        batch = choose(train, 16, f'ppo_update_{update}', strata)
        policy_batches.append([make_spec(row, f'ppo_update_{update}', KINDS[(i+update)%3])
                               for i, row in enumerate(batch)])
    plan = dict(version=VERSION, created_at=now(), registry_sha256=sha(REGISTRY),
        train=[make_spec(row, 'supervised_train', KINDS[i%3]) for i,row in enumerate(train)],
        calibration=[make_spec(row, 'calibration', KINDS[i%3]) for i,row in enumerate(calibration)],
        confirmation=[make_spec(row, 'confirmation', kind) for row in confirmation for kind in KINDS],
        ppo_batches=policy_batches, modes=CONTROL_MODES, alert_cap_per_fleet_hour=ALERT_CAP,
        windows_s={'early':[3.,7.], 'short':[.5,3.]},
        clocks=dict(motion_s=.1, radio_s=.5, beacon_s=1., planning_s=.5, native_s=3.,
                    compute_s=.2, sensor_s=.1, command_delivery_s=.1),
        control_rule='One relay can change to a physics-checked neighboring offset. A residual PPO actor ranks options.',
        forecast_rule='Independent binary LightGBM heads for next-cut windows. Compare with physics and a model without the past 30 s history.',
        endpoints='Actual mission completion, new persistent onsets per hour, total cut time, timely PDR and movement; advisory recall and AP on a common fixed trace.',
        selection_rule='Train only on registered training maps; calibrate alert cutoffs on new calibration geography; use fixed PPO update 2; no model selection on confirmation.',
        acceptance=dict(safety_violations=0, completion_regressions=0, max_timely_loss_pp=.25,
                        max_duration_increase_fraction=.05, paired_outage_rate_upper_95_below_zero=True),
        simulation_sources={name:sha(ROOT/name) for name in SIM_SOURCES},
        upstream_sources={name:sha(WORKSPACE/name) for name in UPSTREAM})
    write(PLAN, plan)
    return plan
