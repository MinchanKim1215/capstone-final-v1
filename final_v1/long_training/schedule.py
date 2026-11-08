"""Separate training, selection, calibration and confirmation geography."""
from collections import Counter, defaultdict
import os

from adaptive_joint_expanded.contracts import DEFAULT
from fast_halfsec.context import registry
from fast_halfsec.protocol import KINDS, choose, make_spec
from final_v1.transformer_upgrade import ARTIFACTS as UPGRADE
from final_v1.transformer_upgrade.release import installed
from forecast_rebuild.contracts import BASELINE, digest, now, read, sha, write

from . import ARTIFACTS, MAP_ARTIFACTS, ROOT, WORKSPACE

VERSION = 'Final V1 long joint training'
PLAN = ARTIFACTS / 'protocol.json'
STATUS = ARTIFACTS / 'status.json'
LATEST = ARTIFACTS / 'latest.json'
START = ARTIFACTS / 'started.json'
SEED = 2026110819
BATCH = 8


def status(stage, **fields):
    write(STATUS, dict(version=VERSION, stage=stage, state=fields.pop('state', 'running'),
                       pid=os.getpid(), updated_at=now(), **fields))


def stratum(row):
    return row.get('source_archive', row.get('province', row['key'].split('_', 1)[0]))


def train_order(entries):
    queues = defaultdict(list)
    for row in entries:
        queues[stratum(row)].append(row)
    for queue in queues.values():
        queue.sort(key=lambda row: digest([SEED, 'train-order', row['key']]))
    order = []
    for index in range(max(map(len, queues.values()))):
        for key in sorted(queues):
            if index < len(queues[key]):
                order.append(queues[key][index])
    return order


def prepare():
    installed()
    originals = (BASELINE, MAP_ARTIFACTS / 'map_registry.json', MAP_ARTIFACTS / 'protocol.json',
                 WORKSPACE / 'halfsec_choice/artifacts/protocol.json', UPGRADE / 'protocol.json',
                 UPGRADE / 'integration.json', UPGRADE / 'freeze.json',
                 WORKSPACE / 'final_v1/artifacts/release.json',
                 WORKSPACE / 'adaptive_joint_expanded/learning.py', WORKSPACE / 'adaptive_joint_expanded/data.py',
                 ROOT / '__init__.py', ROOT / 'episode.py', ROOT / 'schedule.py', ROOT / 'run.py')
    pins = {str(path): sha(path) for path in originals}
    if PLAN.exists():
        plan = read(PLAN)
        if plan['pins'] != pins:
            raise ValueError('The frozen training inputs have changed')
        return plan

    maps = registry()
    train = train_order([m for m in maps.values() if m['study_role'] == 'train'])
    train_cases = [make_spec(row, f'long_joint_train_{i}_{SEED}', KINDS[i % 3])
                   for i, row in enumerate(train)]
    while len(train_cases) % BATCH:
        i = len(train_cases)
        train_cases.append(make_spec(train[0], f'long_joint_extra_{i}_{SEED}', KINDS[i % 3]))

    cal = [m for m in maps.values() if m['study_role'] == 'calibration']
    select = choose(cal, 10, 'long_joint_selection', stratum)
    seen = {m['study_group'] for m in select}
    calibration = choose([m for m in cal if m['study_group'] not in seen], 16, 'long_joint_calibration', stratum)
    supervise = choose(train, 96, 'long_joint_supervision', stratum)
    used_specs = (read(MAP_ARTIFACTS / 'protocol.json')['confirmation']
                  + read(WORKSPACE / 'halfsec_choice/artifacts/protocol.json')['fresh_confirmation']
                  + read(UPGRADE / 'protocol.json')['cohorts']['confirmation'])
    used_groups = {maps[spec['region']]['study_group'] for spec in used_specs}
    confirmed = [m for m in maps.values() if m['study_role'] == 'confirmation']
    fresh = [m for m in confirmed if m['study_group'] not in used_groups]
    fresh_count = min(15, len({m['study_group'] for m in fresh}))
    new_holdout = choose(fresh, fresh_count, 'long_joint_new_holdout', stratum) if fresh_count else []
    repeated_holdout = choose([m for m in confirmed if m['study_group'] in used_groups],
                              15 - fresh_count, 'long_joint_repeated_holdout', stratum)
    holdout = new_holdout + repeated_holdout
    groups = [{m['study_group'] for m in cohort} for cohort in (train, select, calibration, holdout)]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
        raise ValueError('Training and evaluation geography overlap')

    updates = len(train_cases) // BATCH
    config = dict(DEFAULT, seed=SEED, prediction_mode='joint', updates=updates, epochs=2,
                  batch_episodes=BATCH, minibatch=24, actor_lr=5e-5, encoder_lr=8e-6, timing_lr=4e-5,
                  critic_lr=1.5e-4, selector_lr=3e-5, prediction_coef=.35, target_kl=.012,
                  selector_temperature=1.6, onset_cost=.75, device='cuda', workers=6)
    marks = sorted({updates // 4, updates // 2, 3 * updates // 4, updates})
    plan = dict(version=VERSION, created_at=now(), pins=pins, config=config,
                train=train_cases, unique_train_maps=len(train), new_train_maps=sum(m['new_map'] for m in train),
                training_group_count=len(groups[0]), control_s=3., radio_s=.5,
                interference_counts=dict(Counter(s['kind'] for s in train_cases)),
                checkpoints_for_selection=[0, *marks],
                selection=[make_spec(m, 'long_joint_selection', KINDS[i % 3]) for i, m in enumerate(select)],
                supervise=[make_spec(m, 'long_joint_supervision', KINDS[i % 3]) for i, m in enumerate(supervise)],
                calibration=[make_spec(m, 'long_joint_calibration', KINDS[i % 3]) for i, m in enumerate(calibration)],
                confirmation=[make_spec(m, 'long_joint_holdout', kind) for m in holdout for kind in KINDS],
                new_confirmation_maps=[m['key'] for m in new_holdout],
                previously_evaluated_confirmation_maps=[m['key'] for m in repeated_holdout],
                goals=dict(false_per_fleet_hour=4., max_pdr_loss_pp=.25, max_duration_increase=.02),
                fixed_software='Motion and collision constraints, radio physics, packet deadlines and V21 safety checks.',
                training='Raw RF and received plans. On-policy proposals with native safety intervention and outage-onset reward.')
    write(PLAN, plan)
    return plan
