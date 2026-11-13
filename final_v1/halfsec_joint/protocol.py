"""Freeze the original model, geography, and paired comparison before training."""

import os

from fast_halfsec.context import registry
from fast_halfsec.protocol import KINDS, PLAN as FAST_PLAN, REGISTRY, choose, make_spec, verify
from final_v1.long_training.schedule import train_order
from forecast_rebuild.contracts import now, read, sha, write

from . import ARTIFACTS, ROOT, WORKSPACE


VERSION = 'Final V1 0.5s Transformer + PPO joint training'
PLAN = ARTIFACTS / 'protocol.json'
SEED = 2026110905
UPDATES = 80
BATCH = 6
ARMS = ('native_3s', 'warm_3s', 'warm_05s', 'joint_05s')


def status(stage, **fields):
    write(ARTIFACTS / 'status.json', dict(version=VERSION, stage=stage, pid=os.getpid(),
                                         updated_at=now(), state=fields.pop('state', 'running'),
                                         **fields))


def prepare():
    prior = read(FAST_PLAN)
    verify(prior)
    prior_paths = (
        FAST_PLAN, REGISTRY, WORKSPACE / 'halfsec_choice/choice.py',
        WORKSPACE / 'forecast_learning/artifacts/freeze.json',
        WORKSPACE / 'final_v1/transformer_upgrade/artifacts/integration.json',
        WORKSPACE / 'final_v1/transformer_upgrade/artifacts/freeze.json',
        WORKSPACE / 'adaptive_joint_expanded/observation.py',
        WORKSPACE / 'adaptive_joint_expanded/targets.py',
    )
    prior_paths += tuple(ROOT / name for name in
                         ('__init__.py', 'model.py', 'observations.py', 'mission.py',
                          'training.py', 'protocol.py', 'run.py'))
    pins = {path.relative_to(WORKSPACE).as_posix(): sha(path) for path in prior_paths}
    if PLAN.exists():
        plan = read(PLAN)
        if plan['pins'] != pins:
            raise ValueError('The frozen half-second study inputs have changed')
        return plan

    maps = registry()
    entries = list(maps.values())
    train = train_order([entry for entry in entries if entry['study_role'] == 'train'])
    strata = lambda row: row.get('source_archive', row['key'].split('_', 1)[0])
    development = [entry for entry in entries if entry['study_role'] == 'calibration']
    select = choose(development, 10, 'halfsec_joint_selection', strata)
    used = {entry['study_group'] for entry in select}
    calibrate = choose([entry for entry in development if entry['study_group'] not in used],
                       16, 'halfsec_joint_calibration', strata)

    previous = prior['confirmation'] + read(WORKSPACE / 'halfsec_choice/artifacts/protocol.json')['fresh_confirmation']
    previous += read(WORKSPACE / 'final_v1/transformer_upgrade/artifacts/protocol.json')['cohorts']['confirmation']
    previous += read(WORKSPACE / 'final_v1/long_training/artifacts/protocol.json')['confirmation']
    previous_groups = {maps[spec['region']]['study_group'] for spec in previous}
    eligible = [entry for entry in entries if entry['study_role'] == 'confirmation']
    unused = [entry for entry in eligible if entry['study_group'] not in previous_groups]
    fresh_count = min(15, len({entry['study_group'] for entry in unused}))
    fresh = choose(unused, fresh_count, 'halfsec_joint_fresh', strata) if fresh_count else []
    repeated = choose([entry for entry in eligible if entry['study_group'] in previous_groups],
                      15 - len(fresh), 'halfsec_joint_repeat', strata)
    confirmation = fresh + repeated
    split = [{entry['study_group'] for entry in group}
             for group in (train, select, calibrate, confirmation)]
    if any(left & right for i, left in enumerate(split) for right in split[i+1:]):
        raise ValueError('The study contains overlapping geographical roles')
    train_specs = [make_spec(train[i % len(train)], f'halfsec_joint_train_{i}_{SEED}', KINDS[i % 3])
                   for i in range(BATCH * UPDATES)]
    stress = next(spec for spec in prior['confirmation'] if
                  spec['region'] == 'gyeonggi_20261031_b2_05' and spec['kind'] == 'intermittent')
    plan = dict(version=VERSION, created_at=now(), pins=pins, updates=UPDATES, batch=BATCH,
                train_maps=len(train), train=train_specs,
                selection=[make_spec(entry, 'halfsec_joint_selection', KINDS[i % 3])
                           for i, entry in enumerate(select)],
                calibration=[make_spec(entry, 'halfsec_joint_calibration', KINDS[i % 3])
                             for i, entry in enumerate(calibrate)],
                confirmation=[make_spec(entry, 'halfsec_joint_confirmation', kind)
                              for entry in confirmation for kind in KINDS],
                fresh_confirmation_maps=[entry['key'] for entry in fresh],
                prior_confirmation_maps=[entry['key'] for entry in repeated],
                navigation_stress=stress, selection_steps=[0, 20, 40, 60, 80],
                arms=list(ARMS), clocks_s=dict(motion=.1, rf=.5, decision=.5, reference=3.,
                                              beacon=1., planning=.2, command_delivery=.1),
                objectives=dict(alert_budget_per_fleet_h=4., max_pdr_loss_pp=.25,
                                max_mission_time_increase=.05, safety_violations=0),
                inputs='Successfully received RF reports, coordinator queue, and current shared navigation.',
                comparison='Same-engine warm 3s and 0.5s, trained 0.5s, and installed Final V1 3s.')
    write(PLAN, plan)
    return plan
