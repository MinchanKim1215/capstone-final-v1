"""Immutable study boundaries and a resumable progress record."""
import os

from . import ARTIFACTS, ROOT, WORKSPACE
from forecast_rebuild.contracts import BASELINE, PLAN, digest, folds, maps, now, read, sha, write

os.environ['NUMBA_CACHE_DIR'] = str(ARTIFACTS / 'numba_cache')
V16 = WORKSPACE / 'forecast_causal' / 'artifacts'
OLD = WORKSPACE / 'adaptive_joint_expanded' / 'artifacts'
KINDS = ('clear', 'intermittent', 'continuous')
SEED = 2026110417
ARM_NAMES = ('baseline', 'reactive', 'geometry', 'switch')


def new_spec(region, kind, scope):
    token = digest(['v17-prevention', SEED, scope, region])
    return dict(region=region, kind=kind,
                seed=int(token[:8], 16) % (2**31 - 3000010),
                direction=('east_west', 'north_south')[int(token[8:12], 16) % 2])


def prepare():
    registry = maps()
    data = read(V16 / 'data_manifest.json')['records']
    data += read(V16 / 'confirmation_data_manifest.json')['records']
    eligible = sorted(k for k, m in registry.items()
                      if m['joint_split'] in ('train', 'validation', 'historical_validation'))
    historical = [k for k in eligible if registry[k]['joint_split'] == 'historical_validation']
    selected = []
    for i, region in enumerate(sorted(folds()['selection'])):
        kind = KINDS[i % len(KINDS)]
        selected.append(next(x['source_record'] for x in data if x['region'] == region
                             and x['role'] == 'selection' and x['source_record']['spec']['kind'] == kind))
    parent_paths = [PLAN, BASELINE, OLD / 'protocol.json', V16 / 'delivery_manifest.json',
                    V16 / 'data_manifest.json', V16 / 'confirmation_data_manifest.json',
                    WORKSPACE / 'forecast_current' / 'active_profile.json']
    plan = dict(version='v17-action-conditioned-prevention', created_at=now(),
                dependencies={str(p): sha(p) for p in parent_paths},
                eligible_regions=eligible, archived_records=data,
                historical_specs=[new_spec(region, KINDS[i % 3], 'historical')
                                  for i, region in enumerate(historical)],
                selection_sources=selected, selection_arms=list(ARM_NAMES[1:]),
                fresh_specs=[new_spec(region, kind, 'fresh-confirmation')
                             for region in sorted(folds()['confirmation']) for kind in KINDS],
                primary='Persistent search-network cut onsets >=1s per flight-hour; primary outcome is prevention.',
                secondary=['cut_seconds_per_hour', 'service_onsets', 'timely_pdr', 'mission_completion',
                           'relay_travel', 'coverage', 'observed_preemption', 'shared_prefix_examples'],
                interventions='Only the first relay may override its frozen V7 command. Same V7 fleet selector, motion and radio.',
                source_contract='Local measured RF and only actually received navigation beacons. Cause truth is offline.',
                pairs='Same case seed, fleet, direction, external interference and shadow. Packet and beacon draws follow actions.',
                selection_rule='Compare geometry and switch on15 selection flights. Prefer feasible lower persistent-onset rate, then less outage time, then less relay travel.',
                decision_guards=dict(minimum_mission_completions_relative_to_baseline=0,
                                     max_total_duration_ratio=1.05, max_relay_travel_ratio=1.10,
                                     minimum_timely_pdr_delta=-.005, zero_safety_violations=True),
                final_test_opened=False, excluded_final=32, quarantined=1)
    covered = {x['region'] for x in data} | set(historical)
    if (len(eligible), len(covered), len(data), len(historical), len(selected)) != (312, 312, 377, 23, 15):
        raise ValueError('Map coverage or cohort changed')
    path = ARTIFACTS / 'protocol.json'
    if path.exists():
        previous = read(path)
        if {k: v for k, v in previous.items() if k != 'created_at'} != {k: v for k, v in plan.items() if k != 'created_at'}:
            raise ValueError('Study protocol changed')
        return previous
    write(path, plan)
    return plan


def status(stage, **details):
    value = dict(stage=stage, updated_at=now(), state=details.pop('state', 'running'), **details)
    write(ARTIFACTS / 'status.json', value)


def verify_freeze():
    frozen = read(ARTIFACTS / 'freeze.json')
    if sha(ARTIFACTS / 'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Protocol changed after freeze')
    for name in ('selection', 'cause_summary'):
        if sha(ARTIFACTS / (name + '.json')) != frozen[name + '_sha256']:
            raise ValueError('Frozen study evidence changed: ' + name)
    for name, expected in frozen['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen prevention code changed: ' + name)
    for path, expected in read(ARTIFACTS / 'protocol.json')['dependencies'].items():
        if sha(path) != expected:
            raise ValueError('Frozen predecessor changed: ' + path)
    return frozen

