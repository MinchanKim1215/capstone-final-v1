"""Predeclare study splits, comparison gates and evidence dependencies."""
import os

from . import ARTIFACTS, ROOT
from forecast_rebuild.contracts import BASELINE, digest, folds, maps, now, read, sha, verify_prior, write

V17 = ROOT.parent / 'forecast_prevention' / 'artifacts'
CANDIDATES = ('joint_geometry', 'joint_radio')
KINDS = ('clear', 'intermittent', 'continuous')
SEED = 2026110418
SOURCES = ('__init__.py', 'protocol.py', 'controller.py', 'runner.py', 'campaign.py', 'evaluation.py')
os.environ['NUMBA_CACHE_DIR'] = str(ARTIFACTS / 'numba_cache')


def prepare():
    from forecast_prevention.protocol import verify_freeze
    verify_prior()
    verify_freeze()
    previous = read(V17 / 'protocol.json')
    selection = read(V17 / 'selection.json')
    registry = maps()
    regions = sorted(folds()['confirmation'])
    fresh = []
    for region in regions:
        token = digest([SEED, region, 'confirmed-independent-seed'])
        seed = int(token[:8], 16) % (2**31 - 3000010)
        direction = ('east_west', 'north_south')[int(token[8:12], 16) % 2]
        fresh.extend(dict(region=region, kind=kind, seed=seed, direction=direction) for kind in KINDS)
    if len(fresh) != 42 or not all(registry[r]['joint_split'] == 'validation' for r in regions):
        raise ValueError('Unexpected confirmation cohort')
    older = {(r['region'], r['seed']) for r in previous['fresh_specs']}
    older |= {(r['region'], r['source_record']['spec']['seed']) for r in previous['archived_records']}
    if any((r['region'], r['seed']) in older for r in fresh):
        raise ValueError('Prior seed appeared in confirmation')
    files = [BASELINE, V17 / 'protocol.json', V17 / 'selection.json', V17 / 'freeze.json',
             V17 / 'cause_summary.json', V17 / 'survey_manifest.json', V17 / 'completed.json',
             ROOT.parent / 'forecast_navigation' / 'preview.py',
             ROOT.parent / 'forecast_navigation' / 'wire.py',
             ROOT.parent / 'forecast_current' / 'active_profile.json']
    plan = dict(version='v18-coordinated-outage-avoidance', dependencies={str(p): sha(p) for p in files},
                selection_specs=[source['spec'] for source in previous['selection_sources']],
                development_reference=selection['baseline'], fresh_specs=fresh,
                primary='New search-network cut runs of >=1s per flight-hour.',
                guard=previous['decision_guards'], candidates=list(CANDIDATES),
                selection_rule='Feasible, then fewer onsets/h, then less cut seconds/h, then relay travel.',
                adoption_rule='Lower paired 95% confidence bounds are insufficient; upper bound below zero against V7 and V17, with feasibility and favorable development direction.',
                input='Shared planner current navigation state, its provisional 3s segments, received beacons and RF. Later segments assume held actions. Hidden RF and jammer schedule are outcome-only.',
                candidates_description='Change at most one relay proposal from V7; any relay is eligible. Common movement resolver may change other endpoints.',
                final_test_opened=False, alarm_min_lead_s=3., alarm_max_lead_s=7.)
    path = ARTIFACTS / 'protocol.json'
    if path.exists():
        old = read(path)
        if {key: value for key, value in old.items() if key != 'created_at'} != plan:
            raise ValueError('Saved study boundaries changed')
        return old
    plan['created_at'] = now()
    write(path, plan)
    return plan


def status(stage, **details):
    write(ARTIFACTS / 'status.json', dict(stage=stage, updated_at=now(),
                                          state=details.pop('state', 'running'), **details))
