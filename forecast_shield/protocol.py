"""Keep calibration, algorithm selection and new paired confirmation separate."""
from . import ARTIFACTS, ROOT
from forecast_rebuild.contracts import BASELINE, PLAN, digest, folds, maps, now, read, sha, verify_prior, write

V18 = ROOT.parent/'forecast_avoidance'
V19 = ROOT.parent/'outage_atlas'/'artifacts'
KINDS = ('clear', 'intermittent', 'continuous')
VERSION = 'v20-shared-plan-building-warning'
SEED = 2026110420
SOURCES = ('__init__.py', 'forecaster.py', 'runner.py', 'protocol.py', 'evaluate.py', 'campaign.py')
VARIANTS = ('planner_3s', 'shield_3s', 'shield_1s')
CAP = 4.


def prepare():
    verify_prior()
    old_plan = read(V18/'artifacts'/'protocol.json')
    done = read(V19/'completed.json')
    manifest = V19/'manifest.json'
    if sha(manifest) != done['manifest_sha256']:
        raise ValueError('Completed V19 records changed')
    archive = {r['region']:r for r in read(manifest)['records']}
    registry = maps()
    partition = folds()
    development = {split:[archive[k] for k in partition[split]] for split in ('calibration','selection')}
    for references in development.values():
        if not all(r['split']=='validation' for r in references):
            raise ValueError('Diagnostic-only geography in development')
    fresh = []
    for region in partition['confirmation']:
        if registry[region]['joint_split'] != 'validation':
            raise ValueError('Diagnostic-only geography in confirmation')
        token = digest([VERSION, SEED, region])
        seed = int(token[:8],16) % (2**31-3000010)
        prior = [r['seed'] for r in old_plan['fresh_specs'] if r['region']==region]
        prior.append(read(archive[region]['path'])['spec']['seed'])
        if seed in prior:
            raise ValueError('New confirmation seed reused')
        direction = ('east_west','north_south')[int(token[8:12],16) % 2]
        fresh.extend(dict(region=region, kind=kind, direction=direction, seed=seed) for kind in KINDS)
    if [len(development['calibration']),len(development['selection']),len(fresh)] != [15,15,42]:
        raise ValueError('Evaluation cohorts changed')
    files = [BASELINE, PLAN, V19/'completed.json', V19/'diagnostic_access.json', manifest,
             V18/'artifacts'/'completed.json', V18/'artifacts'/'freeze.json', V18/'controller.py',
             ROOT.parent/'forecast_navigation'/'preview.py', ROOT.parent/'forecast_navigation'/'wire.py',
             ROOT.parent/'forecast_prevention'/'controller.py', ROOT.parent/'forecast_current'/'active_profile.json']
    plan = dict(version=VERSION, development=development, fresh_specs=fresh, variants=list(VARIANTS),
                maximum_false_or_late_per_fleet_hour=CAP,
                prior_cost_profile='Calibrate another threshold at or below V18 calibration false-or-late rate.',
                intervals=dict(forecast_tick_s=1., controller_tick_s=3., radio_step_s=.5, min_lead_s=3., max_lead_exclusive_s=7.),
                definition='First alarm per contiguous eligible interval, matched one-to-one to new persistent network cuts.',
                selection='Select the feasible highest recall variant under fixed calibration threshold, then precision, then lower cost.',
                boundaries='V19 exposure maps (32 old test + 1 quarantine) excluded. Fresh seeds on known geography. Frozen V7 moves, V18 reference unchanged.',
                inputs='Shared current navigation plan and only RF reception up to the timestamp. Unknown or stale RF broadens stress. No future RF or actual future motion.',
                dependencies={str(p):sha(p) for p in files}, sources={name:sha(ROOT/name) for name in SOURCES})
    path = ARTIFACTS/'protocol.json'
    if path.exists():
        saved = read(path)
        if {k:v for k,v in saved.items() if k!='created_at'} != plan:
            raise ValueError('Saved study protocol or source changed')
        return saved
    plan['created_at'] = now()
    write(path,plan)
    return plan


def status(stage, **details):
    write(ARTIFACTS/'status.json',dict(stage=stage,state=details.pop('state','running'),updated_at=now(),**details))
