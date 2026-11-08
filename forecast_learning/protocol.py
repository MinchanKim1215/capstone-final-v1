"""Fix the training and paired-confirmation cohorts before extraction."""
from . import ARTIFACTS, ROOT
from forecast_rebuild.contracts import BASELINE, PLAN, digest, folds, maps, now, read, sha, verify_prior, write

V20 = ROOT.parent/'forecast_shield'
V19 = ROOT.parent/'outage_atlas'/'artifacts'
V18 = ROOT.parent/'forecast_avoidance'
V17 = ROOT.parent/'forecast_prevention'
V16 = ROOT.parent/'forecast_causal'/'artifacts'
KINDS = ('clear','intermittent','continuous')
VERSION = 'v21-supervised-shared-path'
CAP = 4.
V20_THRESHOLD = .60
DATA_SOURCES = ('__init__.py','protocol.py','paths.py','features.py','runner.py','campaign.py')


def prepare():
    verify_prior()
    maps_by_key = maps()
    roles = folds()
    index = read(V19/'manifest.json')
    if sha(V19/'manifest.json')!=read(V19/'completed.json')['manifest_sha256']:
        raise ValueError('V19 training sources changed')
    v19 = {row['region']:row for row in index['records']}
    archive = read(V16/'data_manifest.json')['records']
    train = [row for row in index['records'] if row['split']=='train']
    dev = {k:[x['source_record'] for x in archive if x['role']==k] for k in ('calibration','selection')}
    if len(train)!=245 or any(len(rows)!=45 for rows in dev.values()):
        raise ValueError('Unexpected source population')
    for key,rows in dev.items():
        if {s['spec']['region'] for s in rows}!=set(roles[key]):
            raise ValueError('Development fold changed')
    fresh = []
    past = [read(pkg/'artifacts'/'protocol.json')['fresh_specs'] for pkg in (V17,V18,V20)]
    past.append([x['source_record']['spec'] for x in read(V16/'confirmation_data_manifest.json')['records']])
    for region in roles['confirmation']:
        token = digest([VERSION,region,2026110521])
        seed = int(token[:8],16)%(2**31-3000010)
        previous = {s['seed'] for suite in past for s in suite if s['region']==region}
        previous.add(read(v19[region]['path'])['spec']['seed'])
        if seed in previous or maps_by_key[region]['joint_split']!='validation':
            raise ValueError('Known seed or diagnostic-only geography in confirmation')
        direction = ('east_west','north_south')[int(token[8:12],16)%2]
        fresh.extend(dict(region=region,kind=kind,seed=seed,direction=direction) for kind in KINDS)
    files = [PLAN,BASELINE,V19/'completed.json',V19/'diagnostic_access.json',V19/'manifest.json',
             V16/'data_manifest.json',V16/'confirmation_data_manifest.json',V20/'artifacts'/'completed.json',
             V20/'forecaster.py',V20/'runner.py',V20/'evaluate.py',V18/'controller.py',V18/'runner.py',
             V18/'artifacts'/'completed.json',ROOT.parent/'forecast_navigation'/'preview.py',
             ROOT.parent/'forecast_current'/'active_profile.json']
    plan = dict(version=VERSION,train=train,development=dev,fresh_specs=fresh,
                source_sha256={name:sha(ROOT/name) for name in DATA_SOURCES},
                dependencies={str(p):sha(p) for p in files},
                decision_guards=read(V17/'artifacts'/'protocol.json')['decision_guards'],
                labels='Whole search network; new >=1s cut; healthy at query; binary 0.5–<3s and 3–<7s heads; uncensored training only.',
                learned_selection='Threshold from calibration under 4 false-or-late / fleet-hour. Selection recall, precision, lower false-or-late. Freeze all model and controller choices before confirmation.',
                primary='Same-motion early event recall at 3–<7s. Separate paired actual prevention vs V7 and V18.',
                scope='245 train maps, 15 calibration and 15 selection maps with 3 conditions each, 14 confirmation maps with 3 fresh conditions. No diagnostic-only prior final or quarantined geography.',
                maximum_false_or_late_per_fleet_hour=CAP)
    path = ARTIFACTS/'protocol.json'
    if path.exists():
        old = read(path)
        if {k:v for k,v in old.items() if k!='created_at'}!=plan:
            raise ValueError('Pinned inputs or collection source changed')
        return old
    plan['created_at'] = now()
    write(path,plan)
    return plan


def status(stage,**details):
    write(ARTIFACTS/'status.json',dict(stage=stage,state=details.pop('state','running'),updated_at=now(),**details))
