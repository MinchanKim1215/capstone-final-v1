"""Pin the maps, diagnostic exposure and assignment before any full flight."""
from . import ARTIFACTS, ROOT
from forecast_rebuild.contracts import BASELINE, PLAN, digest, maps, now, read, sha, verify_prior, write

KINDS = ('clear', 'intermittent', 'continuous')
VERSION = 'v19-all-map-outage-attribution'
SEED = 2026110419
SOURCES = ('__init__.py', 'protocol.py', 'runner.py', 'radio.py', 'events.py', 'campaign.py')
LEGACY = ROOT.parent / 'drone_forecast_joint_training'
V7 = ROOT.parent / 'adaptive_joint_expanded'


def prepare():
    verify_prior()
    registry = maps()
    counts = {role: sum(m['joint_split']==role for m in registry.values()) for role in
              ('train', 'validation', 'historical_validation', 'test', 'quarantine')}
    if list(counts.values()) != [245, 44, 23, 32, 1] or len(registry) != 345:
        raise ValueError('The expected full registry changed')
    cases = []
    ordered = sorted(registry, key=lambda region: digest([VERSION, SEED, region]))
    for i, region in enumerate(ordered):
        entry = registry[region]
        if sha(LEGACY / entry['path']) != entry['sha256']:
            raise ValueError('Map digest mismatch: ' + region)
        token = digest([SEED, VERSION, region])
        spec = dict(region=region, kind=KINDS[i % len(KINDS)],
                    seed=int(token[:8], 16) % (2**31 - 3000010),
                    direction=('east_west', 'north_south')[i % 2])
        cases.append(dict(spec=spec, split=entry['joint_split'], map_sha256=entry['sha256'],
                          spatial_group=entry['joint_spatial_group']))
    dependency_paths = [PLAN, BASELINE, V7/'artifacts/protocol.json',
                        ROOT.parent/'forecast_prevention/diagnostics.py',
                        ROOT.parent/'forecast_causal/recorder.py',
                        ROOT.parent/'forecast_causal/instrumented_packet.py',
                        ROOT.parent/'forecast_current/active_profile.json',
                        ROOT.parent/'forecast_avoidance/artifacts/completed.json']
    plan = dict(version=VERSION, counts=counts, cases=cases,
                source_sha256={name: sha(ROOT/name) for name in SOURCES},
                dependencies={str(path): sha(path) for path in dependency_paths},
                coverage='One new full frozen V7 closed-loop flight per map, independently of the former split. All three RF regimes replayed on every resulting trajectory.',
                matched='Same path, packet seeds, jammer coordinates and power, and shadow realizations within each map. Intermittent changes only activation times.',
                failures='Every physical network cut of at least two successive 0.5s frames and every service violation of at least two frames. Initial cuts are retained and labeled.',
                attribution='Full-interval combinations of building excess loss, external interference, and correlated shadow loss. At new onsets additionally freeze previous building, free-space, jammer and shadow terms to investigate the transition.',
                holdout='32 former test and 1 geographic-quarantine maps are explicitly exposed for diagnosis. Never feed their outcomes to model fit, threshold selection, or an evaluation advertised as blind.',
                flight_policy='Frozen V7 placement selector, PPO relay actions, unchanged motion and packet engines. One direction and one seed per map.')
    path = ARTIFACTS / 'protocol.json'
    if path.exists():
        old = read(path)
        if {key: value for key, value in old.items() if key != 'created_at'} != plan:
            raise ValueError('Pinned atlas protocol or implementation changed')
        return old
    plan['created_at'] = now()
    write(path, plan)
    return plan


def status(stage, **details):
    write(ARTIFACTS/'status.json', dict(stage=stage, updated_at=now(), state=details.pop('state', 'running'), **details))
