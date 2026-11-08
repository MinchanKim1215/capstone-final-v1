"""Fixed cohorts and release criteria for the four meeting tasks."""
from . import ARTIFACTS, ROOT, VERSION
from forecast_rebuild.contracts import BASELINE, PLAN, digest, folds, maps, now, read, sha, verify_prior, write

V21 = ROOT.parent/"forecast_learning"/"artifacts"
ARMS = ("nominal", "reactive", "v7", "v21", "physics_trigger", "no_short", "no_backup", "ppo_base", "ppo_continuous")
TRANSPORT_ARMS = ("neutral", "local", "local_no_recovery")
KINDS = ("clear", "intermittent", "continuous")
UPDATES = 4
FLIGHTS_PER_UPDATE = 6


def make_spec(region, token, kind):
    code = digest([VERSION, token, region])
    return dict(region=region, kind=kind, seed=int(code[:8], 16) % (2**31-3000010),
                direction=("east_west", "north_south")[int(code[8:12], 16)%2])


def prepare():
    verify_prior()
    cause_path = ARTIFACTS/"causes.json"
    cause = read(cause_path)
    if cause["maps"] != 345 or cause["source_code_sha256"] != sha(ROOT/"causes.py"):
        raise ValueError("Complete and verify the all-map cause audit first")
    v21_freeze = read(V21/"freeze.json")
    if sha(V21/"freeze.json") != read(V21/"completed.json")["freeze_sha256"]:
        raise ValueError("V21 candidate changed")
    old = []
    for package in ("forecast_prevention", "forecast_avoidance", "forecast_shield", "forecast_learning"):
        old.extend(read(ROOT.parent/package/"artifacts"/"protocol.json").get("fresh_specs", []))
    old.extend([item["source_record"]["spec"] for item in
                read(ROOT.parent/"forecast_causal"/"artifacts"/"confirmation_data_manifest.json")["records"]])
    old.extend([dict(region=row["region"], seed=read(row["path"])["spec"]["seed"]) for row in
                read(ROOT.parent/"outage_atlas"/"artifacts"/"manifest.json")["records"]])
    prior_pairs = {(spec["region"], spec["seed"]) for spec in old}
    registry = maps()
    confirm = [make_spec(region, "confirm", kind) for region in folds()["confirmation"] for kind in KINDS]
    if len(confirm) != 42 or any((s["region"], s["seed"]) in prior_pairs or
                                registry[s["region"]]["joint_split"] != "validation" for s in confirm):
        raise ValueError("Confirmation seed or geography changed")
    regions = sorted((key for key, m in registry.items() if m["joint_split"] == "train"),
                     key=lambda key: digest([VERSION, "ppo_maps", key]))
    seen_groups = set()
    chosen = []
    for region in regions:
        group = registry[region]["joint_spatial_group"]
        if group not in seen_groups:
            chosen.append(region)
            seen_groups.add(group)
    if len(chosen) < UPDATES*FLIGHTS_PER_UPDATE:
        raise ValueError("Not enough independent training geography")
    train = [make_spec(region, "ppo", KINDS[index%3]) for index, region in
             enumerate(chosen[:UPDATES*FLIGHTS_PER_UPDATE])]
    # Independent stress conditions with the same 14-map geography, one regime per map.
    stress = [make_spec(region, "stress", KINDS[index%3]) for index, region in enumerate(folds()["confirmation"])]
    if any((s["region"], s["seed"]) in prior_pairs for s in stress):
        raise ValueError("Stress seed overlaps previous experiments")
    pins = [BASELINE, PLAN, cause_path, V21/"freeze.json", V21/"train_manifest.json",
            V21/"calibration_manifest.json", V21/"selection_manifest.json",
            ROOT.parent/"forecast_current"/"active_profile.json",
            ROOT.parent/"forecast_learning"/"control.py",
            ROOT.parent/"forecast_avoidance"/"controller.py"]
    pins.extend(meta["path"] for meta in v21_freeze["models"].values())
    value = dict(version=VERSION, arms=list(ARMS), transport_arms=list(TRANSPORT_ARMS), train=train,
                 confirmation=confirm, stress=stress, updates=UPDATES, flights_per_update=FLIGHTS_PER_UPDATE,
                 source_sha256={name:sha(ROOT/name) for name in
                                ("__init__.py", "causes.py", "protocol.py", "policy.py", "control.py",
                                 "transport.py", "runner.py", "training.py")},
                 dependency_sha256={str(p):sha(p) for p in pins},
                 alert_cap_per_fleet_hour=4., horizon_windows=[[.5, 3.], [3., 5.], [5., 7.], [3., 7.]],
                 promotion=dict(baseline="v21", priority="new persistent outage starts per flight-hour",
                                requires="No safety violations, every mission completed, no more than 0.25pp loss "
                                         "in pooled timely PDR, no more than 2% mission-time increase, "
                                         "paired 95% upper bound below zero versus V21",
                                tied_candidate="v21"),
                 diagnostics_only="The 345-map diagnostic data are segregated from fitting, threshold calibration "
                                  "and confirmation; only 245 original train-map V21 feature files enter supervised fitting.")
    path = ARTIFACTS/"protocol.json"
    if path.exists():
        prior = read(path)
        if {k:v for k,v in prior.items() if k != "created_at"} != value:
            raise ValueError("Frozen protocol, source or dependency changed")
        return prior
    value["created_at"] = now()
    write(path, value)
    return value


def status(stage, **details):
    write(ARTIFACTS/"status.json", dict(version=VERSION, stage=stage, state=details.pop("state", "running"),
                                       updated_at=now(), **details))
