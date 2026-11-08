"""Freeze the validation question, geography and data boundaries."""
from . import ROOT, ARTIFACTS, VERSION
from forecast_rebuild.contracts import BASELINE, PLAN, digest, folds, maps, now, read, sha, verify_prior, write

V21 = ROOT.parent/"forecast_learning"/"artifacts"
V19 = ROOT.parent/"outage_atlas"/"artifacts"
FINAL = ROOT.parent/"final_v1"/"artifacts"
KINDS = ("clear","intermittent","continuous")
CAP = 4.


def prepare():
    verify_prior()
    if sha(FINAL/"release.json") != read(FINAL/"completed.json")["release_sha256"]:
        raise ValueError("Final V1 release changed")
    previous = []
    for package in ("forecast_prevention","forecast_avoidance","forecast_shield","forecast_learning"):
        previous.extend(read(ROOT.parent/package/"artifacts"/"protocol.json").get("fresh_specs",[]))
    final_plan = read(FINAL/"protocol.json")
    previous.extend(final_plan["confirmation"]+final_plan["stress"]+final_plan["train"])
    previous.extend([r["source_record"]["spec"] for r in
                     read(ROOT.parent/"forecast_causal"/"artifacts"/"confirmation_data_manifest.json")["records"]])
    previous.extend(read(r["path"])["spec"] for r in read(V19/"manifest.json")["records"])
    used = {(r["region"],r["seed"]) for r in previous}
    registry = maps()
    fresh = []
    for region in folds()["confirmation"]:
        token = digest([VERSION,region,20261105])
        seed = int(token[:8],16)%(2**31-3000010)
        if (region,seed) in used or registry[region]["joint_split"] != "validation":
            raise ValueError("Repeated confirmation conditions or forbidden map")
        direction = ("east_west","north_south")[int(token[8:12],16)%2]
        fresh.extend(dict(region=region,kind=kind,seed=seed,direction=direction) for kind in KINDS)
    sources = {scope:read(V21/f"{scope}_manifest.json")["records"] for scope in ("train","calibration","selection")}
    if len(sources["train"]) != 245 or len(sources["calibration"]) != 45 or len(sources["selection"]) != 45:
        raise ValueError("Original development split changed")
    v21_plan = read(V21/"protocol.json")
    train_sources = {r["region"]:r for r in read(V19/"manifest.json")["records"] if r["split"]=="train"}
    inputs = [BASELINE,PLAN,FINAL/"release.json",FINAL/"completed.json",V21/"freeze.json",
              *[V21/f"{scope}_manifest.json" for scope in sources]]
    value = dict(version=VERSION,sources=sources,original_train=train_sources,
                 original_development={scope:v21_plan["development"][scope] for scope in ("calibration","selection")},
                 fresh=fresh,maximum_false_or_late_per_fleet_hour=CAP,
                 definitions=dict(primary="Event recall at 3 <= lead < 7 seconds on identical V7 trajectories.",
                                  secondary="Imminent 0.5 <= lead < 3 seconds, never reported as 3-second advance detection.",
                                  cadence_s=3.,observed_lags_s=[.5,1.,2.],
                                  feature_source="First relay actually received beacon/ACK telemetry and explicit missingness; held observations require distinct measurement timestamps for differences.",
                                  diagnostic="Post-onset, underlying SINR and cut-type labels are diagnostic only.",
                                  training="Three identical LightGBM early models: existing physics features, observed differences only, and their concatenation. Additional observed-only imminent head.",
                                  choice="Same train rows and parameters. Select rounds on selection under 4/h, thresholds from calibration. Freeze before confirmation.",
                                  acceptance="Added differences improve early recall with positive paired 95% lower bound, precision decrease no more than 2pp, false-or-late <=4/h."),
                 data_sources={name:sha(ROOT/name) for name in
                               ("__init__.py","features.py","protocol.py","replay.py","campaign.py")},
                 dependencies={str(p):sha(p) for p in inputs})
    path = ARTIFACTS/"protocol.json"
    if path.exists():
        old = read(path)
        if {k:v for k,v in old.items() if k!="created_at"} != value:
            raise ValueError("Pinned data or protocol changed")
        return old
    value["created_at"] = now()
    write(path,value)
    return value


def status(stage,**info):
    write(ARTIFACTS/"status.json",dict(version=VERSION,stage=stage,state=info.pop("state","running"),
                                    updated_at=now(),**info))
