"""Run locked confirmations, command delivery stress tests, and release audit."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np

from adaptive_joint_expanded import ARTIFACTS as NAV_ARTIFACTS, LEGACY
from adaptive_joint_expanded.baseline import ensure_nominal
from adaptive_joint_expanded.contracts import DEPLOYMENT_BY_KEY, lock
from swarm_sim.config import Scenario
from usable.collect import prepare_city_without_unused_radio_cache

from . import ARTIFACTS, ROOT, VERSION
from .causes import events_for
from .evaluation import advisory_case, calibrated_forecast, forecast_model, key, paired, pooled, score_advisory, transport_metrics
from .protocol import ARMS, TRANSPORT_ARMS, V21, maps, now, prepare, read, sha, status, write
from .runner import initialize, run
from .training import read_checkpoint


def preflight():
    initialize()
    name = "incheon_20261030_b2_29_clear"
    row = read(V21/"learned_control"/f"{name}.json")
    trial, data = run(row["spec"], "v21", max_steps=64)
    with np.load(row["archive_path"],allow_pickle=False) as prior:
        for k in ("nodes", "commands", "links", "network"):
            np.testing.assert_array_equal(data[k], prior[k][:len(data[k])])
    if not trial["intervention"] or trial["exact_endpoints"] < 2:
        raise ValueError("Preflight failed to reach both V21 interventions")
    _, transported = run(row["spec"], "v21", transport="local", max_steps=12)
    if not (transported["received"][:,0].all() and not transported["forced"][:,0].any()):
        raise ValueError("Local coordinator incorrectly lost its own command")
    write(ARTIFACTS/"preflight.json",dict(v21_exact=True, checked_steps=64,
                                        interventions=trial["exact_endpoints"], command_channel_causal=True))


def trial(task):
    spec, arm, transport, model, seal = task
    name = f'{spec["region"]}_{spec["kind"]}'
    path = ARTIFACTS/("stress" if transport else "confirm")/(transport or arm)/(name+".json")
    array = path.with_suffix(".npz")
    contract = dict(spec=spec,arm=arm,transport=transport,seal=seal)
    if path.exists():
        row = read(path)
        if row["contract"] != contract or sha(array) != row["archive_sha256"]:
            raise ValueError("Archived simulation changed: "+str(path))
        return row
    state = None
    if model:
        if sha(model["path"]) != model["sha256"]:
            raise ValueError("Controller model changed")
        state = read_checkpoint(model["path"])["state"]
    row, data = run(spec,arm,state_dict=state,transport=transport)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = array.with_suffix(".pending.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(array)
    row.update(contract=contract, archive_path=str(array), archive_sha256=sha(array))
    write(path,row)
    return row


def simulate(stage, tasks):
    records = [None]*len(tasks)
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context("spawn"),initializer=initialize) as pool:
        jobs = {pool.submit(trial, task):i for i,task in enumerate(tasks)}
        for count, job in enumerate(as_completed(jobs),1):
            i = jobs[job]
            try:
                records[i] = job.result()
            except Exception as error:
                for pending in jobs:
                    pending.cancel()
                raise RuntimeError(f"Flight failed: {tasks[i][:3]}") from error
            if count % 6 == 0 or count == len(tasks):
                status(stage, complete=count,total=len(tasks))
                print(stage, count, "/", len(tasks),flush=True)
    return records


def native_patterns(row, registry):
    spec = row["spec"]
    scene = registry[spec["region"]]
    scenario = Scenario(direction=spec["direction"], motion_dt_s=3).validate()
    path = LEGACY/scene["path"]
    if sha(path) != scene["sha256"]:
        raise ValueError("Confirmation city changed")
    city = prepare_city_without_unused_radio_cache(path,scenario,NAV_ARTIFACTS/"cache/city")
    anchor = ensure_nominal(city,DEPLOYMENT_BY_KEY["v8_s3_r5_40"],scenario)
    with np.load(anchor,allow_pickle=False) as nominal:
        duration = (len(nominal["nodes"])-1)*3.
    rng = np.random.default_rng(spec["seed"])
    point = city.target_points[int(rng.integers(len(city.target_points)))]
    start, stop = rng.uniform(.15,.45)*duration, rng.uniform(.65,.9)*duration
    power = rng.uniform(-3.,3.) if spec["kind"] != "continuous" else np.random.default_rng(spec["seed"])
    if spec["kind"] == "continuous":
        power.integers(len(city.target_points))
        power = power.uniform(-3.,3.)
    jammer = dict(jammer_xy_m=point.tolist(), tx_power_dbm=float(power), intermittent_start_s=float(start),
                  intermittent_stop_s=float(stop))
    with np.load(row["archive_path"],allow_pickle=False) as z:
        events = events_for(city,scenario,z["positions"],z["links"],spec,jammer,row["search_drones"])
    return [[e["onset_s"],e["end_s"],e["pattern"]] for e in events]


def exact_branches(baseline, candidate, patterns):
    ref = {key(row):row for row in baseline}
    result = []
    for row in candidate:
        k = key(row)
        old = ref[k]
        with np.load(old["archive_path"],allow_pickle=False) as a, np.load(row["archive_path"],allow_pickle=False) as b:
            ca, cb = a["commands"], b["commands"]
            n = min(len(ca),len(cb))
            changes = np.flatnonzero((ca[:n] != cb[:n]).any(axis=(1,2)))
            if not len(changes):
                continue
            step = int(changes[0])
            frame, t = step*6, step*3.
            possible = [e for e in patterns[k] if 3. <= e[0]-t < 7.]
            if not possible or not frame or frame+24 > len(b["network"]):
                continue
            if not (a["network"][frame-1:int(possible[0][0]*2)].all() and b["network"][frame:frame+24].all()):
                continue
            np.testing.assert_array_equal(a["positions"][:step+1],b["positions"][:step+1])
            np.testing.assert_array_equal(a["links"][:frame],b["links"][:frame])
            result.append(dict(spec=row["spec"], action_time_s=t, baseline_onset_s=possible[0][0],
                               baseline_end_s=possible[0][1], pattern=possible[0][2], connected_after_s=12.))
    return result


def evaluate(plan, training):
    freeze_path = ARTIFACTS/"comparison_freeze.json"
    pins = {name:sha(ROOT/name) for name in ("campaign.py","evaluation.py")}
    inputs = dict(protocol_sha256=sha(ARTIFACTS/"protocol.json"),training_sha256=sha(ARTIFACTS/"training.json"),
                  source_sha256=pins)
    if freeze_path.exists():
        frozen = read(freeze_path)
        if frozen["pins"] != inputs:
            raise ValueError("Confirmation plan or implementation changed")
    else:
        status("3_calibrate_equal_alert_budget")
        calibration = calibrated_forecast(training,plan["alert_cap_per_fleet_hour"])
        frozen = dict(pins=inputs, thresholds=calibration, created_at=now())
        write(freeze_path,frozen)
    seal = sha(freeze_path)
    status("3_paired_control", complete=0,total=len(plan["confirmation"])*len(ARMS))
    rows = simulate("3_paired_control",[(spec,arm,None,training["models"].get(arm),seal)
                                      for spec in plan["confirmation"] for arm in ARMS])
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    if any(len(v)!=len(plan["confirmation"]) for v in grouped.values()):
        raise ValueError("Paired control cases missing")
    status("3_same_motion_and_event_patterns")
    model = forecast_model(training)
    registry = maps()
    patterns = {key(row):native_patterns(row,registry) for row in grouped["v7"]}
    fixed = []
    for row in grouped["v7"]:
        with np.load(row["archive_path"],allow_pickle=False) as z:
            x = z["forecast_features"]
            fixed.append(advisory_case(row,np.arange(len(x))*3.,x,z["network"],model,patterns[key(row)]))
    forecasts = {variant:score_advisory(fixed,variant,info["threshold"],True)
                 for variant,info in frozen["thresholds"].items()}
    stats = {arm:pooled(flights) for arm,flights in grouped.items()}
    vs_v21 = {arm:paired(grouped["v21"],flights) for arm,flights in grouped.items() if arm != "v21"}
    vs_base = paired(grouped["ppo_base"],grouped["ppo_continuous"])
    options = []
    for arm in ("physics_trigger","no_short","no_backup","ppo_continuous"):
        change = vs_v21[arm]
        if (stats[arm]["safety"] == 0 and stats[arm]["completed"] == len(plan["confirmation"])
                and change["timely_pdr_pp"] >= -.25 and change["duration_fraction"] <= .02 and change["ci95"][1] < 0.):
            options.append(arm)
    candidate = min(options, key=lambda a: stats[a]["starts_per_h"]) if options else "v21"
    by_type = {}
    for arm in dict.fromkeys(("v7","v21",candidate)):
        groups = patterns if arm == "v7" else {key(r):native_patterns(r,registry) for r in grouped[arm]}
        counts = defaultdict(int)
        for cuts in groups.values():
            for start, _, shape in cuts:
                if start > 0.:
                    counts[shape] += 1
        by_type[arm] = dict(counts)
    result = dict(stats=stats, vs_v21=vs_v21, continuous_vs_matched_ppo=vs_base,
                  alerts=forecasts, pattern_counts=by_type, candidate=candidate, eligible_candidates=options,
                  same_history_examples=exact_branches(grouped["v7"],grouped[candidate],patterns),
                  manifest=[dict(path=str(ARTIFACTS/"confirm"/r["arm"]/
                                          f'{r["spec"]["region"]}_{r["spec"]["kind"]}.json'),
                                 archive_sha256=r["archive_sha256"]) for r in rows], freeze_sha256=seal)
    write(ARTIFACTS/"comparison.json",result)
    return result, grouped


def stress(plan, training, evaluation):
    seal = sha(ARTIFACTS/"comparison.json")
    candidate = evaluation["candidate"]
    status("4_command_delivery_and_recovery", complete=0,total=len(plan["stress"])*len(TRANSPORT_ARMS))
    rows = simulate("4_command_delivery_and_recovery",[
        (spec,candidate,mode,training["models"].get(candidate),seal)
        for spec in plan["stress"] for mode in TRANSPORT_ARMS])
    grouped = {mode:[r for r in rows if r["transport"] == mode] for mode in TRANSPORT_ARMS}
    result = dict(control={mode:pooled(rows) for mode,rows in grouped.items()},
                  transport={mode:transport_metrics(rows) for mode,rows in grouped.items()},
                  fallback_delta=paired(grouped["neutral"],grouped["local"]),
                  recovery_delta=paired(grouped["local_no_recovery"],grouped["local"]),
                  cases=len(plan["stress"]), decision_sha256=seal)
    write(ARTIFACTS/"transport.json",result)
    return result,grouped


def main():
    plan = prepare()
    training = read(ARTIFACTS/"training.json")
    for model in training["models"].values():
        if sha(model["path"]) != model["sha256"]:
            raise ValueError("Frozen trained model changed")
    if not (ARTIFACTS/"preflight.json").exists():
        status("3_preflight")
        preflight()
    result, rows = evaluate(plan,training)
    transport, stress_rows = stress(plan,training,result)
    status("4_replay_and_release_audit")
    from .delivery import finish
    finish(plan,training,result,rows,transport,stress_rows)


if __name__ == "__main__":
    with lock(ARTIFACTS/"campaign.lock"):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            main()
        except BaseException as error:
            status("campaign",state="failed",error=repr(error))
            traceback.print_exc()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
