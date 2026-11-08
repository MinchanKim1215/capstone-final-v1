"""Extract received RF differences, fit matched predictors, and verify fresh flights."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import ctypes
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np

from adaptive_joint_expanded.contracts import lock
from forecast_learning.training import load

from . import ROOT,ARTIFACTS
from .protocol import V21,prepare,read,sha,status,write
from .replay import initialize,run


def source_task(plan,scope,record):
    meta = read(record["path"])
    spec = meta["spec"]
    if scope == "train":
        source = read(plan["original_train"][spec["region"]]["path"])
        return dict(scope=scope,record=record,spec=spec,commands=source["archive_path"],
                    commands_sha256=source["archive_sha256"],link_key="actual_links")
    sources = plan["original_development"][scope]
    source = next(s for s in sources if s["spec"]==spec)
    return dict(scope=scope,record=record,spec=spec,commands=source["array_path"],
                commands_sha256=source["array_sha256"],link_key="links")


def trial(task):
    scope,spec = task["scope"],task["spec"]
    path = ARTIFACTS/"data"/scope/f'{spec["region"]}_{spec["kind"]}.json'
    array = path.with_suffix(".npz")
    contract = dict(task=task,protocol_sha256=sha(ARTIFACTS/"protocol.json"))
    if path.exists():
        row = read(path)
        if row["contract"] != contract or sha(array) != row["archive_sha256"]:
            raise ValueError("Input feature cache changed")
        return dict(path=str(path),sha256=sha(path),region=spec["region"])
    if scope != "fresh":
        f = load(task["record"])
        if sha(task["commands"]) != task["commands_sha256"]:
            raise ValueError("Archived original command stream changed")
        with np.load(task["commands"],allow_pickle=False) as original:
            row,data = run(spec,original["commands"],(f.x,f.raw))
            for name in ("nodes","commands","links"):
                np.testing.assert_array_equal(data[name],original[task["link_key"] if name=="links" else name])
        with np.load(f.row["archive_path"],allow_pickle=False) as original:
            np.testing.assert_array_equal(data["network"],original["network"])
        for name in ("generated_packets","delivered_packets","timely_pdr","retries","beacon_rx"):
            if row[name] != f.row[name]:
                raise ValueError("Replay changed packets or beacon reception: "+name)
        row["original_trajectory_and_packets_exact"] = True
        data = {name:data[name] for name in ("network","x","rf","times","event_aligned","onsets","physical_scores")}
    else:
        row,data = run(spec)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = array.with_suffix(".pending.npz")
    np.savez_compressed(pending,**data)
    pending.replace(array)
    row.update(contract=contract,archive_path=str(array),archive_sha256=sha(array))
    write(path,row)
    return dict(path=str(path),sha256=sha(path),region=spec["region"])


def collect(stage,tasks):
    path = ARTIFACTS/f"{stage}_manifest.json"
    if path.exists():
        value = read(path)
        if value["protocol_sha256"] != sha(ARTIFACTS/"protocol.json"):
            raise ValueError("Cached manifest has another protocol")
        return value["records"]
    rows = [None]*len(tasks)
    status(stage,complete=0,total=len(tasks))
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context("spawn"),initializer=initialize) as pool:
        jobs = {pool.submit(trial,task):i for i,task in enumerate(tasks)}
        for count,future in enumerate(as_completed(jobs),1):
            rows[jobs[future]] = future.result()
            if count % 12 == 0 or count == len(tasks):
                status(stage,complete=count,total=len(tasks))
                print(stage,count,"/",len(tasks),flush=True)
    write(path,dict(records=rows,protocol_sha256=sha(ARTIFACTS/"protocol.json")))
    return rows


def preflight(plan):
    initialize()
    reference = read(V21/"fresh"/"incheon_20261030_b2_29_clear.json")
    # Use the original V21 fresh mission as an untouched regression fixture.
    with np.load(reference["archive_path"],allow_pickle=False) as prior:
        row,data = run(reference["spec"],max_steps=24)
        for name in ("nodes","commands","links"):
            np.testing.assert_array_equal(data[name],prior[name][:len(data[name])])
        keep = prior["scores"][:,0] < row["duration_s"]
        np.testing.assert_allclose(data["x"],prior["x"][keep],equal_nan=True,rtol=0,atol=0)
    if not row["future_prefix_invariant"] or not row["masked_value_invariant"]:
        raise ValueError("Input causality regression failed")
    write(ARTIFACTS/"preflight.json",dict(prior_commands_positions_links_and_features_exact=True,
                                        future_prefix_invariant=True,masked_value_invariant=True))


def main():
    plan = prepare()
    if (ARTIFACTS/"completed.json").exists():
        final = read(ARTIFACTS/"completed.json")
        if sha(ROOT/"RESULTS.md") != final["report_sha256"]:
            raise ValueError("Completed report changed")
        return
    if not (ARTIFACTS/"preflight.json").exists():
        preflight(plan)
    development = [source_task(plan,scope,record) for scope,records in plan["sources"].items() for record in records]
    extracted = collect("development",development)
    from .training import fit
    freeze = fit(extracted)
    if freeze["development_manifest_sha256"] != sha(ARTIFACTS/"development_manifest.json"):
        raise ValueError("Learning cohort changed")
    access = ARTIFACTS/"fresh_access.json"
    seal = sha(ARTIFACTS/"freeze.json")
    if access.exists():
        if read(access)["freeze_sha256"] != seal:
            raise ValueError("Model changed after confirmation opened")
    else:
        write(access,dict(freeze_sha256=seal))
    fresh = collect("fresh",[dict(scope="fresh",spec=spec) for spec in plan["fresh"]])
    from .report import finish
    status("fresh_prediction_and_pre_signal_analysis")
    finish(plan,freeze,fresh)


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
