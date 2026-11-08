"""Verify the release against all archived link traces and the browser QA."""
import argparse
from pathlib import Path

import numpy as np

from forecast_prevention.diagnostics import connected, outage_runs

from . import ARTIFACTS, ROOT
from .protocol import now, read, sha, status, write


def audit_flight(path):
    row = read(path)
    archive = path.with_suffix(".npz")
    if sha(archive) != row["archive_sha256"] or not row["audited"]["passed"]:
        raise ValueError("Flight archive or safety audit changed: "+str(path))
    with np.load(archive,allow_pickle=False) as data:
        network = connected(data["links"],row["search_drones"])
        np.testing.assert_array_equal(network,data["network"])
        cuts = [[a*.5,b*.5] for a,b in outage_runs(network)]
        if cuts != row["persistent_cut_intervals"]:
            raise ValueError("Outage ledger differs from recorded links")
        if "received" in data:
            if not data["received"][:,0].all() or (data["forced"] & data["received"]).any():
                raise ValueError("Command delivery contradicts simulated receipt")
            if ((data["hops"] < 0) != ~data["received"]).any():
                raise ValueError("Command receipt and routing log differ")
    return sha(path)


def verify(browser_report):
    release_path = ARTIFACTS/"release.json"
    release = read(release_path)
    browser = read(browser_report)
    if (browser["cases"] != release["replay_cases"] or browser["errors"] or not browser["valid"]
            or browser["stressCase"] < 0 or browser["interventionCase"] < 0):
        raise ValueError("Offline replay verification missing")
    for file, expected in release["evidence_sha256"].items():
        if sha(file) != expected:
            raise ValueError("Release evidence changed: "+file)
    if sha(ROOT/"RESULTS.md") != release["report_sha256"] or sha(release["viewer"]["path"]) != release["viewer"]["sha256"]:
        raise ValueError("Report or replay digest changed")
    ledgers = {}
    counts = {}
    for stage, field in (("confirm","confirmation_flights"),("stress","stress_flights")):
        paths = sorted((ARTIFACTS/stage).glob("*/*.json"))
        if len(paths) != release[field]:
            raise ValueError("Incorrect paired flight coverage")
        ledgers[stage] = {str(path.relative_to(ROOT)):audit_flight(path) for path in paths}
        counts[field] = len(paths)
    counts["replay_cases"] = browser["cases"]
    complete = dict(release_sha256=sha(release_path), browser_report_sha256=sha(browser_report),
                    counts=counts, audit=ledgers, completed_at=now())
    write(ARTIFACTS/"completed.json",complete)
    status("all_four_stages_verified",state="completed",**counts)
    print("Final V1 verified",counts,flush=True)
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("browser_report",type=Path)
    args = parser.parse_args()
    verify(args.browser_report)
