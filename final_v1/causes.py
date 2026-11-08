"""Classify the shape of each archived native outage using exact RF replay."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np

from adaptive_joint_expanded import ARTIFACTS as NAV_ARTIFACTS, LEGACY
from adaptive_joint_expanded.baseline import paired_shadow
from adaptive_joint_expanded.features import THRESHOLD, widest
from swarm_sim.config import Scenario
from usable.collect import prepare_city_without_unused_radio_cache
from forecast_prevention.diagnostics import connected, outage_runs
from forecast_rebuild.contracts import maps, read, sha, write
from forecast_shield.forecaster import geometry
from outage_atlas.radio import probabilities, radio_graph

from . import ARTIFACTS, ROOT

ATLAS = ROOT.parent / "outage_atlas" / "artifacts"
RULE = {"abrupt_db_per_half_second": 6., "gradual_db_over_history": 3., "history_s": 6.}


def radio_samples(city, scenario, positions, frames, seed, kind, jammer, search_count):
    step, fraction = np.divmod(frames, 6)
    fraction = fraction[:, None, None] / 6.
    xy = positions[step] * (1.-fraction) + positions[step+1] * fraction
    loss, _ = geometry(city, scenario, xy)
    n = positions.shape[1]
    shadow = paired_shadow(seed, int(frames[-1])+1, n, .5, scenario.shadow_sigma_db,
                           scenario.shadow_correlation_s)[frames]
    noise = np.zeros((len(frames), n))
    if kind != "clear":
        receiver = np.c_[xy.reshape(-1, 2), np.full(len(frames)*n, scenario.altitude_m)]
        jammer_loss = city.radio.propagation(np.array([jammer["jammer_xy_m"]+[2.]]), receiver, scenario)[0]
        noise = 10**((jammer["tx_power_dbm"]-jammer_loss.reshape(-1, n))/10.)
        if kind == "intermittent":
            active = (frames*.5 >= jammer["intermittent_start_s"]) & (frames*.5 < jammer["intermittent_stop_s"])
            noise *= active[:, None]
    directed = 10.-loss-shadow-10.*np.log10(10**(-10.7)+noise)[:, None, :]-THRESHOLD
    margin = widest(np.minimum(directed, directed.swapaxes(-2, -1)), search_count).min(axis=-1)
    return margin, radio_graph(probabilities(loss, noise, shadow))


def pattern(trace, abrupt_db=6., gradual_db=3.):
    if len(trace) < 13:
        return "insufficient_history"
    jump = trace[-2]-trace[-1]
    earlier = float(np.median(trace[:4])-np.median(trace[-5:-1]))
    abrupt, gradual = jump >= abrupt_db, earlier >= gradual_db
    return ("mixed" if gradual else "abrupt") if abrupt else ("gradual" if gradual else "other")


def events_for(city, scenario, positions, links, spec, jammer, ns):
    starts_stops = outage_runs(connected(links, ns))
    if not starts_stops:
        return []
    frames = np.unique(np.concatenate([np.arange(max(0, start-12), start+1) for start, _ in starts_stops]))
    margin, rebuilt = radio_samples(city, scenario, positions, frames, spec["seed"], spec["kind"], jammer, ns)
    np.testing.assert_array_equal(rebuilt, links[frames])
    results = []
    for start, stop in starts_stops:
        left, right = np.searchsorted(frames, [max(0, start-12), start+1])
        trace = margin[left:right]
        results.append(dict(onset_s=start*.5, end_s=stop*.5, pattern=pattern(trace),
                            sensitivity_4_2=pattern(trace, 4., 2.),
                            sensitivity_8_4=pattern(trace, 8., 4.),
                            margin_db=np.round(trace, 2).tolist(),
                            right_censored=stop==len(links)))
    return results


def analyze_archive(task):
    entry, registry = task
    if sha(entry["path"]) != entry["sha256"]:
        raise ValueError("Atlas metadata changed")
    data = read(entry["path"])
    spec = data["spec"]
    path = LEGACY / registry["path"]
    if sha(path) != registry["sha256"] or sha(data["archive_path"]) != data["archive_sha256"]:
        raise ValueError("Atlas map or RF trace changed")
    scenario = Scenario(direction=spec["direction"], motion_dt_s=3).validate()
    city = prepare_city_without_unused_radio_cache(path, scenario, NAV_ARTIFACTS/"cache/city")
    with np.load(data["archive_path"], allow_pickle=False) as arrays:
        events = events_for(city, scenario, city.points[arrays["nodes"]], arrays["actual_links"],
                            spec, data["jammer"], data["mission"]["search_drones"])
    original = data["rf_cases"][spec["kind"]]["network_events"]
    if [(r["onset_s"], r["end_s"]) for r in events] != [(r["onset_s"], r["onset_s"]+r["duration_s"]) for r in original]:
        raise ValueError("Atlas onsets changed")
    for event, source in zip(events, original):
        event["repair_factors"] = source["first_second_minimal_repairs"]
        event["transition_factors"] = source.get("minimal_prior_term_restorations")
        event["observed_before"] = source.get("observed_before")
    return dict(region=spec["region"], kind=spec["kind"], split=entry["split"], events=events)


def main():
    path = ATLAS/"manifest.json"
    if sha(path) != read(ATLAS/"completed.json")["manifest_sha256"]:
        raise ValueError("Atlas manifest changed")
    registry = maps()
    records = read(path)["records"]
    if len(records) != 345 or len({item["region"] for item in records}) != 345:
        raise ValueError("Unexpected map coverage")
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn")) as pool:
        tasks = [(entry, registry[entry["region"]]) for entry in records]
        results = []
        for count, row in enumerate(pool.map(analyze_archive, tasks, chunksize=3), 1):
            results.append(row)
            if count % 30 == 0 or count == len(records):
                print("cause maps", count, "/", len(records), flush=True)
    by_pattern = defaultdict(Counter)
    sensitivity = {key: Counter() for key in ("pattern", "sensitivity_4_2", "sensitivity_8_4")}
    left = 0
    for flight in results:
        for event in flight["events"]:
            if not event["onset_s"]:
                left += 1
                continue
            for key in sensitivity:
                sensitivity[key][event[key]] += 1
            summary = by_pattern[event["pattern"]]
            summary["events"] += 1
            summary["building_alone_repairs_first_second"] += ["building"] in event["repair_factors"]
            summary["received_history_available"] += bool((event["observed_before"] or {}).get("decision_window"))
            summary["all_search_beacons_and_rf_at_eligible_tick"] += bool((event["observed_before"] or {}).get("qualified_decisions"))
    if sum(sensitivity["pattern"].values()) != 2572 or left != 14:
        raise ValueError("Unexpected event denominator")
    value = dict(maps=len(results), new_outages=2572, initially_down=left,
                 rule=RULE, source_sha256=sha(path), source_code_sha256=sha(Path(__file__)),
                 by_pattern=by_pattern, sensitivity=sensitivity, flights=results)
    write(ARTIFACTS/"causes.json", value)
    print(dict(by_pattern), flush=True)
    return value


if __name__ == "__main__":
    main()
