"""Paired control and same-trajectory alert evaluation."""
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from forecast_learning.control import load_profile
from forecast_learning.features import COLUMNS
from forecast_learning.training import load, make_score

from .policy import Policy, scores, timing_labels
from .protocol import V21, maps, read
from .training import read_checkpoint

WINDOWS = dict(short=(.5,3.), three_to_five=(3.,5.), five_to_seven=(5.,7.), early=(3.,7.))
VARIANTS = dict(v20="early", v21="early", joint_short="short", joint_3to5="three_to_five",
                joint_5to7="five_to_seven", joint_early="early")
PHYSICAL_COLUMN = COLUMNS.index("held_shield")


def forecast_model(training):
    model = Policy(True)
    model.load_state_dict(read_checkpoint(training["models"]["ppo_continuous"]["path"])["state"])
    return model.eval()


def forecast_scores(model, x):
    probability = scores(model, x)
    _, profile, early, _ = load_profile()
    physical = x[:, PHYSICAL_COLUMN]
    blended = np.full(len(x), np.nan)
    usable = np.isfinite(physical)
    blended[usable] = make_score(early, profile, x[usable], physical[usable])
    return dict(v20=physical, v21=blended, joint_short=probability[:, 1],
                joint_3to5=probability[:, 2], joint_5to7=probability[:, 3],
                joint_early=probability[:, 2]+probability[:, 3])


def advisory_case(row, times, x, network, model, events=None):
    events = row["persistent_cut_intervals"] if events is None else events
    cuts = [event for event in events if event[0] > 0.]
    return dict(hours=row["duration_s"]/3600., times=times,
                onsets=np.asarray([e[0] for e in cuts]),
                patterns=[e[2] if len(e)>2 else "unclassified" for e in cuts],
                scores=forecast_scores(model,x), usable=times+7. <= row["duration_s"], network=network)


def score_advisory(flights, variant, threshold, detail=False):
    stats = Counter()
    patterns = defaultdict(Counter)
    low, high = WINDOWS[VARIANTS[variant]]
    leads, y, p = [], [], []
    for case in flights:
        stats["hours"] += case["hours"]
        t, s, onsets = case["times"], case["scores"][variant], case["onsets"]
        stats["events"] += len(onsets)
        active = s >= threshold
        alerts = t[active & ~np.r_[False,active[:-1]]]
        hits = np.zeros(len(onsets), dtype=bool)
        stats["alerts"] += len(alerts)
        for when in alerts:
            first, stop = np.searchsorted(onsets, [when+low,when+high])
            for k in range(first,stop):
                if not hits[k]:
                    hits[k] = True
                    leads.append(float(onsets[k]-when))
                    break
        stats["hits"] += int(hits.sum())
        if detail:
            for k, pattern in enumerate(case["patterns"]):
                group = patterns[pattern]
                group["events"] += 1
                group["hits"] += int(hits[k])
                window = (t > onsets[k]-high) & (t <= onsets[k]-low)
                group["eligible_ticks"] += bool(window.any())
                group["supported_ticks"] += bool(np.isfinite(s[window]).any())
            usable = case["usable"] & np.isfinite(s) & (t > 0)
            label = timing_labels(t[usable], onsets, case["network"])
            bins = dict(short=(1,), three_to_five=(2,), five_to_seven=(3,), early=(2,3))
            y.extend(np.isin(label,bins[VARIANTS[variant]]))
            p.extend(s[usable])
    stats["false_or_late"] = stats["alerts"]-stats["hits"]
    result = dict(stats, recall=stats["hits"]/max(1,stats["events"]),
                  precision=stats["hits"]/max(1,stats["alerts"]),
                  false_or_late_per_fleet_h=stats["false_or_late"]/max(1e-9,stats["hours"]))
    if detail:
        result.update(by_pattern=patterns, median_lead_s=float(np.median(leads)) if leads else None,
                      average_precision=float(average_precision_score(y,p)) if any(y) else None,
                      ranking_rows=len(y), ranking_positives=int(sum(y)))
    return result


def calibrated_forecast(training, cap):
    model = forecast_model(training)
    calibration, selection = [], []
    for scope, target in (("calibration", calibration), ("selection", selection)):
        for record in read(V21/f"{scope}_manifest.json")["records"]:
            flight = load(record)
            with np.load(flight.row["archive_path"], allow_pickle=False) as z:
                target.append(advisory_case(flight.row, flight.raw[:,0], flight.x, z["network"], model))
    result = {}
    for variant, window in VARIANTS.items():
        finite = np.concatenate([case["scores"][variant][np.isfinite(case["scores"][variant])]
                                 for case in calibration])
        thresholds = np.unique(np.r_[1.01, np.quantile(finite, np.linspace(.40, 1., 91))])
        attempts = [(float(t),score_advisory(calibration,variant,t)) for t in thresholds]
        candidates = [(t,s) for t,s in attempts if s["false_or_late_per_fleet_h"]<=cap]
        threshold, metrics = max(candidates,key=lambda item:(item[1]["recall"],item[1]["precision"],
                                                             -item[1]["false_or_late_per_fleet_h"],item[0]))
        result[variant] = dict(threshold=threshold, window=WINDOWS[window], calibration=metrics,
                               selection=score_advisory(selection,variant,threshold,True))
    return result


def pooled(rows):
    hours = sum(r["duration_s"] for r in rows)/3600.
    generated = sum(r["generated_packets"] for r in rows)
    timely = sum(r["timely_pdr"]*r["generated_packets"] for r in rows)
    starts = sum(r["persistent_onsets"] for r in rows)
    seconds = sum(r["persistent_cut_seconds"] for r in rows)
    return dict(flights=len(rows), hours=hours, starts=starts, starts_per_h=starts/hours,
                cut_s_per_h=seconds/hours, timely_pdr=timely/generated,
                completed=sum(r["mission_completed"] for r in rows),
                safety=sum(r["actual_safety_violations"] for r in rows),
                mean_relay_m=float(np.mean([r["relay_travel_m"] for r in rows])),
                step_p95_median_ms=float(np.median([r["simulator_step_p95_ms"] for r in rows])))


def key(row):
    spec = row["spec"]
    return spec["region"], spec["kind"], spec["seed"], spec["direction"]


def paired(reference, alternative):
    ref, alt = ({key(r):r for r in rows} for rows in (reference,alternative))
    if ref.keys() != alt.keys() or len(ref) != len(reference) or len(alt) != len(alternative):
        raise ValueError("Unmatched paired simulation")
    groups = defaultdict(list)
    registry = maps()
    for k in sorted(ref):
        a,b = ref[k],alt[k]
        groups[registry[k[0]]["joint_spatial_group"]].append(
            (a["persistent_onsets"], a["duration_s"]/3600., b["persistent_onsets"], b["duration_s"]/3600.))
    totals = np.stack([np.sum(rows,axis=0) for rows in groups.values()])
    rng = np.random.default_rng(2026110501)
    draws = rng.integers(0,len(totals),size=(4000,len(totals)))
    x, h1, y, h2 = totals[draws].sum(1).T
    ci = np.quantile(y/h2-x/h1, [.025,.975])
    r,a = pooled(reference),pooled(alternative)
    return dict(starts_per_h_delta=a["starts_per_h"]-r["starts_per_h"], ci95=ci.tolist(), groups=len(groups),
                timely_pdr_pp=(a["timely_pdr"]-r["timely_pdr"])*100.,
                duration_fraction=a["hours"]/r["hours"]-1.,
                cut_s_per_h_delta=a["cut_s_per_h"]-r["cut_s_per_h"])


def transport_metrics(rows):
    count = Counter()
    durations = []
    for r in rows:
        with np.load(r["archive_path"], allow_pickle=False) as data:
            rx, forced = data["received"][:,1:], data["forced"][:,1:]
            count["remote_commands"] += rx.size
            count["remote_received"] += int(rx.sum())
            count["forced_misses"] += int((~rx & forced).sum())
            count["missed_changed_movement"] += int((~rx &
                (data["commands"][:, r["search_drones"]+1:] != 0.).any(-1)).sum())
            count["multihop_deliveries"] += int((data["hops"][:,1:] >= 2).sum())
            count.update(data["fallback"][:,1:].ravel().tolist())
            network = data["network"]
            for start, stop in r["persistent_cut_intervals"]:
                if start == 0.:
                    continue
                count["outages"] += 1
                frame = int(stop*2)
                # A return requires at least two consecutive healthy RF frames.
                health_pairs = np.flatnonzero(network[frame:-1] & network[frame+1:])
                if len(health_pairs):
                    duration = stop-start + health_pairs[0]*.5
                    durations.append(float(duration))
                    count["recovered_within_3s"] += int(duration <= 3.)
                    count["recovered_within_10s"] += int(duration <= 10.)
                else:
                    count["no_stable_recovery_before_end"] += 1
        for event in r["intervention"]:
            if event["reason"] == "recovery":
                count["recovery_commands"] += 1
                count["recovery_commands_while_down"] += not event["actual_pre_action_connected"]
                count["recovery_commands_not_received"] += not event["command_received"]
    result = dict(count, **{k:count[k] for k in ("outages","recovered_within_3s","recovered_within_10s",
                                                  "no_stable_recovery_before_end","recovery_commands",
                                                  "recovery_commands_while_down","recovery_commands_not_received")})
    result.update(median_stable_recovery_s=float(np.median(durations)) if durations else None,
                  command_delivery_rate=count["remote_received"]/count["remote_commands"])
    return result
