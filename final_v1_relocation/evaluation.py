"""개선안 문서의 확인 지표를 계산한다.

* 공통: 새 단절/시간, 단절 초/시간, 기한 내 전달, 임무 시간, 중계 이동, 완주, 안전 위반
* 개선안 1: 10초 이상 단절 수와 단절 시간 비중, 단절 길이 중앙값과 90%, 복구 명령 수
* 개선안 2: 10초 안 안정 복구 비율, 복구 시간 중앙값, 10초 이상 단절, 안전 위반
* 개선안 3: 예방 명령 수, 중계 이동 거리, 패킷 지연
* 개선안 4: 급락형 단절 수, 중계 이동 거리
V21 대비 차이는 Final V1과 같이 공간 그룹 단위 부트스트랩(4,000회) 95% 구간으로 낸다.
"""
from collections import Counter, defaultdict

import numpy as np

LONG_S = 10.
SEED = 2026111101

# 개선안별 주지표. 채택 판단은 Final V1과 같은 안전·완주·전달·임무 시간 조건에 주지표 상한 < 0을 더한다.
PRIMARY = {
    "r1_cut_recovery": "long_per_h",
    "r13_combined": "starts_per_h",
    "r4_bypass": "starts_per_h",
}


def hours(rows):
    return sum(r["duration_s"] for r in rows)/3600.


def new_cuts(row):
    return [(start, stop) for start, stop in row["persistent_cut_intervals"] if start > 0.]


def outage_profile(rows):
    lengths = np.asarray([stop-start for r in rows for start, stop in new_cuts(r)], dtype=float)
    total = float(sum(r["persistent_cut_seconds"] for r in rows))
    long = lengths[lengths >= LONG_S]
    h = hours(rows)
    some = len(lengths) > 0
    return dict(new_outages=int(len(lengths)), long_outages=int(len(long)), long_outages_per_h=len(long)/h,
                cut_seconds=total, long_cut_seconds=float(long.sum()),
                long_share_of_cut_time=float(long.sum()/total) if total else 0.,
                outage_median_s=float(np.median(lengths)) if some else None,
                outage_p90_s=float(np.percentile(lengths, 90)) if some else None,
                outage_max_s=float(lengths.max()) if some else None)


def commands(rows):
    count = Counter()
    v18 = Counter()
    internal = Counter()
    decisions = Counter()
    for row in rows:
        for event in row["intervention"]:
            reason = event["reason"]
            count[reason+"_commands"] += 1
            if reason == "recovery":
                count["recovery_while_actually_down"] += not event["actual_pre_action_connected"]
                if event.get("expert") == "cut_recovery":
                    count["cut_recovery_commands"] += 1
                    count["cut_recovery_while_down"] += not event["actual_pre_action_connected"]
            if reason == "prevention" and event.get("ambiguous_band"):
                count["ambiguous_band_prevention"] += 1
            count["executed_as_requested"] += bool(event["executed_as_requested"])
            count["not_received"] += not event.get("command_received", True)
        v18.update(row.get("v18_reasons", {}))
        internal.update(row.get("relocation_counts", {}))
        decisions.update(row.get("decision_counts", {}))
    keys = ("recovery_commands", "cut_recovery_commands", "cut_recovery_while_down", "recovery_while_actually_down",
            "prevention_commands",
            "ambiguous_band_prevention", "bypass_commands")
    total_v18 = sum(v18.values())
    return dict({k: int(count[k]) for k in keys}, all_commands=dict(count), decision_counts=dict(decisions),
                v18_reasons=dict(v18), relocation_counts=dict(internal),
                v18_insufficient_observation_share=v18["insufficient_observation"]/total_v18 if total_v18 else None)


def service(rows):
    p95 = [r["delay_p95_s"] for r in rows if r.get("delay_p95_s") is not None]
    mean = [r["delay_mean_s"] for r in rows if r.get("delay_mean_s") is not None]
    return dict(delay_mean_s=float(np.mean(mean)) if mean else None,
                delay_p95_mean_s=float(np.mean(p95)) if p95 else None)


def pooled(rows):
    h = hours(rows)
    generated = sum(r["generated_packets"] for r in rows)
    timely = sum(r["timely_pdr"]*r["generated_packets"] for r in rows)
    starts = sum(r["persistent_onsets"] for r in rows)
    seconds = sum(r["persistent_cut_seconds"] for r in rows)
    return dict(flights=len(rows), hours=h, starts=int(starts), starts_per_h=starts/h, cut_s_per_h=seconds/h,
                timely_pdr=timely/generated if generated else None,
                mean_mission_min=float(np.mean([r["duration_s"] for r in rows]))/60.,
                completed=int(sum(bool(r["mission_completed"]) for r in rows)),
                safety=int(sum(r["actual_safety_violations"] for r in rows)),
                mean_relay_km=float(np.mean([r["relay_travel_m"] for r in rows]))/1000.)


def describe(rows):
    return dict(pooled(rows), **outage_profile(rows), **commands(rows), **service(rows))


def key(row):
    spec = row["spec"]
    return spec["region"], spec["kind"], spec["seed"], spec["direction"]


METRICS = {
    "starts_per_h": lambda r: r["persistent_onsets"],
    "long_per_h": lambda r: sum(stop-start >= LONG_S for start, stop in new_cuts(r)),
    "cut_s_per_h": lambda r: r["persistent_cut_seconds"],
}


def compare(reference, alternative, group_of, draws=4000):
    """같은 지도·난수 쌍의 차이. 공간 그룹을 단위로 다시 뽑는다."""
    ref, alt = ({key(r): r for r in rows} for rows in (reference, alternative))
    if ref.keys() != alt.keys():
        raise ValueError("Unmatched paired simulation")
    groups = defaultdict(list)
    for k in sorted(ref):
        a, b = ref[k], alt[k]
        groups[group_of(a)].append([a["duration_s"]/3600., b["duration_s"]/3600.]
                                   + [f(a) for f in METRICS.values()] + [f(b) for f in METRICS.values()])
    totals = np.stack([np.sum(v, axis=0) for v in groups.values()])
    picks = np.random.default_rng(SEED).integers(0, len(totals), size=(draws, len(totals)))
    sample = totals[picks].sum(axis=1)
    m = len(METRICS)
    result = dict(groups=len(groups))
    for index, name in enumerate(METRICS):
        a = totals[:, 2+index].sum()/totals[:, 0].sum()
        b = totals[:, 2+m+index].sum()/totals[:, 1].sum()
        spread = sample[:, 2+m+index]/sample[:, 1]-sample[:, 2+index]/sample[:, 0]
        result[name+"_delta"] = float(b-a)
        result[name+"_ci95"] = np.quantile(spread, [.025, .975]).tolist()
    p, q = pooled(reference), pooled(alternative)
    result.update(timely_pdr_pp=(q["timely_pdr"]-p["timely_pdr"])*100. if p["timely_pdr"] else None,
                  duration_fraction=q["hours"]/p["hours"]-1.,
                  relay_km_delta=q["mean_relay_km"]-p["mean_relay_km"],
                  long_outages_delta=sum(METRICS["long_per_h"](r) for r in alternative)
                  - sum(METRICS["long_per_h"](r) for r in reference))
    return result


def adoptable(stats, change, flights, primary="starts_per_h"):
    """Final V1 승격 조건과 같다: 안전 위반 0, 전원 완주, 기한 내 전달 손실 0.25%p 이하,
    임무 시간 증가 2% 이하, 주지표 95% 구간 상한 < 0."""
    return bool(stats["safety"] == 0 and stats["completed"] == flights
                and (change["timely_pdr_pp"] is None or change["timely_pdr_pp"] >= -.25)
                and change["duration_fraction"] <= .02 and change[primary+"_ci95"][1] < 0.)


def stable_recovery(rows, archive):
    """명령 손실 조건의 복구: 연속 두 프레임(1초) 이상 연결이 이어지기 시작한 시점까지."""
    count = Counter()
    durations = []
    for row in rows:
        network = archive(row)["network"]
        for start, stop in new_cuts(row):
            count["outages"] += 1
            count["long_outages"] += stop-start >= LONG_S
            frame = int(stop*2)
            pairs = np.flatnonzero(network[frame:-1] & network[frame+1:])
            if len(pairs):
                duration = stop-start+pairs[0]*.5
                durations.append(duration)
                count["recovered_within_10s"] += duration <= LONG_S
            else:
                count["no_stable_recovery_before_end"] += 1
    return dict({k: int(count[k]) for k in ("outages", "long_outages", "recovered_within_10s",
                                           "no_stable_recovery_before_end")},
                recovered_within_10s_rate=count["recovered_within_10s"]/count["outages"] if count["outages"] else None,
                median_stable_recovery_s=float(np.median(durations)) if durations else None)


def fallback_reasons(rows, archive):
    count = Counter()
    for row in rows:
        data = archive(row)
        count.update(data["fallback"][:, 1:].ravel().tolist())
        rx = data["received"][:, 1:]
        count["remote_commands"] += int(rx.size)
        count["remote_received"] += int(rx.sum())
    return dict(count)


def select_evidence(stats, arms):
    """복구 근거 선택 규칙 (개발 비행 전에 고정). 10초 이상 단절/시간이 V21보다 나쁘지 않은 설정 중
    단절 초/시간이 가장 작은 것, 같으면 복구 명령이 적은 것."""
    candidates = [a for a in arms if a != "v21"]
    eligible = [a for a in candidates if stats[a]["long_outages_per_h"] <= stats["v21"]["long_outages_per_h"]]
    pick = min(eligible or candidates, key=lambda a: (round(stats[a]["cut_s_per_h"], 6), stats[a]["recovery_commands"]))
    return dict(selected=pick, eligible=eligible,
                rule="10초 이상 단절/시간이 V21보다 나쁘지 않은 설정 중 단절 초/시간 최소, 같으면 복구 명령 최소")
