"""개별 지표 분석·검증 기반 선정·지역 단위 대응 부트스트랩."""

from pathlib import Path
import json, csv
import numpy as np
from swarm_sim.regions import ROOT, sha256_file

METRICS = [
    "search_connected_mean",
    "pair_reach_mean",
    "outage_total_s",
    "outage_max_s",
    "outage_events",
    "pdr",
    "timely_pdr",
    "delay_mean_s",
    "delay_p95_s",
    "aoi_mean_s",
    "duration_ratio",
    "duration_s",
    "relay_travel_m",
    "search_travel_m",
    "observable_coverage",
    "information_shared_fraction",
    "minimum_separation_m",
    "relocation_fraction",
    "protection_fraction",
    "forecast_p95_ms",
    "search_formation_rmse_m",
    "actual_delivered_hops_mean",
    "delivered_via_relay_fraction",
    "decision_p95_ms",
]


def key(r):
    return (r["region"], r["direction"], r["kind"], r["seed"])


def rows(folder):
    result = []
    for path in Path(folder).rglob("*.json"):
        if path.name.endswith(".frames.json") or ".tmp." in path.name:
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        if "search_connected_mean" not in r:
            continue
        array = path.with_suffix(".npz")
        if array.exists():
            with np.load(array) as z:
                stat = z["C"]
                packet = z["packet_frames"]
                last = packet[-1]
                r.update(
                    all_drones_connected_mean=float(stat[:, 2].mean()),
                    mean_components=float(stat[:, 3].mean()),
                    largest_component_mean=float(stat[:, 4].mean()),
                    mean_degree=float(stat[:, 5].mean()),
                    isolated_drone_fraction=float(stat[:, 6].mean()),
                    network_efficiency_mean=float(stat[:, 8].mean()),
                    search_shortest_hops_mean=float(stat[:, 9].mean()),
                    shared_observed_fraction=float(last[15] / max(last[14], 1)),
                    observed_target_fraction=float(last[3]),
                )
        r["result_path"] = str(path.relative_to(ROOT))
        result.append(r)
    return result


def summarize(items):
    regions = sorted({r["region"] for r in items})
    by = {}
    metrics = METRICS + [
        "all_drones_connected_mean",
        "mean_components",
        "largest_component_mean",
        "mean_degree",
        "isolated_drone_fraction",
        "network_efficiency_mean",
        "search_shortest_hops_mean",
        "shared_observed_fraction",
    ]
    for region in regions:
        part = [r for r in items if r["region"] == region]
        by[region] = {
            k: (
                float(np.mean([r[k] for r in part if r.get(k) is not None]))
                if any(r.get(k) is not None for r in part)
                else None
            )
            for k in metrics
        }
    macro = {
        k: (
            float(np.mean([v[k] for v in by.values() if v[k] is not None]))
            if any(v[k] is not None for v in by.values())
            else None
        )
        for k in metrics
    }
    return {
        "episodes": len(items),
        "regions": len(regions),
        "complete_count": sum(r["mission_completed"] for r in items),
        "safety_violations": sum(r["actual_safety_violations"] for r in items),
        "max_duration_ratio": max(r["duration_ratio"] for r in items),
        "minimum_separation_global": min(r["minimum_separation_m"] for r in items),
        "macro": macro,
        "by_region": by,
    }


def paired_ci(a, b, metric):
    aa = {key(r): r for r in a}
    bb = {key(r): r for r in b}
    assert aa.keys() == bb.keys(), "대응 시나리오 불일치"
    regions = sorted({r["region"] for r in a})
    values = np.array(
        [
            np.mean([aa[k][metric] - bb[k][metric] for k in aa if k[0] == region])
            for region in regions
        ]
    )
    boot = (
        np.random.default_rng(2026102699)
        .choice(values, (10000, len(values)), replace=True)
        .mean(1)
    )
    return {
        "difference": float(values.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "region_differences": dict(zip(regions, values.tolist())),
        "method": "paired region-cluster percentile bootstrap 10000; descriptive, 8 test regions",
    }


def validation_select(candidates):
    protocol = json.loads(
        (ROOT / "configs/cooperative_evaluation.json").read_text(encoding="utf-8")
    )
    expected = protocol["validation"]
    expected_keys = {key(r) for r in expected}
    table = {}
    nominal = None
    for label, item in candidates.items():
        rr = rows(ROOT / item["folder"])
        rr = [r for r in rr if r["controller"] == item["controller"]]
        if {key(r) for r in rr} != expected_keys or len(rr) != len(expected):
            raise ValueError(f"{label}: 전체 검증 28회 필요")
        if item.get("checkpoint"):
            expected_sha = sha256_file(ROOT / item["checkpoint"])
            if any(r.get("checkpoint_sha256") != expected_sha for r in rr):
                raise ValueError(f"{label}: 검증 결과 가중치 지문 재확인 필요")
        table[label] = dict(
            summarize(rr),
            controller=item["controller"],
            checkpoint=item.get("checkpoint"),
            raw=rr,
        )
        if item["controller"] == "nominal":
            nominal = table[label]
    if nominal is None:
        raise ValueError("명목 기준 실험 필요")
    feasible = []
    for label, v in table.items():
        reasons = []
        if v["complete_count"] != v["episodes"]:
            reasons.append("임무 미완주")
        if v["safety_violations"]:
            reasons.append("실제 안전 제한 위반")
        if v["macro"]["pdr"] < nominal["macro"]["pdr"] - 0.005:
            reasons.append("평균 PDR의 명목 대비 0.5%p 초과 감소")
        if v["max_duration_ratio"] > 1.15:
            reasons.append("일부 임무의 시간 비율 1.15 초과")
        v["feasible"] = not reasons
        v["rejection_reasons"] = reasons
        if not reasons:
            feasible.append(label)
    order = sorted(
        feasible,
        key=lambda k: (
            1 - table[k]["macro"]["search_connected_mean"],
            -table[k]["macro"]["timely_pdr"],
            table[k]["macro"]["relay_travel_m"],
        ),
    )
    selected = order[0] if order else "nominal"
    chosen = table[selected]
    record = {k: v for k, v in chosen.items() if k != "raw"}
    output = {
        "format": "closed-loop-validation-selection-v1",
        "selected": selected,
        "selected_policy": record,
        "ordered_feasible": order,
        "criteria": protocol["selection_constraints"],
        "ranking": protocol["selection_order"],
        "candidates": {
            k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in table.items()
        },
        "test_seen_before_selection": False,
    }
    if record["checkpoint"]:
        import shutil

        source = record["checkpoint"]
        dest = ROOT / "model/cooperative/selected.pt"
        shutil.copy2(ROOT / source, dest)
        record["source_checkpoint"] = source
        record["checkpoint"] = dest.relative_to(ROOT).as_posix()
        output["checkpoint_sha256"] = sha256_file(dest)
    (ROOT / "results/cooperative/selection.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def test_summary(groups):
    protocol = json.loads(
        (ROOT / "configs/cooperative_evaluation.json").read_text(encoding="utf-8")
    )
    expected = {key(r) for r in protocol["test"]}
    summaries = {}
    data = {}
    for label, item in groups.items():
        rr = rows(ROOT / item["folder"])
        rr = [r for r in rr if r["controller"] == item["controller"]]
        if len(rr) != len(expected) or {key(r) for r in rr} != expected:
            raise ValueError(f"{label}: 64회 시험 누락 또는 중복")
        data[label] = rr
        summaries[label] = summarize(rr)
    selection = json.loads(
        (ROOT / "results/cooperative/selection.json").read_text(encoding="utf-8")
    )
    selected = selection["selected"]
    comparisons = {
        name: {
            metric: paired_ci(data[selected], rr, metric)
            for metric in (
                "search_connected_mean",
                "pdr",
                "timely_pdr",
                "duration_ratio",
                "relay_travel_m",
            )
        }
        for name, rr in data.items()
        if name != selected
    }
    out = {
        "selected_before_test": selected,
        "groups": summaries,
        "paired_comparisons": comparisons,
        "episodes": sum(len(v) for v in data.values()),
        "rows": data,
    }
    dest = ROOT / "results/cooperative/test_summary.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = [dict(group=label, **r) for label, rr in data.items() for r in rr]
    keys = sorted({k for r in flat for k in r})
    with (dest.parent / "comparison_all_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    return out
