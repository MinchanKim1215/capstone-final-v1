"""선정 변경 없이 결과 재계산·독립 궤적/연결성 검사·HTML 생성."""

from pathlib import Path
import argparse, json
import numpy as np
import shapely
from swarm_sim.regions import ROOT, sha256_file
from swarm_sim.config import Scenario
from usable.collect import prepare_city_without_unused_radio_cache
from cooperative.analyze import rows, summarize, paired_ci, test_summary, key


def audit_trajectories(groups):
    entries = {
        e["key"]: e
        for e in json.loads(
            (ROOT / "configs/usable_protocol.json").read_text(encoding="utf-8")
        )["maps"]
    }
    cities = {}
    records = []
    ii, jj = np.triu_indices(8, 1)
    for label, item in groups.items():
        for r in rows(ROOT / item["folder"]):
            region = r["region"]
            if region not in cities:
                cities[region] = prepare_city_without_unused_radio_cache(
                    ROOT / entries[region]["path"],
                    Scenario(
                        duration_s=7200, motion_dt_s=3, direction=r["direction"]
                    ).validate(),
                    ROOT / "cache/usable",
                )
            city = cities[region]
            file = ROOT / r["result_path"]
            with np.load(file.with_suffix(".npz")) as z:
                nodes = z["nodes"]
                p = (
                    np.asarray(z["positions_xy"], float)
                    if "positions_xy" in z
                    else city.points[nodes]
                )
                if not np.array_equal(p, city.points[nodes]):
                    raise ValueError("좌표와 노드 불일치")
                if np.any(np.abs(p) > 195 + 1e-8):
                    raise ValueError("지도 경계 위반")
                d = np.diff(p, axis=0)
                speed = float(np.linalg.norm(d, axis=-1).max() / 3)
                a = p[:-1, ii] - p[:-1, jj]
                dv = d[:, ii] - d[:, jj]
                scale = np.sum(dv * dv, axis=-1)
                u = np.zeros_like(scale)
                np.divide(-np.sum(a * dv, axis=-1), scale, out=u, where=scale > 0)
                u = np.clip(u, 0, 1)
                separation = float(np.linalg.norm(a + u[..., None] * dv, axis=-1).min())
                segments = shapely.linestrings(
                    np.stack((p[:-1], p[1:]), axis=-2).reshape(-1, 2, 2)
                )
                blocked = int(
                    np.count_nonzero(shapely.intersects(segments, city.flight_blocker))
                )
                if speed > 5 + 1e-7 or separation < 5 - 1e-7 or blocked:
                    raise ValueError("궤적 안전 제한 위반")
                if abs(separation - r["minimum_separation_m"]) > 1e-7:
                    raise ValueError("연속 최소 분리 거리 불일치")
                observed = np.any(city.visible[nodes[:, :3]], axis=(0, 1))
                coverage = float(observed[city.observable].mean())
                if abs(coverage - r["observable_coverage"]) > 1e-10:
                    raise ValueError("수색 완료율 불일치")
                if r["mission_completed"] and coverage < 1:
                    raise ValueError("완주 표기와 실제 수색 누락")
                # 원 엔진의 Floyd-Warshall과 별개인 불리언 도달 폐포로 다시 계산.
                reach = z["links"].copy()
                for k in range(8):
                    reach |= reach[:, :, k, None] & reach[:, None, k, :]
                connected = reach[:, 0, 1] & reach[:, 0, 2]
                if not np.array_equal(connected, z["C"][:, 0]):
                    raise ValueError("수색군 연결성 원본과 불일치")
                outage = float((~connected).sum() * 0.5)
                padded = np.r_[False, ~connected, False].astype(np.int8)
                change = np.diff(padded)
                begin = np.flatnonzero(change == 1)
                end = np.flatnonzero(change == -1)
                longest = float(np.max(end - begin) * 0.5) if len(begin) else 0.0
                if (
                    outage != r["outage_total_s"]
                    or longest != r["outage_max_s"]
                    or len(begin) != r["outage_events"]
                ):
                    raise ValueError("단절 시간·횟수 불일치")
                if abs(connected.mean() - r["search_connected_mean"]) > 1e-12:
                    raise ValueError("연결 유지율 평균 불일치")
                if (
                    r["generated_packets"]
                    != r["delivered_packets"]
                    + r["ttl_drops"]
                    + r["retry_drops"]
                    + r["overflow_drops"]
                ):
                    raise ValueError("패킷 보존 위반")
                if (
                    abs(r["delivered_packets"] / r["generated_packets"] - r["pdr"])
                    > 1e-12
                ):
                    raise ValueError("PDR 불일치")
            records.append(
                dict(
                    group=label,
                    region=region,
                    seed=r["seed"],
                    kind=r["kind"],
                    direction=r["direction"],
                    mission_completed=r["mission_completed"],
                    coverage=coverage,
                    minimum_swept_separation_m=separation,
                    maximum_speed_mps=speed,
                    blocked_segments=blocked,
                    connectivity_recomputed=True,
                )
            )
    result = {
        "episodes": len(records),
        "all_valid": True,
        "minimum_swept_separation_m": min(
            r["minimum_swept_separation_m"] for r in records
        ),
        "maximum_speed_mps": max(r["maximum_speed_mps"] for r in records),
        "blocked_segments": sum(r["blocked_segments"] for r in records),
        "records": records,
    }
    (ROOT / "results/cooperative/trajectory_integrity_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def finalize(html=True):
    selection_path = ROOT / "results/cooperative/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    freeze = json.loads(
        (ROOT / "configs/cooperative_test_freeze.json").read_text(encoding="utf-8")
    )
    if sha256_file(selection_path) != freeze["selection_sha256"]:
        raise ValueError("시험 전 선정 파일이 바뀜")
    learned = selection["best_learned"]["label"]
    checkpoint = selection["best_learned"]
    if sha256_file(ROOT / checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]:
        raise ValueError("최선 학습 가중치 변경")
    groups = {
        label: {"folder": "results/cooperative/test/final/" + ctl, "controller": ctl}
        for label, ctl in [
            ("nominal", "nominal"),
            ("reactive", "reactive"),
            ("mpc", "mpc"),
            (learned, "ppo"),
        ]
    }
    for r in rows(ROOT / groups[learned]["folder"]):
        if r.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
            raise ValueError("시험 결과 가중치 불일치")
    summary = test_summary(groups)
    full = ROOT / "results/cooperative/ablation_full_summary.json"
    if full.exists():
        comparison = []
        for label, original, removed in [
            ("예측 MPC", "mpc", "mpc_no_risk"),
            ("학습 정책", learned, "ppo_no_risk"),
        ]:
            a = rows(ROOT / groups[original]["folder"])
            b = rows(ROOT / "results/cooperative/test/ablations" / removed)
            if len(a) != 64 or len(b) != 64:
                raise ValueError("전체 입력 제거 진단 64회 누락")
            comparison.append(
                dict(
                    label=label,
                    with_risk=summarize(a),
                    without_risk=summarize(b),
                    difference=paired_ci(a, b, "search_connected_mean"),
                    pdr_difference=paired_ci(a, b, "pdr"),
                )
            )
        full.write_text(
            json.dumps(
                {
                    "comparisons": comparison,
                    "episodes": 128,
                    "scenario_count": 64,
                    "unique_episodes_includes_preplanned_subset": True,
                    "selection_changed": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        groups.update(
            {
                k: {
                    "folder": "results/cooperative/test/ablations/" + k,
                    "controller": k,
                }
                for k in ("mpc_no_risk", "ppo_no_risk")
            }
        )
    audit = audit_trajectories(groups)
    if html:
        from cooperative.report import build as report
        from cooperative.replay import build as replay

        report()
        replay()
    return {
        "primary_test_episodes": summary["episodes"],
        "audited_episodes": audit["episodes"],
        "selected": selection["selected"],
        "minimum_separation_m": audit["minimum_swept_separation_m"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-html", action="store_true")
    a = p.parse_args()
    print(json.dumps(finalize(not a.skip_html), ensure_ascii=False, indent=2))
