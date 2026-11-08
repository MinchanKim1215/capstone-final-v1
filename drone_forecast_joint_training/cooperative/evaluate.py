"""고정된 공통 시나리오에서 정책을 비교. 테스트 결과로 재학습/튜닝 금지."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, json, time, hashlib
from cooperative.run import episode
from swarm_sim.regions import ROOT, sha256_file
from cooperative.io import validated_result


def make_protocol():
    data = json.loads(
        (ROOT / "configs/usable_protocol.json").read_text(encoding="utf-8")
    )
    protocol = {
        "format": "cooperative-comparison-v1",
        "split_source": "configs/usable_protocol.json",
        "region_count": {"train": 30, "val": 7, "test": 8},
        "selection_constraints": {
            "all_missions_completed": True,
            "actual_safety_violations": 0,
            "macro_pdr_difference_vs_nominal_min": -0.005,
            "max_duration_ratio": 1.15,
        },
        "selection_order": [
            "lowest macro outage fraction",
            "highest macro timely PDR",
            "lowest macro relay travel",
        ],
        "threshold_provenance": "project engineering choices frozen before validation, not published safety standards",
        "forecast_frozen": True,
        "strict_joint_end_to_end_training": False,
        "test_used_for_tuning": False,
        "validation": [],
        "test": [],
    }
    for split, sets in [("val", 1), ("test", 2)]:
        for entry in sorted(
            (e for e in data["maps"] if e["split"] == split), key=lambda e: e["key"]
        ):
            for direction in ("east_west", "north_south"):
                for kind in ("continuous", "intermittent"):
                    for repeat in range(sets):
                        token = f"closed-loop-v1:{split}:{entry['key']}:{direction}:{kind}:{repeat}"
                        seed = int(
                            hashlib.sha256(token.encode()).hexdigest()[:8], 16
                        ) % (2**31 - 3000004)
                        protocol["validation" if split == "val" else "test"].append(
                            dict(
                                region=entry["key"],
                                direction=direction,
                                kind=kind,
                                seed=seed,
                            )
                        )
    out = ROOT / "configs/cooperative_evaluation.json"
    if out.exists():
        assert (
            json.loads(out.read_text(encoding="utf-8")) == protocol
        ), "동결 평가 계획 변경 금지"
    else:
        out.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return protocol


def job(item):
    spec, controller, ck, path = item
    meta = Path(path).with_suffix(".json")
    array = Path(path).with_suffix(".npz")
    expected = sha256_file(ck) if ck else None
    previous = validated_result(path, spec, expected)
    if previous is not None:
        return previous
    if meta.exists() or array.exists():
        abandoned = ROOT / "results/cooperative/abandoned_evaluations"
        abandoned.mkdir(parents=True, exist_ok=True)
        for file in (meta, array):
            if file.exists():
                file.replace(abandoned / file.name)
    r = episode(spec, controller, ck, path, capture=False)
    r["checkpoint_sha256"] = expected
    meta.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--controllers", nargs="+", default=["nominal", "reactive", "mpc"])
    p.add_argument("--checkpoint")
    p.add_argument("--tag", default="base")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--direction-filter", choices=["east_west", "north_south"])
    p.add_argument("--kind-filter", choices=["continuous", "intermittent"])
    a = p.parse_args()
    proto = make_protocol()
    specs = [
        s
        for s in proto[a.split]
        if (a.direction_filter is None or s["direction"] == a.direction_filter)
        and (a.kind_filter is None or s["kind"] == a.kind_filter)
    ][: a.limit]
    jobs = []
    for ctl in a.controllers:
        if ctl.startswith("ppo") and not a.checkpoint:
            raise ValueError("정책 가중치 필요")
        ck = str(Path(a.checkpoint).resolve()) if ctl.startswith("ppo") else None
        for spec in specs:
            path = (
                ROOT
                / "results/cooperative"
                / a.split
                / a.tag
                / ctl
                / f"{spec['region']}_{spec['direction']}_{spec['kind']}_{spec['seed']}"
            )
            jobs.append((spec, ctl, ck, str(path)))
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(job, j) for j in jobs]):
            r = f.result()
            results.append(r)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": len(jobs),
                        "controller": r["controller"],
                        "region": r["region"],
                        "C": r["search_connected_mean"],
                        "P": r["pdr"],
                        "T": r["duration_ratio"],
                        "ok": r["mission_completed"],
                        "elapsed_s": round(time.perf_counter() - start, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    dest = ROOT / "results/cooperative" / a.split / (a.tag + "_complete.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {"results": results, "elapsed_s": time.perf_counter() - start},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
