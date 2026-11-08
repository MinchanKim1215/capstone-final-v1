"""사전 지정한 학습·시험 세 지역의 두 방식 전체 임무 재생 자료."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, json
from swarm_sim.regions import ROOT
from cooperative.run import episode


def plan():
    protocol = json.loads(
        (ROOT / "configs/cooperative_evaluation.json").read_text(encoding="utf-8")
    )
    maps = json.loads(
        (ROOT / "configs/usable_protocol.json").read_text(encoding="utf-8")
    )["maps"]
    specifications = [
        dict(
            region="teheran",
            seed=2026102601,
            kind="intermittent",
            direction="east_west",
        )
    ]
    for region, direction, kind in [
        ("bucheon_jungdong_east", "east_west", "continuous"),
        ("seoul_20261023_03", "north_south", "intermittent"),
    ]:
        specifications.append(
            next(
                s
                for s in protocol["test"]
                if s["region"] == region
                and s["direction"] == direction
                and s["kind"] == kind
            )
        )
    names = {
        "teheran": "테헤란 · 학습 지역",
        "bucheon_jungdong_east": "부천 중동 · 시험 지역",
        "seoul_20261023_03": "서울 03 · 시험 지역",
    }
    cases = []
    for s in specifications:
        entry = next(m for m in maps if m["key"] == s["region"])
        prefix = "results/cooperative/demos/" + s["region"]
        cases.append(
            dict(
                **s,
                name=names[s["region"]],
                map=entry["path"],
                info="400m×400m · 고도 50m · 전체 임무 · 두 방식 공통 간섭 시드",
                runs=[prefix + "_nominal", prefix + "_selected"]
            )
        )
    result = {
        "format": "cooperative-demos-v1",
        "selection_by_test_outcomes": False,
        "cases": cases,
    }
    file = ROOT / "configs/cooperative_demos.json"
    if file.exists():
        assert json.loads(file.read_text(encoding="utf-8")) == result
    else:
        file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def worker(item):
    spec, controller, checkpoint, prefix = item
    meta = Path(prefix).with_suffix(".json")
    frames = Path(prefix).with_suffix(".frames.json")
    array = Path(prefix).with_suffix(".npz")
    from cooperative.io import validated_result
    from swarm_sim.regions import sha256_file

    expected = sha256_file(checkpoint) if checkpoint else None
    previous = validated_result(prefix, spec, expected)
    if previous is not None and frames.exists():
        r = json.loads(frames.read_text(encoding="utf-8"))
        if (
            r["frames"][0]["time_s"] == 0
            and r["frames"][-1]["time_s"] == previous["duration_s"]
            and r.get("capture_schema") == "cooperative-telemetry-v2"
        ):
            return previous
    current = episode(spec, controller, checkpoint, prefix, capture=True)
    if previous is not None:
        # 출력 상태를 보강하는 재기록이 기존 실험의 수치를 바꾸지 않는지 확인.
        keys = (
            "duration_s",
            "search_connected_mean",
            "pair_reach_mean",
            "pdr",
            "generated_packets",
            "delivered_packets",
            "outage_total_s",
            "outage_max_s",
            "outage_events",
            "relay_travel_m",
            "search_travel_m",
            "observable_coverage",
            "minimum_separation_m",
        )
        for k in keys:
            if abs(current[k] - previous[k]) > 1e-10:
                raise RuntimeError("재기록의 실험 수치 불일치: " + k)
        current["reproduction_of_prior_capture"] = True
        meta.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    selection = json.loads(
        (ROOT / "results/cooperative/selection.json").read_text(encoding="utf-8")
    )
    policy = selection["selected_policy"]
    jobs = []
    for c in plan()["cases"]:
        spec = {k: c[k] for k in ("region", "seed", "kind", "direction")}
        for index in (0, 1):
            controller = "nominal" if index == 0 else policy["controller"]
            ck = (
                None
                if index == 0 or not policy.get("checkpoint")
                else str(ROOT / policy["checkpoint"])
            )
            jobs.append((spec, controller, ck, str(ROOT / c["runs"][index])))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker, j) for j in jobs]):
            r = f.result()
            results.append(r)
            print(
                json.dumps(
                    {
                        "region": r["region"],
                        "controller": r["controller"],
                        "duration_s": r["duration_s"],
                        "complete": r["mission_completed"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    (ROOT / "results/cooperative/demos_complete.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
