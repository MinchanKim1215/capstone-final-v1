"""모델·확률 보정·경고 정책 동결 후 새 난수 24개 비행 생성."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from swarm_sim.config import Scenario, Design
from swarm_sim.forecast import atomic_npz
from upgrade.map_setup import digest
from usable.collect import (
    prepare_city_without_unused_radio_cache,
    trajectory,
    collect_one,
)
from usable.spatial import features
from advance.infer import selected
from advance.data import RESULT
from advance.train import persist


def job(spec, p):
    e = next(e for e in p["maps"] if e["key"] == spec["region"])
    s = Scenario(duration_s=7200, motion_dt_s=3, direction=spec["direction"]).validate()
    d = Design(**p["design"])
    city = prepare_city_without_unused_radio_cache(
        ROOT / e["path"], s, ROOT / "cache/usable"
    )
    a, meta = trajectory(city, d, s, ROOT / "cache/usable")
    id = (
        f"advance24_{spec['region']}_{spec['direction']}_{spec['kind']}_s{spec['seed']}"
    )
    folder = ROOT / "datasets/advance_confirmation/episodes" / id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "episode.json").exists():
        item = read_json(folder / "episode.json")
    else:
        item = collect_one(
            city, a, meta, d, s, spec["seed"], spec["kind"], folder, p, "confirmation"
        )
    if not (folder / "spatial_observations.npz").exists():
        with np.load(folder / "observations.npz") as f:
            x, m = f["features"], f["mask"]
        z, known = features(city, x, m)
        atomic_npz(
            folder / "spatial_observations.npz",
            features=z,
            mask=known,
            source_sha256=sha256_file(ROOT / "usable/spatial.py"),
        )
    print("fresh flight", id, flush=True)
    return {
        "id": id,
        "region": spec["region"],
        "split": "confirmation",
        "path": folder.relative_to(ROOT).as_posix(),
        "metadata": item,
    }


def main():
    path, _ = selected()
    p = read_json(ROOT / "configs/usable_protocol.json")
    jobs = []
    for q, key in enumerate(p["splits"]["test"]):
        for k, kind in enumerate(["clear", "continuous", "intermittent"]):
            direction = ["east_west", "north_south"][(q + k) % 2]
            seed = int(
                digest(["advance-resumed-fresh24-20261026-v3", key, direction, kind])[
                    :8
                ],
                16,
            ) % (2**31 - 2000004)
            jobs.append(
                {"region": key, "direction": direction, "kind": kind, "seed": seed}
            )
    plan = {
        "jobs": jobs,
        "checkpoint_sha256": sha256_file(path),
        "calibration_sha256": sha256_file(RESULT / "selected/calibration.json"),
        "policy_sha256": sha256_file(RESULT / "selected/alarm_policy.json"),
        "frozen_before_collection": True,
        "used_for_selection": False,
        "new_randomness_same_eight_known_geographies": True,
        "same_fixed_motion_policy": True,
    }
    file = ROOT / "configs/advance_confirmation24.json"
    if file.exists() and read_json(file) != plan:
        raise ValueError("확인 실험 계획 변경 금지")
    write_json(file, plan)
    records = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job, j, p) for j in jobs]):
            records.append(f.result())
    write_json(
        ROOT / "datasets/advance_confirmation/collection_manifest.json",
        {
            "complete": len(records) == 24,
            "records": sorted(records, key=lambda r: r["id"]),
            "plan": plan,
        },
    )
    persist("fresh24_collected")


if __name__ == "__main__":
    main()
