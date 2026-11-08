"""45개 지역·2개 방향 전체 안전 궤적의 보정 0 동등성 검사."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json, time
import numpy as np
from swarm_sim.regions import ROOT
from swarm_sim.config import Scenario, Design
from usable.collect import prepare_city_without_unused_radio_cache
from cooperative.motion import motion_stream


def one(entry):
    rows = []
    for direction in ("east_west", "north_south"):
        scenario = Scenario(
            duration_s=7200, motion_dt_s=3, direction=direction
        ).validate()
        city = prepare_city_without_unused_radio_cache(
            ROOT / entry["path"], scenario, ROOT / "cache/usable"
        )
        stream = motion_stream(city, Design(8, 5, "v", 40).validate(), scenario)
        proposal = next(stream)
        nodes = [proposal["nodes"]]
        while True:
            event = stream.send(np.zeros((8, 2)))
            nodes.append(event["new"])
            try:
                proposal = next(stream)
            except StopIteration as end:
                mission = end.value
                break
        with np.load(
            ROOT / "cache/usable" / f"{entry['key']}_{direction}_motion.npz"
        ) as z:
            assert np.array_equal(z["nodes"], nodes), (entry["key"], direction)
            assert np.array_equal(z["desired"], mission["desired"])
            assert np.array_equal(z["phase"], mission["phase"])
        rows.append(
            {
                "region": entry["key"],
                "direction": direction,
                "duration_s": mission["duration_s"],
                "complete": mission["mission_completed"],
                "minimum_separation_m": mission["minimum_swept_separation_m"],
                "exact": True,
            }
        )
    return rows


if __name__ == "__main__":
    started = time.perf_counter()
    maps = json.loads(
        (ROOT / "configs/usable_protocol.json").read_text(encoding="utf-8")
    )["maps"]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one, e) for e in maps]):
            r = f.result()
            rows.extend(r)
            print(
                json.dumps({"checked": len(rows), "last": r[-1]["region"]}), flush=True
            )
    result = {
        "paths": sorted(rows, key=lambda x: (x["region"], x["direction"])),
        "count": len(rows),
        "all_exact": all(r["exact"] for r in rows),
        "all_completed": all(r["complete"] for r in rows),
        "wall_s": time.perf_counter() - started,
    }
    (ROOT / "results/cooperative/motion_parity90.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
