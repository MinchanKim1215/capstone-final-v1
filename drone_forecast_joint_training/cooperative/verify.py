"""전체 임무 스트리밍 동등성 및 인과성·패킷 보존 검증."""

from pathlib import Path
import json, time
import numpy as np
from cooperative.world import ClosedLoopWorld
from cooperative.candidates import local_candidates
from swarm_sim.regions import ROOT


def full_parity():
    started = time.perf_counter()
    folder = next(
        p
        for p in (ROOT / "datasets/usable45/episodes").iterdir()
        if p.name.startswith("teheran_")
    )
    meta = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
    with np.load(folder / "observations.npz") as z:
        x = z["features"]
        mask = z["mask"]
    with np.load(folder / "truth.npz") as z:
        truth = z["link_available"]
    env = ClosedLoopWorld(
        meta["region"], meta["seed"], meta["kind"], meta["direction"], forecast=False
    )
    while not env.done:
        env.step(np.zeros((8, 2)))
    nt = env.telemetry.t
    assert nt == len(x), (nt, len(x))
    assert np.array_equal(env.telemetry.mask[:nt], mask), "수신 마스크 불일치"
    delta = float(np.max(np.abs(env.telemetry.x[:nt] - x)))
    assert delta < 2e-5, delta
    assert np.array_equal(
        np.asarray(env.channel_available), truth
    ), "실제 링크 정답 불일치"
    assert np.array_equal(env.node_log, env.baseline_nodes), "기존 전체 이동과 불일치"
    metrics = env.summary
    assert (
        metrics["generated_packets"]
        == metrics["delivered_packets"]
        + metrics["ttl_drops"]
        + metrics["retry_drops"]
        + metrics["overflow_drops"]
    )
    original = meta["metrics"]
    assert abs(metrics["pdr"] - original["pdr"]) < 1e-12, (
        metrics["pdr"],
        original["pdr"],
    )
    assert (
        abs(metrics["search_connected_mean"] - original["search_connected_fraction"])
        < 1e-12
    )
    record = {
        "test": "full_mission_zero_control_parity",
        "passed": True,
        "frames": nt,
        "duration_s": nt * 0.5,
        "maximum_observation_difference": delta,
        "pdr": metrics["pdr"],
        "mask_equal": True,
        "channel_equal": True,
        "trajectory_equal": True,
        "packet_conservation": True,
        "wall_s": time.perf_counter() - started,
    }
    (ROOT / "results/cooperative/full_parity.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    full_parity()
