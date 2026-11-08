"""단일 완주 실험. 난수·지표·실제 행동 궤적을 재현 가능한 파일로 저장."""

from pathlib import Path
import argparse, json, time, hashlib
import numpy as np
import torch
from cooperative.world import ClosedLoopWorld
from cooperative.candidates import (
    heuristic_action,
    observe_policy,
    indices_to_correction,
)
from cooperative.policy import critic_state, action, load_policy


def episode(
    spec,
    controller="nominal",
    checkpoint=None,
    out=None,
    record=False,
    capture=False,
    max_steps=None,
    progress=False,
):
    torch.set_num_threads(1)
    started = time.perf_counter()
    if progress:
        print(
            json.dumps(
                {
                    "stage": "initializing",
                    "region": spec["region"],
                    "controller": controller,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    net, ck = load_policy(checkpoint) if checkpoint else (None, None)
    use_risk = (ck or {}).get("use_risk", True) and controller not in (
        "mpc_no_risk", "ppo_no_risk", "ppo_sample_no_risk"
    )
    env = ClosedLoopWorld(
        **spec,
        forecast=use_risk and controller not in ("nominal", "reactive"),
        capture=capture,
        max_steps=max_steps
    )
    rows = []
    actions = []
    decision_ms = []
    while not env.done:
        tick = time.perf_counter()
        if controller == "nominal":
            a = np.zeros(5, int)
            obs = None
            logp = np.zeros(5)
        elif controller in ("reactive", "mpc", "mpc_no_risk"):
            a, obs = heuristic_action(
                env, controller != "reactive", use_risk
            )
            logp = np.zeros(5)
        elif controller in ("ppo", "ppo_sample", "ppo_no_risk", "ppo_sample_no_risk", "ppo_guarded"):
            obs = observe_policy(env, True, use_risk)
            a, logp = action(
                net, obs, controller not in ("ppo_sample", "ppo_sample_no_risk"),
                max_teacher_regret=(ck or {}).get("guard_regret", 0.35) if controller == "ppo_guarded" else None,
            )
        else:
            raise ValueError(controller)
        decision_ms.append((time.perf_counter() - tick) * 1000)
        if record:
            if obs is None:
                obs = observe_policy(env, True)
            state = critic_state(env, use_risk=use_risk)
            with torch.no_grad():
                v = (
                    float(net.value(torch.from_numpy(state)))
                    if net is not None
                    else 0.0
                )
            rows.append((*obs[:3], obs[3].astype(np.float32), state, a.copy(), logp, v))
        actions.append(a.copy())
        env.step(indices_to_correction(a))
        if progress and (env.k % 100 == 0 or env.done):
            print(
                json.dumps(
                    {
                        "stage": "simulating",
                        "simulation_s": env.k * 3,
                        "wall_s": round(time.perf_counter() - started, 1),
                        "complete": bool(env.done),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    result = dict(
        env.summary, controller=controller, wall_s=time.perf_counter() - started
    )
    result["checkpoint"] = str(checkpoint) if checkpoint else None
    result["source_version"] = "cooperative-v2-on-policy"
    result["risk_input_enabled"] = bool(use_risk and controller not in ("nominal", "reactive"))
    result["guard_regret"] = (ck or {}).get("guard_regret", 0.35) if controller == "ppo_guarded" else None
    result["decision_p95_ms"] = float(np.percentile(decision_ms, 95))
    if ck:
        result["training_iteration"] = ck.get("iteration")
    if checkpoint:
        from swarm_sim.regions import sha256_file, ROOT

        result["checkpoint_sha256"] = sha256_file(checkpoint)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "nodes": np.asarray(env.node_log),
            "positions_xy": env.city.points[np.asarray(env.node_log)].astype(
                np.float32
            ),
            "actions": np.asarray(actions),
            "C": np.asarray(env.stats),
            "links": np.asarray(env.channel_available),
            "packet_frames": np.asarray(env.packet_frames),
            "rewards": np.array(
                [env.settled[k]["reward"] for k in range(env.k)], np.float32
            ),
        }
        if record:
            for key, index in [
                ("context", 0),
                ("candidates", 1),
                ("mask", 2),
                ("teacher", 3),
                ("state", 4),
                ("chosen", 5),
                ("logp", 6),
                ("value", 7),
            ]:
                arrays[key] = np.asarray([r[index] for r in rows])
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path.with_suffix(".npz"))
        from swarm_sim.regions import sha256_file

        result["array_sha256"] = sha256_file(path.with_suffix(".npz"))
        if capture:
            path.with_suffix(".frames.json").write_text(
                json.dumps(
                    {
                        "frames": env.frames,
                        "capture_schema": "cooperative-telemetry-v2",
                        "jammer": env.jammer.tolist(),
                        "jam_start": env.jam_start,
                        "jam_end": (
                            None if not np.isfinite(env.jam_end) else env.jam_end
                        ),
                        "map": str(
                            env.map_path.relative_to(
                                Path(__file__).resolve().parents[1]
                            )
                        ),
                        "summary": result,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf8",
            )
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
        )
        tmp.replace(path.with_suffix(".json"))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="teheran")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--kind", default="intermittent")
    p.add_argument("--direction", default="east_west")
    p.add_argument("--controller", default="nominal")
    p.add_argument("--checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--record", action="store_true")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--max-steps", type=int)
    a = p.parse_args()
    r = episode(
        dict(region=a.region, seed=a.seed, kind=a.kind, direction=a.direction),
        a.controller,
        a.checkpoint,
        a.out,
        a.record,
        a.capture,
        a.max_steps,
    )
    print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
