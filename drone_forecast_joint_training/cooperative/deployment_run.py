"""단일 완주 실험. 난수·지표·실제 행동 궤적을 재현 가능한 파일로 저장."""

from pathlib import Path
import argparse, json, time, hashlib
import numpy as np
import torch
from cooperative.continuity_world import ContinuityWorld as ClosedLoopWorld, training_state
from cooperative.continuity_control import observe as priority_observe, choose as priority_choose
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
    profile="linear",
    sources=None,
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
    if controller=='consensus_ppo':
        from cooperative.deployment_policy import load_consensus
        net,ck=load_consensus(checkpoint)
        if record:raise ValueError('Consensus is a frozen inference policy, not an on-policy training rollout')
    else:
        net, ck = load_policy(checkpoint) if checkpoint else (None, None)
    use_risk = (ck or {}).get("use_risk", True) and controller not in (
        "mpc_no_risk", "ppo_no_risk", "ppo_sample_no_risk", "priority_mpc_no_risk"
    )
    if controller in ('priority_ppo','priority_ppo_guarded','priority_sample') and (ck or {}).get('teacher_profile')!=profile:
        raise ValueError('Checkpoint and priority teacher profile differ')
    env = ClosedLoopWorld(
        **spec,
        forecast=use_risk and controller not in ("nominal", "reactive"),
        capture=capture,
        max_steps=max_steps
    )
    rows = []
    actions = []
    commands=[];applied=[];switches=[];motion=[];search_targets=[];phases=[]
    last_request=np.zeros(5,int)
    decision_ms = []
    while not env.done:
        search_targets.append(env.proposal['nominal'][:3].copy())
        phases.append(env.proposal['phase'])
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
        elif controller=='consensus_ppo':
            from cooperative.continuity_control import transform_observation
            obs=observe_policy(env,True,use_risk)
            newer=transform_observation(obs,env.previous_actions,profile)
            a,logp=net.action_pair(obs,newer)
        elif controller in ("priority_mpc","priority_mpc_no_risk","persistent_mpc"):
            obs=priority_observe(env,use_risk,profile)
            a=priority_choose(obs,env.previous_actions,controller=="persistent_mpc" or profile=="persistent")
            logp=np.zeros(5)
        elif controller in ("ppo", "ppo_sample", "ppo_no_risk", "ppo_sample_no_risk", "ppo_guarded","priority_ppo","priority_sample","priority_ppo_guarded"):
            obs = priority_observe(env,use_risk,profile) if controller.startswith("priority_") else observe_policy(env, True, use_risk)
            a, logp = action(
                net, obs, controller not in ("ppo_sample", "ppo_sample_no_risk","priority_sample"),
                max_teacher_regret=(.15 if controller=="priority_ppo_guarded" else (ck or {}).get("guard_regret",.35)) if controller in ("ppo_guarded","priority_ppo_guarded") else None,
            )
        else:
            raise ValueError(controller)
        decision_ms.append((time.perf_counter() - tick) * 1000)
        if record:
            if obs is None:
                obs = observe_policy(env, True)
            state = training_state(env, use_risk=use_risk)
            with torch.no_grad():
                v = (
                    float(net.value(torch.from_numpy(state)))
                    if net is not None
                    else 0.0
                )
            rows.append((*obs[:3], obs[3].astype(np.float32), state, a.copy(), logp, v))
        actions.append(a.copy())
        env.step(indices_to_correction(a))
        commands.append(indices_to_correction(a)[3:].copy())
        applied.append(env.history[-1]['correction'][3:].copy())
        switches.append(a!=last_request);last_request=a.copy()
        motion.append(env.city.points[env.node_log[-1]][3:]-env.city.points[env.node_log[-2]][3:])
        env.previous_actions=np.zeros(5,int) if env.history[-1]["protected"] else a.copy()
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
    result["source_version"] = "communications-deployment-v3.1"
    result["teacher_profile"]=profile if controller.startswith("priority_") or controller in ("persistent_mpc","consensus_ppo") else "v2"
    result["continuity_sources"]={k:v for k,v in sources.items() if k not in ("cooperative/deployment_run.py","cooperative/deployment_policy.py","cooperative/deployment.py")} if sources else None
    result["deployment_sources"]=sources
    result["guard_regret"]=.15 if controller=="priority_ppo_guarded" else None
    result['command_switch_count']=int(np.asarray(switches).sum())
    result['relay_stationary_fraction']=float((np.linalg.norm(motion,axis=-1)<1e-9).mean())
    result["risk_input_enabled"] = bool(use_risk and controller not in ("nominal", "reactive"))
    result["decision_p95_ms"] = float(np.percentile(decision_ms, 95))
    if ck:
        result["training_iteration"] = ck.get("iteration")
    if checkpoint:
        from swarm_sim.regions import sha256_file, ROOT

        result["checkpoint_sha256"] = sha256_file(checkpoint)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        g=int(env.packets.ic[0]);born=env.packets.born[:g]
        bounds=np.searchsorted(born,np.arange(env.k+1)*3.-1e-9)
        delivered=env.packets.status[:g]==1
        weighted=delivered*np.minimum(1.,.25/np.maximum(env.packets.delays[:g],.005))
        cumulative=np.column_stack([np.arange(g+1),np.r_[0,np.cumsum(delivered)],np.r_[0,np.cumsum(weighted)]])
        cohorts=cumulative[bounds[1:]]-cumulative[bounds[:-1]]
        if int(cohorts[:,0].sum())!=g or int(cohorts[:,1].sum())!=result['delivered_packets']:
            raise ValueError('Birth cohort packet conservation failed')
        arrays = {
            'packet_cohorts':cohorts,
            'nominal_search_targets':np.asarray(search_targets),
            'phase_before_action':np.asarray(phases),
            'requested_corrections':np.asarray(commands),
            'applied_corrections':np.asarray(applied),
            'command_switches':np.asarray(switches),
            'relay_motion':np.asarray(motion),
            "reward_terms":np.stack([env.settled[k]["reward_terms"] for k in range(env.k)]),
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

