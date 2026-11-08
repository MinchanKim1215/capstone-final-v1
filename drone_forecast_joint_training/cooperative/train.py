"""실제 시뮬레이션 상호작용을 사용하는 중앙 학습·로컬 실행 PPO.

정책의 고정 MPC 점수는 현재 수신 관측으로 계산. 정답/미래 간섭 사용 금지.
동일한 지도를 여러 번 보는 학습과 미관측 지역 평가를 명확히 분리.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, json, time, random, os, hashlib
import numpy as np
import torch
from torch import nn
from cooperative.policy import ActorCritic
from cooperative.run import episode
from swarm_sim.regions import ROOT
from swarm_sim.regions import sha256_file
from cooperative.io import validated_result


def worker(item):
    spec, ck, path = item
    torch.set_num_threads(1)
    # 패킷/관측 난수와 별개의 행동 표본 시드.
    torch.manual_seed(int(spec["seed"]) + 9173)
    meta = Path(path).with_suffix(".json")
    previous = validated_result(path, spec, sha256_file(ck), rollout=True)
    if previous is not None:
        return previous
    if meta.exists() or Path(path).with_suffix(".npz").exists():
        abandoned = ROOT / "results/cooperative/abandoned_rollouts"
        abandoned.mkdir(parents=True, exist_ok=True)
        for file in (meta, Path(path).with_suffix(".npz")):
            if file.exists():
                file.replace(abandoned / file.name)
    result = episode(spec, "ppo_sample", ck, path, record=True)
    result["checkpoint_sha256"] = sha256_file(ck)
    meta.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def gae(rewards, values, gamma=0.995, lam=0.95):
    advantage = np.zeros_like(rewards)
    last = 0.0
    next_value = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantage[t] = last
        next_value = values[t]
    return advantage, advantage + values


def update(net, optim, paths, seed, epochs=6, clip=0.15, warmup=False, target_kl=0.02):
    # 행동 수집과 첫 PPO 갱신 사이에 정책 파라미터를 바꾸지 않는다.
    # 특히 잔차 크기의 외부 보정은 저장된 old log-probability를 무효화한다.
    data = {
        k: []
        for k in (
            "context",
            "candidates",
            "mask",
            "teacher",
            "state",
            "chosen",
            "logp",
            "value",
            "advantage",
            "returns",
        )
    }
    for path in paths:
        with np.load(path) as z:
            a, r = gae(z["rewards"], z["value"])
            for key in data:
                data[key].append(
                    a if key == "advantage" else r if key == "returns" else z[key]
                )
    ten = {k: torch.as_tensor(np.concatenate(v)) for k, v in data.items()}
    n = len(ten["state"])
    if n < 2:
        raise ValueError("PPO 갱신에는 두 개 이상의 전이 필요")
    initial_logp_max_abs = assert_rollout_policy(net, ten)
    ten["advantage"] = (ten["advantage"] - ten["advantage"].mean()) / (
        ten["advantage"].std() + 1e-6
    )
    # 값 손실은 배치 반환 규모로 정규화. actor의 보상 정의는 변경하지 않음.
    value_scale = max(float(ten["returns"].std()), 1.0)
    rng = np.random.default_rng(seed)
    stats = []
    net.train()
    for epoch in range(epochs):
        for begin in range(0, n, 256):
            if begin == 0:
                permutation = rng.permutation(n)
            idx = permutation[begin : begin + 256]
            c, f, m, t, actions = [
                ten[k][idx]
                for k in ("context", "candidates", "mask", "teacher", "chosen")
            ]
            logits = net.actor(c, f, m, t)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)
            ratio = (logp - ten["logp"][idx]).exp()
            advantage = ten["advantage"][idx, None]
            policy = -torch.minimum(
                ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
            ).mean()
            v = net.value(ten["state"][idx])
            target = ten["returns"][idx]
            value = ((v - target) / value_scale).square().mean()
            entropy = dist.entropy().mean()
            # 초기 지도 기반 정책에서의 급격한 이탈을 억제하는 약한 분포 규제.
            anchor = torch.distributions.Categorical(logits=t.masked_fill(~m, -1e9))
            kl = torch.distributions.kl_divergence(dist, anchor).mean()
            loss = policy + 0.5 * value - 0.003 * entropy + 0.003 * kl
            if not torch.isfinite(loss):
                raise FloatingPointError("유한하지 않은 PPO 손실")
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optim.step()
            stats.append(
                [
                    float(policy.detach()),
                    float(value.detach()),
                    float(entropy.detach()),
                    float(kl.detach()),
                    float(((ratio - 1) - ratio.log()).mean().detach()),
                ]
            )
        # 지나친 정책 변경 시 해당 반복의 나머지 epoch 중단.
        if np.mean(np.asarray(stats)[-max(1, (n + 255) // 256) :, -1]) > target_kl:
            break
    net.eval()
    mean = np.mean(stats, axis=0)
    return dict(
        policy_loss=mean[0],
        value_loss=mean[1],
        entropy=mean[2],
        anchor_kl=mean[3],
        approx_kl=mean[4],
        residual_scale=float(net.actor.residual_scale.detach()),
        learning_rate=optim.param_groups[0]["lr"],
        samples=n,
        epochs=epoch + 1,
        reward_return_mean=float(ten["returns"].mean()),
        value_scale=value_scale,
        initial_logp_max_abs=initial_logp_max_abs,
    )


def assert_rollout_policy(net, data, tolerance=2e-5):
    """모든 수집 행동의 old logp와 현재 정책을 갱신 전에 대조한다."""
    largest = 0.0
    with torch.no_grad():
        for begin in range(0, len(data["state"]), 512):
            ix = slice(begin, begin + 512)
            logits = net.actor(*(data[k][ix] for k in ("context", "candidates", "mask", "teacher")))
            actual = torch.distributions.Categorical(logits=logits).log_prob(data["chosen"][ix])
            largest = max(largest, float((actual - data["logp"][ix]).abs().max()))
    if not np.isfinite(largest) or largest > tolerance:
        raise RuntimeError(
            f"수집 정책과 갱신 시작 정책 불일치: 최대 logp 차이 {largest:.8g}. "
            "정책을 복원하고 해당 정책으로 행동을 다시 수집해야 합니다."
        )
    return largest


def save(path, net, optim, iteration, history, seed, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "format": "cooperative-shared-ppo-v1",
        "model": net.state_dict(),
        "optimizer": optim.state_dict(),
        "iteration": iteration,
        "history": history,
        "seed": seed,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "algorithm": "centralized-critic parameter-sharing PPO, model-based local score + learned residual",
        "forecast_checkpoint": "model/advance/selected/best.pt",
        "forecast_frozen": True,
        "future_truth_in_actor": False,
        "candidate_count": 25,
        "context_dim": 91,
        "candidate_dim": 12,
    }
    ck.update(metadata)
    tmp = path.with_suffix(".tmp")
    torch.save(ck, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--batch-episodes", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026102607)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model_dir = ROOT / "model/cooperative"
    model_dir.mkdir(parents=True, exist_ok=True)
    from cooperative.locking import TrainingLock

    lock = TrainingLock(model_dir / "training.lock")
    net = ActorCritic()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-5)
    history = []
    start = 0
    if (model_dir / "last.pt").exists() and not args.resume:
        raise ValueError(
            "기존 학습 가중치 존재. --resume으로 이어서 학습하거나 별도 프로젝트 복사본 사용."
        )
    if args.resume and (model_dir / "last.pt").exists():
        ck = torch.load(model_dir / "last.pt", map_location="cpu", weights_only=False)
        if ck["seed"] != args.seed or (
            ck["history"] and ck["history"][-1]["episodes"] != args.batch_episodes
        ):
            raise ValueError(
                "체크포인트의 난수 시드와 반복당 실행 수를 유지해야 재개 가능"
            )
        net.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])
        history = ck["history"]
        start = ck["iteration"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
    else:
        save(model_dir / "initial.pt", net, optim, 0, [], args.seed)
    for group in optim.param_groups:
        group["lr"] = 1e-3
    protocol = json.loads(
        (ROOT / "configs/usable_protocol.json").read_text(encoding="utf-8")
    )
    maps = sorted(e["key"] for e in protocol["maps"] if e["split"] == "train")
    # 순서는 독립 난수로 고정. 검증·시험 지도는 학습 목록에서 제외.
    schedule = np.random.default_rng(args.seed).permutation(maps).tolist()
    began = time.perf_counter()
    for iteration in range(start, args.iterations):
        ckpath = model_dir / f"iteration_{iteration:02d}.pt"
        if ckpath.exists():
            existing = torch.load(ckpath, map_location="cpu", weights_only=False)
            if existing["iteration"] != iteration or any(
                not torch.equal(v, existing["model"][k])
                for k, v in net.state_dict().items()
            ):
                raise ValueError(
                    "같은 반복 번호의 기존 정책이 현재 재개 상태와 다름. 덮어쓰기 금지."
                )
        else:
            save(ckpath, net, optim, iteration, history, args.seed)
        jobs = []
        paths = []
        for j in range(args.batch_episodes):
            k = iteration * args.batch_episodes + j
            cycle = k // len(schedule)
            spec = {
                "region": schedule[k % len(schedule)],
                "seed": args.seed + 10000 + k * 101,
                "kind": ("continuous", "intermittent", "clear")[
                    (k + (cycle if iteration >= 3 else 0)) % 3
                ],
                "direction": ("east_west", "north_south")[
                    (k // 3 + (cycle if iteration >= 3 else 0)) % 2
                ],
            }
            path = ROOT / "results/cooperative/training" / f"i{iteration:02d}_e{j:02d}"
            jobs.append((spec, str(ckpath), str(path)))
            paths.append(path.with_suffix(".npz"))
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, item): item for item in jobs}
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                print(
                    json.dumps(
                        {
                            "stage": "rollout",
                            "iteration": iteration,
                            "region": r["region"],
                            "C": r["search_connected_mean"],
                            "pdr": r["pdr"],
                            "duration_ratio": r["duration_ratio"],
                            "complete": r["mission_completed"],
                            "wall_s": r["wall_s"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        for group in optim.param_groups:
            group["lr"] = 3e-4 if iteration == 0 else 1e-3
        stats = update(net, optim, paths, args.seed + iteration, warmup=iteration == 0)
        stats.update(
            iteration=iteration + 1,
            episodes=len(results),
            connection_mean=float(
                np.mean([r["search_connected_mean"] for r in results])
            ),
            pdr_mean=float(np.mean([r["pdr"] for r in results])),
            complete_count=sum(r["mission_completed"] for r in results),
            elapsed_s=time.perf_counter() - began,
        )
        history.append(stats)
        save(model_dir / "last.pt", net, optim, iteration + 1, history, args.seed)
        save(
            model_dir / f"iteration_{iteration+1:02d}.pt",
            net,
            optim,
            iteration + 1,
            history,
            args.seed,
        )
        (ROOT / "results/cooperative/training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps({"stage": "update", **stats}), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
