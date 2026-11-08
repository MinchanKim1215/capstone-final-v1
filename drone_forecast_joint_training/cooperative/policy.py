"""공유 중계 정책과 학습 전용 중앙 가치함수.

실행 정책에는 local_candidates가 만든 수신 관측과 정적 지도 특징만 전달.
중앙 가치함수의 실제 채널 정보는 학습 손실에만 사용.
"""

from pathlib import Path
import numpy as np
import torch
from torch import nn

CONTEXT_DIM = 91
CANDIDATE_DIM = 12
STATE_DIM = 224


def critic_state(env, use_risk=True):
    """학습용 전역 상태. actor.forward에 전달 금지."""
    pos = env.city.points[env.nodes] / 200.0
    delta = (env.proposal["nominal"] - env.city.points[env.nodes]) / 40.0
    p = np.clip(env.p_now, 0, 1).ravel()
    r = np.nan_to_num(env.risk, nan=0.0).ravel() if use_risk else np.zeros(64)
    q = env.packets.queue_now / 64.0
    v = np.r_[
        pos.ravel(),
        delta.ravel(),
        p,
        r,
        q,
        env.k * 3 / 3000,
        env.proposal["phase"] / 2,
        env.proposal["observed_cells"] / max(env.proposal["observable_cells"], 1),
    ]
    # 16+16+64+64+8+3=171; 고정 차원 명세로 패딩.
    result = np.zeros(STATE_DIM, np.float32)
    result[: len(v)] = v
    return result


class SharedActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh()
        )
        self.score = nn.Sequential(
            nn.Linear(32 + CANDIDATE_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        # 명목/간단 MPC를 학습 초기부터 포함하는 신뢰 가능한 잔차 정책.
        # 처음부터 같은 척도로 행동을 수집. 이후 변경도 PPO 경사 갱신만 사용.
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, context, candidates, mask, teacher=None):
        z = self.context(context)
        z = z.unsqueeze(-2).expand(*candidates.shape[:-1], z.shape[-1])
        residual = self.score(torch.cat((z, candidates), -1)).squeeze(-1)
        # teacher는 미래 정답이 아닌 동일한 로컬 후보의 모델 기반 점수.
        # 고정 특징 skip을 두되 전체 제어 가중치는 PPO로 갱신 가능.
        if teacher is None:
            b3, b6, mean6 = (
                candidates[..., 0] * 10,
                candidates[..., 1] * 10,
                candidates[..., 3] * 10,
            )
            teacher = (
                0.65 * b3.clamp(-10, 6)
                + 0.35 * b6.clamp(-10, 6)
                + 0.1 * mean6.clamp(-10, 6)
                + 1.5 * candidates[..., 5]
                - 0.16 * candidates[..., 6]
                - 0.03 * candidates[..., 8]
            )
        logits = self.residual_scale.clamp(0.01, 4) * residual + teacher
        return logits.masked_fill(~mask, -1e9)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = SharedActor()
        self.critic = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def value(self, state):
        return self.critic(state).squeeze(-1)


class EnsembleActorCritic(nn.Module):
    """독립적으로 학습한 같은 관측 계약 정책의 동일 가중치 logit 평균."""

    def __init__(self, states):
        super().__init__()
        self.members = nn.ModuleList([ActorCritic() for _ in states])
        for model, state in zip(self.members, states):
            model.load_state_dict(state)

    def actor(self, *obs):
        return torch.stack([model.actor(*obs) for model in self.members]).mean(0)

    def value(self, state):
        return torch.stack([model.value(state) for model in self.members]).mean(0)


def load_policy(path):
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "ensemble_models" in ck:
        net = EnsembleActorCritic(ck["ensemble_models"])
    else:
        net = ActorCritic()
        net.load_state_dict(ck["model"])
    net.eval()
    return net, ck


def action(net, obs, deterministic=True, max_teacher_regret=None):
    c, f, m, teacher = obs
    with torch.no_grad():
        logits = net.actor(
            torch.as_tensor(c),
            torch.as_tensor(f),
            torch.as_tensor(m),
            torch.as_tensor(teacher, dtype=torch.float32),
        )
        distribution = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else distribution.sample()
        if max_teacher_regret is not None:
            if not deterministic or not np.isfinite(max_teacher_regret) or max_teacher_regret < 0:
                raise ValueError("모델 점수 가드는 결정적 실행의 0 이상 유한한 한계 필요")
            teacher = torch.as_tensor(obs[3], dtype=torch.float32).masked_fill(~torch.as_tensor(obs[2]), -1e9)
            best = teacher.argmax(-1)
            regret = teacher.gather(-1, best[:, None]).squeeze(-1) - teacher.gather(-1, a[:, None]).squeeze(-1)
            a = torch.where(regret > max_teacher_regret, best, a)
    return a.numpy(), distribution.log_prob(a).numpy()
