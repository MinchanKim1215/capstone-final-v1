"""Update PPO and both neural prediction heads on fresh half-second rollouts."""

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from adaptive_joint_expanded.contracts import HISTORY
from adaptive_joint_expanded.targets import timing_labels
from fast_halfsec.ppo import GAMMA_TICK, LAMBDA_TICK
from final_v1.transformer_upgrade.capture import label_network
from forecast_rebuild.contracts import sha

from .model import OBSERVATIONS


POLICY_INPUTS = ('states', 'options', 'priors', 'valid')
NETWORK_INPUTS = ('geometry', 'support', 'route', 'native')
HISTORY_INPUTS = ('values', 'masks', 'intents')


def advantages(frames, reward, values):
    result = np.zeros(len(frames), np.float32)
    running = 0.
    stops = np.r_[frames[1:], len(reward)]
    for i in reversed(range(len(frames))):
        start, stop = int(frames[i]), int(stops[i])
        ticks = stop - start
        if ticks <= 0:
            raise ValueError('The policy recorded an invalid decision interval')
        total = reward[start:stop] @ (GAMMA_TICK**np.arange(ticks))
        following = values[i+1] if i+1 < len(frames) else 0.
        delta = total + GAMMA_TICK**ticks * following - values[i]
        running = delta + (GAMMA_TICK * LAMBDA_TICK)**ticks * running
        result[i] = running
    return result


class Trace:
    def __init__(self, row):
        if sha(row['archive_path']) != row['archive_sha256']:
            raise ValueError('A joint rollout was changed')
        with np.load(row['archive_path'], allow_pickle=False) as arrays:
            self.data = {key: arrays[key] for key in (*POLICY_INPUTS, *NETWORK_INPUTS,
                         'action_samples', 'actions', 'logp', 'plan_frames')}
            self.history = {}
            for key in HISTORY_INPUTS:
                history = arrays['history_' + key]
                self.history[key] = np.pad(history, ((HISTORY-1, 0), *((0, 0),) * (history.ndim - 1)))
            frames = self.data['plan_frames']
            if len(frames) and frames.min() < 1:
                raise ValueError('A planning observation preceded the first RF frame')
            self.shape = self.data['support'].shape[1:]
            nr, n = self.shape
            links, valid = timing_labels(arrays['links'], frames, n - nr)
            self.link_y = np.where(valid & self.data['support'], links, -1)
            self.fleet_y = label_network(arrays['network'], row['cuts'], frames * .5)
            if len(self.data['actions']):
                np.testing.assert_array_equal(arrays['action_frames'], frames[self.data['action_samples']] * 5)
                self.advantage = advantages(arrays['action_frames'], arrays['rewards'], arrays['values'])
                self.returns = self.advantage + arrays['values']
            else:
                self.advantage = self.returns = np.empty(0, np.float32)

    def observation(self, index, key):
        if key not in self.history:
            return self.data[key][index]
        end = self.data['plan_frames'][index]
        window = self.history[key][end-1:end+HISTORY-1]
        return window.transpose(1, 2, 0, 3)

    def supervision(self, rng):
        known = self.data['support'].any(axis=(1, 2))
        valid = np.flatnonzero(known & (self.fleet_y >= 0))
        if not len(valid):
            return []
        groups = [rng.choice(valid, min(96, len(valid)), replace=False)]
        for target in (1, 2, 3):
            positive = valid[self.fleet_y[valid] == target]
            groups.append(rng.choice(positive, min(12, len(positive)), replace=False))
        return [(self, int(i)) for i in np.unique(np.concatenate(groups))]


def stack(items, accessor, device):
    return torch.as_tensor(np.stack([accessor(trace, index) for trace, index in items]), device=device)


def forecast(model, items, device):
    return model.forecast(*(stack(items, lambda t, i: t.observation(i, key), device)
                            for key in OBSERVATIONS))


def evaluate_actions(model, actions, device):
    samples = [(trace, trace.data['action_samples'][index]) for trace, index in actions]
    context = forecast(model, samples, device)[2]
    inputs = [stack(actions, lambda t, i: t.data[key][i], device) for key in POLICY_INPUTS]
    logits, value = model.policy(*inputs, context)
    distribution = torch.distributions.Categorical(logits=logits)
    choices = stack(actions, lambda t, i: t.data['actions'][i], device).long()
    old = stack(actions, lambda t, i: t.data['logp'][i], device).float()
    return distribution, choices, old, value, inputs[2].masked_fill(~inputs[3], -1e9)


def optimizer_for(model):
    return torch.optim.AdamW([
        dict(params=model.forecaster.backbone.encoder_parameters(), lr=6e-6),
        dict(params=model.forecaster.timing.parameters(), lr=2e-5),
        dict(params=model.fleet.parameters(), lr=3e-5),
        dict(params=model.moves.parameters(), lr=2e-4),
    ], eps=1e-5)


def optimize(model, optimizer, rows, device, update, seed):
    model.eval()  # The behavior policy uses deterministic Transformer/fleet features.
    rng = np.random.default_rng(seed + update)
    traces = [Trace(row) for row in rows]
    actors, auxiliary = defaultdict(list), defaultdict(list)
    for trace in traces:
        actors[trace.shape].extend((trace, i) for i in range(len(trace.advantage)))
        auxiliary[trace.shape].extend(trace.supervision(rng))
    decisions = sum(len(items) for items in actors.values())
    labels = sum(len(items) for items in auxiliary.values())
    if not decisions and not labels:
        return dict(decisions=0, supervision_samples=0, skipped=True, device=str(device))
    all_advantage = np.concatenate([trace.advantage for trace in traces])
    all_returns = np.concatenate([trace.returns for trace in traces])
    mean = float(all_advantage.mean()) if decisions else 0.
    scale = max(1e-7, float(all_advantage.std())) if decisions else 1.
    value_scale = max(.1, float(all_returns.std())) if decisions else 1.
    batch_size = 24
    replay_error = 0.
    with torch.no_grad():
        for items in actors.values():
            for start in range(0, len(items), batch_size):
                d, choice, old, _, _ = evaluate_actions(model, items[start:start+batch_size], device)
                replay_error = max(replay_error, float((d.log_prob(choice) - old).abs().max()))
    if replay_error > 5e-4:
        raise ValueError(f'PPO log probability differs from the simulated behavior: {replay_error:.7f}')

    modules = {'encoder': model.forecaster.backbone, 'timing': model.forecaster.timing,
               'fleet': model.fleet, 'ppo': model.moves}
    before = {name: [p.detach().clone() for p in part.parameters()]
              for name, part in modules.items()}
    totals, actor_losses, link_losses, fleet_losses = [], [], [], []
    ppo_to_timing = 0.
    for _ in range(2):
        for shape in sorted(set(actors) | set(auxiliary)):
            actions, samples = actors[shape].copy(), auxiliary[shape].copy()
            rng.shuffle(actions)
            rng.shuffle(samples)
            for start in range(0, max(len(actions), len(samples)), batch_size):
                moves, examples = actions[start:start+batch_size], samples[start:start+batch_size]
                loss = torch.zeros((), device=device)
                if moves:
                    d, choice, old, value, priors = evaluate_actions(model, moves, device)
                    adv = stack(moves, lambda t, i: t.advantage[i], device)
                    adv = (adv - mean) / scale
                    returns = stack(moves, lambda t, i: t.returns[i], device)
                    ratio = (d.log_prob(choice) - old).exp()
                    objective = -torch.minimum(adv * ratio, adv * ratio.clamp(.85, 1.15)).mean()
                    critic = ((value - returns) / value_scale).square().mean()
                    prior = torch.distributions.Categorical(logits=priors)
                    loss = objective + .5 * critic + .03 * torch.distributions.kl_divergence(d, prior).mean()
                    loss = loss - .005 * d.entropy().mean()
                    if ppo_to_timing == 0. and objective.requires_grad:
                        grad, = torch.autograd.grad(objective, model.forecaster.timing.weight,
                                                    allow_unused=True, retain_graph=True)
                        if grad is not None:
                            ppo_to_timing = float(grad.norm())
                    actor_losses.append(float(objective.detach()))
                if examples:
                    timing, fleet, _ = forecast(model, examples, device)
                    link_y = stack(examples, lambda t, i: t.link_y[i], device).long().reshape(-1)
                    fleet_y = stack(examples, lambda t, i: t.fleet_y[i], device).long()
                    link = F.cross_entropy(timing.reshape(-1, 17), link_y, ignore_index=-1,
                                           reduction='sum') / (link_y >= 0).sum().clamp_min(1)
                    network = F.cross_entropy(fleet, fleet_y)
                    loss = loss + .18 * link + .30 * network
                    link_losses.append(float(link.detach()))
                    fleet_losses.append(float(network.detach()))
                if not torch.isfinite(loss):
                    raise ValueError('Joint loss became nonfinite')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not torch.isfinite(clip_grad_norm_(model.parameters(), .5)):
                    raise ValueError('Joint gradient became nonfinite')
                optimizer.step()
                totals.append(float(loss.detach()))
    changed = {name: sum(float((new.detach() - old).square().sum())
                         for new, old in zip(part.parameters(), before[name]))**.5
               for name, part in modules.items()}
    expected = (('ppo',) if decisions else ()) + (('encoder', 'timing', 'fleet') if labels else ())
    if any(changed[name] <= 0. for name in expected):
        raise ValueError('A trainable part of the combined model did not update')
    return dict(decisions=decisions, supervision_samples=labels, logp_error=replay_error,
                mean_loss=float(np.mean(totals)), ppo_loss=float(np.mean(actor_losses)) if actor_losses else None,
                link_loss=float(np.mean(link_losses)) if link_losses else None,
                fleet_loss=float(np.mean(fleet_losses)) if fleet_losses else None,
                ppo_gradient_to_timing=ppo_to_timing, weight_change_l2=changed, device=str(device))
