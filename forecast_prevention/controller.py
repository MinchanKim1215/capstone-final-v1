"""First-relay prevention from received plans and measured radio only."""
from dataclasses import dataclass

import numpy as np

from adaptive_joint_expanded.features import OFFSETS, THRESHOLD, widest
from forecast_graph.codec import at, unpack

HORIZONS = np.arange(9, dtype=float)


@dataclass
class RadioState:
    paths: np.ndarray
    known: np.ndarray
    age: np.ndarray
    commitment: np.ndarray
    power: np.ndarray
    noise: np.ndarray
    noise_trend: np.ndarray
    radio_trend_available: bool
    queue_watch: bool
    local_rf_available: bool


class Inbox:
    """Use bytes delivered to the first relay and its own telemetry row."""

    def __init__(self, n, n_search):
        self.n = n
        self.observer = n_search
        self.messages = [None] * n
        self.previous_noise = np.full(n, np.nan)
        self.previous_time = None
        self.previous_queue = 0.

    def receive(self, now, packets):
        for packet in packets:
            message = unpack(packet)
            if (message['n'] != self.n or message['ns'] != self.observer or
                    message['stamp'] > now or message['issued'] > message['stamp']):
                raise ValueError('Invalid received navigation message')
            peer = message['sender']
            old = self.messages[peer]
            if old is None or old['stamp'] <= message['stamp']:
                self.messages[peer] = message

    def state(self, now, values, masks):
        own = self.observer
        paths = np.zeros((self.n, len(HORIZONS), 2))
        age = np.zeros(self.n)
        commitment = np.zeros(self.n)
        known = np.zeros(self.n, bool)
        power = np.full((self.n, self.n), np.nan)
        measured_noise = np.full(self.n, np.nan)
        for peer, message in enumerate(self.messages):
            if message is None or peer == own:
                continue
            age[peer] = now - message['stamp']
            if age[peer] > 3:
                continue
            known[peer] = True
            paths[peer] = at(message, now + HORIZONS)
            commitment[peer] = max(0., message['commit_until'] - now)
            fresh = (message['report_age'] + age[peer] <= 3.) & np.isfinite(message['power'])
            power[fresh, peer] = message['power'][fresh]
            if fresh.any():
                measured_noise[peer] = np.median(message['power'][fresh] - message['sinr'][fresh])
        paths[own] = values[0, 12:14]
        known[own] = bool(masks[0, 12:14].all())
        local_rf = masks[:, 28] & masks[:, 29] & masks[:, 30] & (values[:, 30] <= 2.)
        local_rf[own] = False
        power[local_rf, own] = values[local_rf, 28]
        if local_rf.any():
            measured_noise[own] = np.median(values[local_rf, 28] - values[local_rf, 29])
        measured_noise = np.clip(measured_noise, -107., -50.)
        trend = np.zeros(self.n)
        if self.previous_time is not None:
            usable = np.isfinite(measured_noise) & np.isfinite(self.previous_noise)
            trend[usable] = np.clip((measured_noise[usable] - self.previous_noise[usable]) /
                                    (now - self.previous_time), -2., 2.)
        radio_trend_available = bool(np.any((measured_noise >= -104.) & (trend >= .6)))
        queue = float(values[0, 11]) if masks[0, 11] else 0.
        queue_watch = queue >= 4 and queue - self.previous_queue >= 2
        self.previous_queue = queue
        self.previous_time = now
        self.previous_noise = measured_noise
        return RadioState(paths, known, age, commitment, power,
                          np.nan_to_num(measured_noise, nan=-107.), trend,
                          radio_trend_available, queue_watch, bool(local_rf.any()))


def spatial_loss(city, scenario, tx, rx):
    tx, rx = np.broadcast_arrays(tx, rx)
    shape = tx.shape[:-1]
    tx = np.c_[tx.reshape(-1, 2), np.full(tx.size // 2, 50.)]
    rx = np.c_[rx.reshape(-1, 2), np.full(rx.size // 2, 50.)]
    return city.radio.propagation(tx, rx, scenario)[0].reshape(shape)


def own_paths(city, position, nominal, actions):
    start = city.nearest(position)
    targets = city.tree.query(np.clip(nominal + OFFSETS[actions], -195., 195.))[1]
    nodes = [np.full(len(actions), start, int)]
    for _ in range(3):
        next_nodes = city.predecessor[targets, nodes[-1]]
        nodes.append(np.where((next_nodes < 0) | (nodes[-1] == targets), nodes[-1], next_nodes))
    knots = city.points[np.stack(nodes, axis=1)]
    left, fraction = np.divmod(HORIZONS.astype(int), 3)
    fraction = fraction[None, :, None] / 3
    return knots[:, left] * (1 - fraction) + knots[:, left + 1] * fraction


class Forecast:
    """Expected bottleneck margin in dB for each candidate and future second."""

    def __init__(self, city, scenario, state, observer):
        self.city = city
        self.scenario = scenario
        self.state = state
        self.observer = observer
        self.paths = state.paths.transpose(1, 0, 2)
        self.loss = spatial_loss(city, scenario, self.paths[:, :, None], self.paths[:, None, :])
        self.shadow = np.nan_to_num(np.clip(state.power - (10. - self.loss[0]), -6., 6.))
        self.uncertainty = .2 * np.maximum(0., HORIZONS[:, None] - state.commitment) + .25 * state.age
        self.uncertainty[:, observer] = .15 * HORIZONS
        self.uncertainty[0] = 0.

    def margins(self, own, radio_trend=False):
        i = self.observer
        state = self.state
        loss = np.broadcast_to(self.loss, (len(own), *self.loss.shape)).copy()
        replacement = spatial_loss(self.city, self.scenario, own[:, :, None], self.paths[None])
        loss[:, :, i, :] = replacement
        loss[:, :, :, i] = replacement
        noise = np.broadcast_to(state.noise, (len(HORIZONS), len(state.noise)))
        if radio_trend:
            # Only positive measured deterioration is projected, with a cap.
            increment = np.minimum(8., HORIZONS[:, None] * np.maximum(state.noise_trend, 0.))
            noise = np.minimum(-50., noise + increment)
        directed = 10. - loss + self.shadow * np.exp(-HORIZONS[:, None, None]/4) - noise[:, None, :] - THRESHOLD
        margin = np.minimum(directed, directed.swapaxes(-2, -1))
        margin -= np.maximum(self.uncertainty[:, :, None], self.uncertainty[:, None, :])
        margin[..., ~state.known, :] = -50.
        margin[..., :, ~state.known] = -50.
        ids = np.arange(len(state.known))
        margin[..., ids, ids] = 30.
        return widest(margin, i).min(-1)


def predicted_onset(margin):
    down = margin[1:] < 0.
    indices = np.flatnonzero(down[:-1] & down[1:])
    return int(indices[0] + 1) if len(indices) else None


def decide(arm, city, scenario, state, observer, nominal, base_action, action_logits, valid):
    """Return the action and the causal evidence passed to the decision rule."""
    if not state.local_rf_available or not state.known[:observer+1].all():
        return base_action, dict(reason='insufficient_observation')
    if valid.sum() <= 1:
        return base_action, dict(reason='no_alternative')
    forecast = Forecast(city, scenario, state, observer)
    original_path = own_paths(city, state.paths[observer, 0], nominal, [base_action])
    original_margin = forecast.margins(original_path)[0]
    current = float(original_margin[0])
    recovery = current < 0.
    if arm == 'reactive' and not recovery:
        return base_action, dict(reason='connected_reactive_guard')
    use_radio = arm == 'switch' and state.radio_trend_available
    if use_radio:
        original_margin = forecast.margins(original_path, radio_trend=True)[0]
    onset = predicted_onset(original_margin)
    expert = 'radio_trend' if use_radio else 'geometry'
    detail = dict(expert=expert, observed_current_margin_db=current,
                  predicted_onset_in_s=onset, estimated_connected=not recovery,
                  original_margin_db=original_margin.tolist())
    if onset is None and (not recovery or (original_margin[1:4] >= 0.).all()):
        return base_action, dict(reason='queue_watch' if arm == 'switch' and state.queue_watch else 'no_predicted_cut', **detail)
    ranked = np.argsort(-action_logits, kind='stable')
    actions = list(dict.fromkeys([base_action, 0] + [int(k) for k in ranked if valid[k]][:8]))
    actions = np.asarray([k for k in actions if valid[k]], dtype=int)
    paths = own_paths(city, state.paths[observer, 0], nominal, actions)
    margins = forecast.margins(paths, radio_trend=use_radio)
    expected_cost = np.maximum(0., .5 - margins[:, 1:]).sum(axis=1)
    expected_cost += .025 * np.linalg.norm(OFFSETS[actions] - OFFSETS[base_action], axis=1)
    distinct_first_move = np.linalg.norm(paths[:, 3] - paths[0, 3], axis=1) > 1e-8
    early_safe = (margins[:, 1:3] >= np.minimum(margins[0, 1:3], 0.) - .3).all(axis=1)
    candidates = np.flatnonzero(distinct_first_move & early_safe)
    if not len(candidates):
        return base_action, dict(reason='no_reachable_improvement', **detail)
    best = candidates[np.argmin(expected_cost[candidates])]
    gain = float(expected_cost[0] - expected_cost[best])
    old_deficit = int(np.sum(margins[0, 1:] < 0))
    new_deficit = int(np.sum(margins[best, 1:] < 0))
    if gain < 2. or (not recovery and new_deficit >= old_deficit):
        return base_action, dict(reason='insufficient_expected_gain', **detail)
    return int(actions[best]), dict(reason='recovery' if recovery else 'prevention',
                                     alternative_margin_db=margins[best].tolist(),
                                     estimated_cost_reduction=gain,
                                     projected_own_endpoint=paths[best, 3].tolist(), **detail)
