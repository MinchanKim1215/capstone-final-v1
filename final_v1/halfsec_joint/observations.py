"""Project received reports and shared navigation onto the installed input schema."""

from types import SimpleNamespace

import numpy as np

from adaptive_joint_expanded.features import THRESHOLD
from adaptive_joint_expanded.observation import Observer
from final_v1.transformer_upgrade.capture import route_features
from forecast_avoidance.controller import TIMES
from forecast_learning.features import COLUMNS, describe
from forecast_shield.forecaster import evaluate_terms, geometry

ROUTE_TIMES = np.r_[TIMES, 9.]


class History(Observer):
    def sample(self, now, nav, sensor, queue):
        ns, n = self.deployment.ns, self.deployment.n
        xy, velocity = nav.motion.positions()
        receiver = np.arange(ns, n)[:, None]
        peer = np.arange(n)[None, :]
        distinct = receiver != peer
        power = sensor.remote_power[peer, receiver]
        sinr = sensor.remote_sinr[peer, receiver]
        stamp = sensor.remote_stamp[peer, receiver]
        reverse = sensor.remote_sinr[receiver, peer]
        reverse_stamp = sensor.remote_stamp[receiver, peer]
        age, reverse_age = now - stamp, now - reverse_stamp
        known = distinct & np.isfinite(power) & np.isfinite(sinr) & (age >= -1e-8)
        reverse_known = distinct & np.isfinite(reverse) & (reverse_age >= -1e-8)
        recent = known & (age <= 2.)
        both = recent & reverse_known & (reverse_age <= 2.)
        x = np.zeros((n - ns, n, 48), np.float32)
        mask = np.zeros_like(x, bool)

        def feature(index, value, valid):
            x[..., index] = np.where(valid, value, 0.)
            mask[..., index] = valid

        feature(0, velocity[receiver, 0], distinct)
        feature(1, velocity[receiver, 1], distinct)
        feature(2, nav.city._cooperative_scenario.altitude_m, distinct)
        feature(3, 1., distinct)
        feature(4, peer >= ns, distinct)
        feature(11, queue, distinct & (receiver == ns))
        feature(12, xy[receiver, 0], distinct)
        feature(13, xy[receiver, 1], distinct)
        feature(28, power, known)
        feature(29, sinr, known)
        feature(30, np.clip(age, 0., 30.), known)
        feature(31, reverse, reverse_known)
        feature(32, np.clip(reverse_age, 0., 30.), reverse_known)
        feature(33, age < .51, distinct)
        relative = xy[peer] - xy[receiver]
        relative_v = velocity[peer] - velocity[receiver]
        distance = np.linalg.norm(relative, axis=-1)
        feature(34, relative[..., 0], recent)
        feature(35, relative[..., 1], recent)
        feature(36, relative_v[..., 0], recent)
        feature(37, relative_v[..., 1], recent)
        feature(38, distance, recent)
        feature(39, (relative * relative_v).sum(-1) / np.maximum(distance, 1.), recent)
        feature(40, np.minimum(sinr, reverse), both)
        feature(41, np.minimum(sinr, reverse) - THRESHOLD, both)
        if len(self.values) >= 6:
            previous, previous_mask = self.values[-6], self.masks[-6]
            feature(42, (sinr - previous[..., 29]) / 3.,
                    recent & previous_mask[..., 29] & (previous[..., 30] <= 2.))
            feature(43, (reverse - previous[..., 31]) / 3.,
                    reverse_known & (reverse_age <= 2.) & previous_mask[..., 31]
                    & (previous[..., 32] <= 2.))
        feature(46, both, distinct)
        # Shared navigation exposes the current reserved edge. Missing packet
        # measurements and unknown peer intents keep their masks unset.
        end = nav.city.points[nav.motion.end] - xy
        intent = np.zeros((n - ns, n, 9), np.float32)
        intent[..., :2] = end[receiver]
        intent[..., 2:4] = np.where(recent[..., None], end[peer], 0.)
        intent[..., 4] = np.where(recent, age, 0.)
        intent[..., 5] = np.minimum(np.linalg.norm(end[receiver], axis=-1) / 5., 3.)
        intent[..., 6] = np.linalg.norm(end[receiver], axis=-1)
        intent[..., 7] = np.where(recent, np.linalg.norm(end[peer], axis=-1), 0.)
        intent[..., 8] = recent
        self.values.append(x.astype(np.float16))
        self.masks.append(mask)
        self.intents.append(intent.astype(np.float16))


def path_block(nav, scenario, xy, state):
    loss, extra = geometry(nav.city, scenario, xy)
    warning = evaluate_terms(loss, extra, state, nav.ns, 0.)
    fields, margins, graphs = describe(xy, loss, extra, state, nav.ns, warning)
    return fields, margins, graphs, warning


def path_inputs(nav, scenario, applied, estimate, history, queue):
    state = SimpleNamespace(power=estimate.power, noise=estimate.noise,
                            known=estimate.measured, local_rf_available=estimate.local_rf)
    held = nav.forecast(applied, applied, commit_frames=0, times=ROUTE_TIMES)
    release = nav.forecast(np.zeros_like(applied), applied, commit_frames=30, times=ROUTE_TIMES)
    first = path_block(nav, scenario, held[:-1], state)
    second = first if np.array_equal(held, release) else path_block(nav, scenario, release[:-1], state)
    moving = (held[6] != held[0]).any(-1)
    received = np.isfinite(state.power[:, nav.ns])
    noise_trend = 0.
    if len(history.values) >= 7:
        prior, mask = history.values[-7][0], history.masks[-7][0]
        good = received & mask[:, 28] & mask[:, 29] & (prior[:, 30] <= 2.)
        if good.any():
            prior_noise = (prior[:, 28] - prior[:, 29])[good]
            noise_trend = float((state.noise[nav.ns] - np.median(prior_noise)) / 3.)
    phase = 2 if nav.finished_at is not None else 0
    prefix = [received.sum(), (~state.known[:nav.ns]).sum(), state.known.sum(),
              state.noise.max(), state.noise.mean(), noise_trend, queue,
              moving[nav.ns:].mean(), moving[:nav.ns].mean(), nav.ns, len(applied), phase]
    distance = np.linalg.norm(held[[12, 18]] - release[[12, 18]], axis=-1).mean(axis=1)
    x = np.r_[prefix, first[0], second[0], distance].astype(np.float32)
    if x.shape != (len(COLUMNS),) or np.isinf(x).any():
        raise ValueError('The deployed LightGBM feature schema changed')
    graphs = np.stack((first[2], second[2]))
    route = route_features(SimpleNamespace(design=SimpleNamespace(n_search=nav.ns)),
                           (x, None, None, graphs, None))
    return x, route, first[3].shield_score, state, release
