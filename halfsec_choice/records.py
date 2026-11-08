"""Replay training telemetry and label pairs with shared radio disturbances."""
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit

from adaptive_joint_expanded.baseline import paired_shadow
from adaptive_joint_expanded.features import OFFSETS, THRESHOLD, widest
from cooperative.packets import hop_probabilities
from fast_halfsec.context import interference, registry, survey_for
from fast_halfsec.controller import options_for, state_at
from fast_halfsec.features import RadioHistory
from fast_halfsec.learning import Predictors
from fast_halfsec.navigation import Navigator
from final_v1_fast.runner import intervals
from final_v1_fast.sensing import Sensors
from forecast_rebuild.contracts import read, sha
from forecast_shield.forecaster import geometry

from . import FORECAST
from .choice import HORIZON, INPUTS, guarded_options

ANCHOR_CAP = 35
METRICS = ('starts', 'cut_seconds', 'margin_cost')


@lru_cache(maxsize=1)
def predictors():
    return Predictors(FORECAST)


@njit(cache=True)
def links_from_loss(loss, noise, shadow):
    p = np.empty_like(loss)
    received, sinr = np.empty_like(loss), np.empty_like(loss)
    for index in range(len(p)):
        p[index] = hop_probabilities(loss[index], noise[index], shadow[index], received[index], sinr[index])
    return p, received, sinr


class Radio:
    def __init__(self, city, scenario, spec, n, ticks, anchor_s):
        self.city, self.scenario = city, scenario
        self.shadow = paired_shadow(spec['seed'], ticks, n, .5, scenario.shadow_sigma_db,
                                    scenario.shadow_correlation_s)
        self.jammer, self.begin, self.end, self.power = interference(spec, city, anchor_s)

    def signal(self, xy, ticks):
        loss, _ = geometry(self.city, self.scenario, xy)
        noise = np.zeros(xy.shape[:2])
        active = (ticks * .5 >= self.begin) & (ticks * .5 < self.end)
        if active.any():
            active_xy = xy[active].reshape(-1, 2)
            receivers = np.c_[active_xy, np.full(len(active_xy), self.scenario.altitude_m)]
            jam_loss = self.city.radio.propagation(self.jammer[None], receivers, self.scenario)[0]
            noise[active] = (10 ** ((self.power - jam_loss) / 10.)).reshape(-1, xy.shape[1])
        return links_from_loss(loss, noise, self.shadow[ticks])

    def outcomes(self, route, tick, ns, initial_connected):
        ticks = tick + np.arange(len(route.radio_xy) - 1)
        p, _, sinr = self.signal(route.radio_xy[:-1], ticks)
        graph = (p >= .9) & (p.swapaxes(-1, -2) >= .9)
        net = widest(np.where(graph, 1., -1.), ns).min(axis=-1) >= 0.
        if bool(net[0]) != initial_connected:
            raise ValueError('Branch disagrees with the logged initial radio graph')
        cuts = intervals(net, .5)
        margin = widest(np.minimum(sinr, sinr.swapaxes(-1, -2)) - THRESHOLD, ns).min(axis=-1)
        return (sum(a > 0. for a, _ in cuts), sum(b - a for a, b in cuts),
                .015 * float(np.clip(1.5 - margin, 0., 10.).sum()))


def cost(values):
    return 2. * values[..., 0] + values[..., 1] + values[..., 2]


def replay(record, role, cap=ANCHOR_CAP):
    started = perf_counter()
    row = read(record['metadata'])
    spec = row['spec']
    map_entry = registry()[spec['region']]
    if (map_entry['study_role'] != role or role not in ('train', 'calibration')
            or row['mode'] != 'physics_05s' or row['contract']['scope'] != role
            or row['archive_sha256'] != record['sha256'] or sha(record['archive']) != record['sha256']):
        raise ValueError(f'Unverified learning input: {spec}')
    with np.load(record['archive'], allow_pickle=False) as archive:
        logged = {name: archive[name] for name in ('network', 'links', 'positions', 'applied_offsets',
                                                    'times', 'features')}
    events = [event for event in row['events']
              if event['reason'] == 'prevention' and event['time_s'] + HORIZON <= row['duration_s']]
    anchors = {round(events[i]['time_s'] * 10): events[i] for i in
               np.linspace(0, len(events) - 1, min(cap, len(events)), dtype=int)}
    frames, features, controls, alternatives, connected = [], [], [], [], []
    if anchors:
        city, scenario, fleet, survey, anchor_s = survey_for(spec['region'], spec['direction'])
        n, ns, seed = fleet.n, fleet.ns, spec['seed']
        end_frame = max(anchors)
        nav = Navigator(city, ns, survey)
        sensor = Sensors(n, ns, seed + 113, 1.)
        sensor.random = np.random.default_rng(seed + 113).random((end_frame + 1, n, n), dtype=np.float32)
        radio = Radio(city, scenario, spec, n, end_frame // 5 + round(HORIZON / .5) + 1, anchor_s)
        applied, requested = np.zeros(n - ns, np.int32), np.zeros(n - ns, np.int32)
        history = RadioHistory(n)
        forecast = predictors()
        xy = nav.motion.positions()[0]
        changes = {}
        for t, relay, offset, _ in row['applied_changes']:
            changes.setdefault(round(t * 10), []).append((relay - ns, offset))
        publications = {round(e['published_s'] * 10): e for e in row['events'] if 'published_s' in e}
        input_rows = dict(zip(np.rint(logged['times'] * 10).astype(int), logged['features']))
        for frame in range(end_frame + 1):
            now, tick = frame / 10., frame // 5
            nav.update(now)
            sensor.deliver(now)
            for relay, offset in changes.get(frame, ()):
                applied[relay] = offset
            if frame in publications:
                event = publications[frame]
                offset = np.flatnonzero(np.all(OFFSETS == event['new_offset'], axis=1)).item()
                requested[event['changed_relay'] - ns] = offset
            if frame % 5 == 0:
                if (not np.array_equal(xy.astype(np.float32), logged['positions'][tick])
                        or not np.array_equal(applied, logged['applied_offsets'][tick])):
                    raise ValueError(f'Archived motion does not reproduce at {now}s: {spec}')
                history.accept(now, sensor.remote_sinr[:, ns], sensor.remote_stamp[:, ns])
            if frame in anchors:
                estimate = sensor.estimate(now)
                state = state_at(nav, scenario, now, applied, estimate, history)
                np.testing.assert_allclose(state[0], input_rows[frame], rtol=1e-5, atol=1e-5)
                options, _ = options_for(nav, scenario, now, requested, applied, estimate, state)
                original = anchors[frame]
                if (len(options) < 2 or options[1].event['changed_relay'] != original['changed_relay']
                        or options[1].event['new_offset'] != original['new_offset']):
                    raise ValueError(f'Archived choice does not reproduce at {now}s: {spec}')
                selection = guarded_options(nav, scenario, applied, estimate, state, options, 1, forecast)
                if selection is not None:
                    _, control, candidates, rows = selection
                    live = bool(logged['network'][tick])
                    control_metrics = radio.outcomes(control, tick, ns, live)
                    for (_, alternative), values in zip(candidates, rows, strict=True):
                        frames.append(frame)
                        features.append(values)
                        controls.append(control_metrics)
                        alternatives.append(radio.outcomes(alternative, tick, ns, live))
                        connected.append(live)
            if frame % 10 == 0:
                p, power, sinr = radio.signal(xy[None], np.array([tick]))
                graph = (p[0] >= .9) & (p[0].T >= .9)
                np.testing.assert_array_equal(graph, logged['links'][tick])
                sensor.transmit(frame, now, p[0], power[0], sinr[0])
            xy = nav.step(applied)[0]
    arrays = dict(frames=np.asarray(frames, np.int32), live=np.asarray(connected, bool),
                  features=np.asarray(features, np.float32).reshape(-1, len(INPUTS)),
                  reference=np.asarray(controls).reshape(-1, len(METRICS)),
                  alternative=np.asarray(alternatives).reshape(-1, len(METRICS)))
    gain = cost(arrays['reference']) - cost(arrays['alternative'])
    meta = dict(spec=spec, group=map_entry['study_group'], role=role,
                source=Path(record['metadata']).name, source_archive_sha256=record['sha256'],
                prevention_anchors=len(events), replayed=len(anchors), eligible=len(set(frames)),
                comparisons=len(frames), beneficial=int((gain > 1e-6).sum()),
                fewer_starts=int((arrays['alternative'][:, 0] < arrays['reference'][:, 0]).sum()),
                worse=int((gain < -1e-6).sum()), wall_s=perf_counter() - started)
    return meta, arrays
