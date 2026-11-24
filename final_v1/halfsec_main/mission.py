"""Add a conservative recovery check and waypoint clearance to JointFlight."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np

from adaptive_joint_expanded.features import widest
from fast_halfsec import runner
from fast_halfsec.controller import EARLY_SAFE, PLANNER_DELAY_TICKS, Option
from fast_halfsec.features import N_CANDIDATE
from final_v1.halfsec_joint.mission import JointFlight
from final_v1_fast.controller import TIMES, WEIGHTS, margins, runs
from final_v1_fast.sensing import Commands

from . import DECISION_S
from .navigation import ClearanceNavigator
from .navigation_pass import PassCity


HALF_SEC = np.isclose((TIMES / DECISION_S) % 1., 0.)
BEACON_MAX_AGE_S = 3.
BEACON_PERSISTENCE_S = 1.


def steady_reconnect(margin):
    samples = margin[HALF_SEC]
    stable = np.flatnonzero((samples[:-1] >= 0.) & (samples[1:] >= 0.))
    return float(TIMES[HALF_SEC][stable[0]]) if len(stable) else np.inf


def recovery_metrics(graphs, ns, search_missing, reported_age):
    penalty = np.zeros(graphs.shape[-2:])
    penalty = np.broadcast_to(penalty, graphs.shape).copy()
    lost = np.flatnonzero(search_missing)
    error = .25 * np.minimum(30., reported_age[lost]) + .20 * TIMES[:, None]
    penalty[:, lost, :] += error[:, :, None]
    penalty[:, :, lost] += error[:, None, :]
    pairs = widest(graphs - penalty, ns)
    source, target = np.triu_indices(ns, 1)
    affected = search_missing[source] | search_missing[target]
    reconnect = pairs[:, affected].min(axis=1)
    return (pairs.min(axis=1), pairs[:, ~affected], steady_reconnect(reconnect),
            float(np.dot(np.maximum(0., -reconnect), WEIGHTS)))


def recovers_safely(reference, alternative):
    base, healthy, reconnect, debt = reference
    alt, alt_healthy, alt_reconnect, alt_debt = alternative
    return (
        (alt_healthy[EARLY_SAFE] >= np.minimum(0., healthy[EARLY_SAFE])).all()
        and sum(t > 0. for t in runs(alt)) <= sum(t > 0. for t in runs(base))
        and (alt_reconnect < reconnect or (alt_reconnect == reconnect and alt_debt <= debt - 1.5))
    )


class StressCommands(Commands):
    def __init__(self, n, observer, initial, seed, latency, anchor_s):
        super().__init__(n, observer, initial, seed, latency)
        self.windows = [(max(36., (anchor_s * fraction // 3.) * 3.), observer + (index + 1) % len(initial))
                        for index, fraction in enumerate((.20, .47, .71))]

    def transmit(self, frame, now, p):
        missing = [relay for begin, relay in self.windows if relay != self.observer and begin <= now < begin + 12.]
        if missing:
            p = p.copy()
            p[missing, :] = p[:, missing] = 0.
        return super().transmit(frame, now, p)


class MainFlight(JointFlight):
    def __init__(self, model, threshold):
        super().__init__(model, threshold, explore=False, comparison=None)
        self.navigator_factory = ClearanceNavigator
        self.recovery_filtered = 0

    def survey(self, *args):
        city, *rest = super().survey(*args)
        # 통과 비행은 여러 칸을 한 구간으로 난다. 비행 후 안전 검사가 이 구간을 안전 간선의 일직선 연결로 확인하게 한다.
        value = (PassCity(city), *rest)
        self.anchor_s = value[-1]
        return value

    def sensors(self, *args):
        self.sensor = super().sensors(*args)
        return self.sensor

    def options_for(self, nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery):
        if nav.clearing:
            keep = np.zeros(N_CANDIDATE, np.float32)
            keep[-1] = 1.
            return [Option(requested.copy(), keep, 0., None)], 0
        options, tested = super().options_for(nav, scenario, now, requested, applied, estimate,
                                              state, learned_risk, delivery)
        if len(options) < 2 or state[1][0] >= 0.:
            return options, tested
        stamps = self.sensor.remote_stamp.max(axis=1)
        age = now - stamps
        missing = (stamps[:nav.ns] >= 0.) & (age[:nav.ns] > BEACON_MAX_AGE_S)
        if not missing.any():
            return options, tested
        sustained = missing & (age[:nav.ns] >= BEACON_MAX_AGE_S + BEACON_PERSISTENCE_S)
        # A missing report alone never produces a new move. Require a modeled
        # outage at the current trajectory and compare the existing safe moves.
        if not sustained.any() or not any(t == 0. for t in runs(state[1])):
            self.recovery_filtered += len(options) - 1
            return options[:1], tested

        seen = (stamps >= 0.) & (age <= BEACON_MAX_AGE_S)
        noise = estimate.noise.copy()
        lost = np.flatnonzero(missing)
        if seen.any():
            noise[lost] = np.maximum(noise[lost], noise[seen].max())
        conservative = replace(estimate, noise=noise)

        def evaluate(option, commit):
            xy = nav.forecast(option.offsets, applied, commit)
            _, graphs = margins(nav.city, scenario, xy, conservative, nav.ns)
            return recovery_metrics(graphs, nav.ns, missing, age)

        baseline = evaluate(options[0], delivery)
        allowed = [options[0]]
        for option in options[1:]:
            relay = option.event['changed_relay']
            if relay != nav.ns and not seen[relay]:
                continue
            alternative = evaluate(option, PLANNER_DELAY_TICKS + delivery)
            if recovers_safely(baseline, alternative):
                when = alternative[2]
                option.event.update(recovery_missing_search=lost.tolist(),
                                    projected_reconnect_s=when if np.isfinite(when) else None)
                allowed.append(option)
        self.recovery_filtered += len(options) - len(allowed)
        return allowed, tested

    def run(self, spec, max_seconds, command_loss=False):
        if command_loss:
            def channel(n, observer, initial, seed, latency):
                return StressCommands(n, observer, initial, seed, latency, self.anchor_s)

            with patch.object(runner, 'Commands', channel):
                row, frames = super().run(spec, DECISION_S, max_seconds)
        else:
            row, frames = super().run(spec, DECISION_S, max_seconds)
        row.update(clearance_edges=self.nav.clearance_edges, cleared_stages=self.nav.cleared_stages,
                   recovery_filtered=self.recovery_filtered, command_loss_stress=command_loss)
        return row, frames
