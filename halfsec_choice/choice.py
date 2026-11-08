"""Compare guarded moves using only received RF and published navigation."""
from dataclasses import dataclass, replace

import numpy as np
from numba import njit

from fast_halfsec.controller import EARLY_SAFE, EARLY_TOLERANCE, PLANNER_DELAY_TICKS, options_for
from fast_halfsec.features import EARLY, SHORT, FEATURE_NAMES, deficit
from fast_halfsec.navigation import DT, GUARD, advance_mission, next_stage, planned_goals
from final_v1_fast.controller import TIMES, margins, runs
from final_v1_fast.motion import poses

HORIZON = 12.
SAMPLE_TIMES = np.arange(0., HORIZON + .5, .5)
FRAMES = np.unique(np.rint(np.r_[TIMES, SAMPLE_TIMES] / DT).astype(np.int32))
FORECAST_INDEX = np.searchsorted(FRAMES, np.rint(TIMES / DT))
SAMPLE_INDEX = np.searchsorted(FRAMES, np.rint(SAMPLE_TIMES / DT))
CURRENT_FEATURES = 39
MAX_ALTERNATIVES = 4
OPTION_NAMES = ('margin_gain', 'base_min', 'option_min', 'early_min_delta', 'onsets_avoided',
                'physics_value', 'offset_distance', 'route_distance', 'relay_index',
                'backup_delta', 'hold')
INPUTS = (FEATURE_NAMES[:CURRENT_FEATURES]
          + tuple(f'reference_{name}' for name in OPTION_NAMES)
          + tuple(f'change_{name}' for name in OPTION_NAMES)
          + ('early_risk_reference', 'early_risk_change', 'early_min_change_db',
             'mean_margin_change_db', 'final_waypoint_gain', 'mean_waypoint_gain'))


@njit(cache=True)
def trace(points, predecessor, neighbors, start, end, elapsed, age, waypoints, relay_options,
          stage, ns, applied, proposed, commit, stalled, frames):
    start, end, elapsed, age = start.copy(), end.copy(), elapsed.copy(), age.copy()
    positions = np.empty((len(frames), len(start), 2))
    velocity = np.empty_like(positions)
    stages = np.empty(len(frames), np.int32)
    sample = 0
    for frame in range(frames[-1] + 1):
        if frame and stage < len(waypoints):
            updated = next_stage(points, start, end, elapsed, waypoints, stage, ns)
            stalled = stalled + 1 if updated == stage else 0
            stage = updated
        if frame == frames[sample]:
            positions[sample], velocity[sample] = poses(points, start, end, elapsed)
            stages[sample] = stage
            sample += 1
            if sample == len(frames):
                break
        offsets = applied if frame < commit else proposed
        targets = planned_goals(waypoints, relay_options, stage, offsets, ns)
        advance_mission(points, predecessor, neighbors, start, end, elapsed, age, targets, ns, stalled)
    return positions, velocity, stages


@dataclass
class Branch:
    radio_xy: np.ndarray
    stages: np.ndarray
    blockers: np.ndarray
    margin: np.ndarray


def branch(nav, scenario, estimate, applied, option, commit):
    city, motion, survey = nav.city, nav.motion, nav.survey
    xy, vel, stage = trace(city.points, city.predecessor, city.neighbors, motion.start,
                           motion.end, motion.elapsed, motion.age, survey.waypoints, survey.options,
                           nav.stage, nav.ns, applied, option.offsets, commit, nav.stalled, FRAMES)
    radio_xy, stages = xy[SAMPLE_INDEX], stage[SAMPLE_INDEX]
    indices = np.minimum(stages[:, None] + np.arange(2), len(survey.waypoints) - 1)
    targets = city.points[survey.waypoints[indices, :nav.ns]]
    relay_xy = radio_xy[:, nav.ns:, None, None, :]
    close = np.any(np.sum((relay_xy - targets[:, None]) ** 2, axis=-1) < GUARD ** 2, axis=(2, 3))
    blockers = close & (np.linalg.norm(vel[SAMPLE_INDEX, nav.ns:], axis=-1) < .2)
    margin, _ = margins(city, scenario, xy[FORECAST_INDEX], estimate, nav.ns)
    return Branch(radio_xy, stages, blockers, margin)


def projected_features(raw, margin):
    projected = raw.copy()
    starts = runs(margin)
    projected[:len(TIMES) + 5] = np.r_[margin, margin[EARLY].min(), deficit(margin, EARLY),
                                     margin[SHORT].min(), deficit(margin, SHORT),
                                     starts[0] if starts else 9.]
    return np.clip(projected, -150., 150.)


def guarded_options(nav, scenario, applied, estimate, state, options, delivery, forecast):
    if len(options) < 3 or options[1].physical_value <= 0 or state[1][0] < 0 or nav.stalled >= 60:
        return None
    reference = options[1]
    commit = PLANNER_DELAY_TICKS + delivery
    control = branch(nav, scenario, estimate, applied, reference, commit)
    early_floor = np.minimum(0., control.margin[EARLY_SAFE]) - EARLY_TOLERANCE
    onset_limit = len(runs(control.margin))
    safe = []
    for option in options[2:2 + MAX_ALTERNATIVES]:
        if option.physical_value <= 0:
            break
        alt = branch(nav, scenario, estimate, applied, option, commit)
        if ((alt.stages < control.stages).any() or (alt.blockers & ~control.blockers).any()
                or (alt.margin[EARLY_SAFE] < early_floor).any() or len(runs(alt.margin)) > onset_limit):
            continue
        safe.append((option, alt))
    if not safe:
        return None
    raw = state[0]
    risk = forecast.scores('early_current', np.asarray([
        projected_features(raw, route.margin) for route in (control, *(alt for _, alt in safe))
    ]))
    features = []
    for index, (option, alt) in enumerate(safe, start=1):
        delta = alt.stages - control.stages
        features.append(np.r_[raw[:CURRENT_FEATURES], reference.values, option.values - reference.values,
                              risk[0], risk[index] - risk[0],
                              alt.margin[EARLY].min() - control.margin[EARLY].min(),
                              (alt.margin - control.margin).mean(), delta[-1], delta.mean()])
    return reference, control, safe, np.asarray(features, dtype=np.float32)


class Selector:
    def __init__(self, model, threshold, forecast):
        self.model, self.threshold, self.forecast = model, threshold, forecast
        self.scored = self.eligible = 0

    def __call__(self, nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery):
        options, tested = options_for(nav, scenario, now, requested, applied, estimate, state,
                                      learned_risk, delivery)
        if self.threshold is None:
            return options, tested
        choice = guarded_options(nav, scenario, applied, estimate, state, options, delivery, self.forecast)
        if choice is None:
            return options, tested
        reference, _, candidates, features = choice
        self.eligible += 1
        self.scored += len(candidates)
        scores = self.model.predict(features, num_threads=1)
        index = int(np.argmax(scores))
        if scores[index] < self.threshold:
            return options, tested
        option, _ = candidates[index]
        event = dict(option.event, expected_branch_gain=float(scores[index]),
                     physical_relay=reference.event['changed_relay'],
                     physical_offset=reference.event['new_offset'])
        return [options[0], replace(option, event=event)], tested
