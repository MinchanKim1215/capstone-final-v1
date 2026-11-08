"""Pace an existing public survey plan by arrival on acceleration-safe edges."""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from adaptive_joint_expanded.features import OFFSETS
from final_v1_fast.motion import Fleet, advance, duration, poses, reserve_ok
from final_v1_fast.controller import TIMES

DT = .1
GUARD = 5.26


@njit(cache=True)
def settled(points, start, end, elapsed, i):
    if start[i] == end[i]:
        return True
    length = np.sqrt(((points[end[i]] - points[start[i]]) ** 2).sum())
    return elapsed[i] >= duration(length) - 1e-8


@njit(cache=True)
def next_stage(points, start, end, elapsed, waypoints, stage, ns):
    while stage < len(waypoints):
        arrived = True
        for i in range(ns):
            if end[i] != waypoints[stage, i] or not settled(points, start, end, elapsed, i):
                arrived = False
                break
        if not arrived:
            break
        stage += 1
    return stage


@njit(cache=True)
def planned_goals(waypoints, relay_options, stage, offsets, ns):
    i = min(stage, len(waypoints) - 1)
    targets = waypoints[i].copy()
    for r in range(len(offsets)):
        for target in relay_options[i, r, offsets[r]]:
            if target >= 0 and not (targets[:ns + r] == target).any():
                targets[ns+r] = target
                break
    return targets


@njit(cache=True)
def search_clearance(points, predecessor, start, end, goals, ns, node):
    clearance = 1e6
    for i in range(ns):
        if end[i] == goals[i]:
            continue
        a = points[end[i]]
        hop = predecessor[goals[i], end[i]]
        if hop < 0:
            continue
        delta = points[hop] - a
        delta2 = max(1e-12, np.dot(delta, delta))
        distance = points[node] - a
        fraction = min(1., max(0., np.dot(distance, delta) / delta2))
        gap = np.sqrt(((distance - fraction * delta) ** 2).sum())
        clearance = min(clearance, gap)
    return clearance


@njit(cache=True)
def yield_relay(points, predecessor, neighbors, start, end, elapsed, age, goals, ns, stalled):
    # A relay already parked in the next survey corridor must be able to
    # move aside even if its own shortest route is blocked by another relay.
    limit = GUARD if stalled < 160 else 20.
    chosen_agent, chosen_target = -1, -1
    best = -1e9
    for i in range(ns, len(start)):
        if not settled(points, start, end, elapsed, i):
            continue
        start[i] = end[i]
        elapsed[i] = 0.
        old_clearance = search_clearance(points, predecessor, start, end, goals, ns, start[i])
        if old_clearance >= limit:
            continue
        for target in neighbors[start[i]]:
            if target < 0 or target == start[i]:
                continue
            new_clearance = search_clearance(points, predecessor, start, end, goals, ns, target)
            gain = new_clearance - old_clearance
            if gain <= 1e-8 or not reserve_ok(points, start, end, elapsed, i, target):
                continue
            value = gain + (100. if old_clearance < GUARD else 0.)
            if value > best:
                best = value
                chosen_agent, chosen_target = i, target
    if chosen_agent >= 0:
        end[chosen_agent] = chosen_target
        age[chosen_agent] = 0.


@njit(cache=True)
def advance_mission(points, predecessor, neighbors, start, end, elapsed, age, goals, ns, stalled):
    # A stalled search aircraft can yield at the next safe node. The route
    # cursor stays at the original waypoint until that waypoint is reached.
    if stalled >= 60:
        yield_relay(points, predecessor, neighbors, start, end, elapsed, age, goals, ns, stalled)
    if stalled >= 50:
        for i in range(ns):
            if end[i] == goals[i] or not settled(points, start, end, elapsed, i):
                continue
            start[i] = end[i]
            elapsed[i] = 0.
            default = predecessor[goals[i], start[i]]
            if default >= 0 and reserve_ok(points, start, end, elapsed, i, default):
                continue
            options = neighbors[start[i]]
            for j in range(len(options)):
                target = options[(j + stalled // 30 + i) % len(options)]
                if target >= 0 and target != start[i] and reserve_ok(points, start, end, elapsed, i, target):
                    end[i] = target
                    age[i] = 0.
                    break
            if end[i] != start[i]:
                break
    return advance(points, predecessor, start, end, elapsed, goals, age, ns, DT)


@njit(cache=True)
def preview(points, predecessor, neighbors, start, end, elapsed, age, waypoints,
            relay_options, stage, ns, old_offsets, new_offsets, commit_frames, stalled, times):
    a, b, e, wait = start.copy(), end.copy(), elapsed.copy(), age.copy()
    out = np.empty((len(times), len(a), 2))
    sample = 0
    limit = int(np.rint(times[-1] / DT))
    for frame in range(limit + 1):
        if frame:
            stage_next = next_stage(points, a, b, e, waypoints, stage, ns)
            stalled = stalled + 1 if stage_next == stage else 0
            stage = stage_next
        if sample < len(times) and abs(frame * DT - times[sample]) < 1e-7:
            out[sample], _ = poses(points, a, b, e)
            sample += 1
        if frame < limit:
            offsets = old_offsets if frame < commit_frames else new_offsets
            targets = planned_goals(waypoints, relay_options, stage, offsets, ns)
            advance_mission(points, predecessor, neighbors, a, b, e, wait, targets, ns, stalled)
    return out


def safe_relay_options(city, nodes, ns):
    count = len(nodes) - 1
    options = np.full((count, nodes.shape[1] - ns, len(OFFSETS), 12), -1, dtype=np.int32)
    points = city.points
    for step in range(count):
        source = points[nodes[step, :ns]]
        target = points[nodes[step + 1, :ns]]
        delta = target - source
        relative = points[:, None, :] - source[None, :, :]
        fraction = np.clip(np.sum(relative * delta, axis=-1) /
                           np.maximum(1e-12, np.sum(delta**2, axis=-1)), 0., 1.)
        distance = np.linalg.norm(relative - fraction[..., None] * delta, axis=-1)
        allowed = np.flatnonzero(np.all(distance >= GUARD, axis=1))
        if len(allowed) < nodes.shape[1] - ns:
            raise ValueError('Insufficient safe relay target sites')
        nominal = points[nodes[step + 1, ns:]]
        requested = nominal[:, None, :] + OFFSETS[None, :, :]
        nearest = cKDTree(points[allowed]).query(np.clip(requested, -195., 195.), k=min(12, len(allowed)))[1]
        options[step, :, :, :nearest.shape[-1]] = allowed[nearest]
    return options


@dataclass(frozen=True)
class Survey:
    initial: np.ndarray
    waypoints: np.ndarray
    options: np.ndarray
    baseline_s: float
    observable: int


def make_survey(city, ns, nodes):
    # Skip waits from the nominal radio-free plan. Each distinct search
    # waypoint is visited; the sharing interval is scheduled separately.
    change = np.any(nodes[1:, :ns] != nodes[:-1, :ns], axis=1)
    selected = np.r_[True, change]
    route = np.ascontiguousarray(nodes[selected], dtype=np.int32)
    if len(route) < 2:
        route = np.repeat(route, 2, axis=0)
    return Survey(route[0], route[1:], safe_relay_options(city, route, ns),
                  (len(nodes) - 1) * 3., int(city.observable.sum()))


class Navigator:
    def __init__(self, city, ns, survey):
        self.city = city
        self.ns = ns
        self.survey = survey
        self.motion = Fleet(city, survey.initial, ns)
        self.stage = 0
        self.stalled = 0
        self.covered = city.visible[survey.initial[:ns]].any(axis=0)
        self.finished_at = None

    def update(self, time_s):
        motion = self.motion
        updated = next_stage(self.city.points, motion.start, motion.end, motion.elapsed,
                             self.survey.waypoints, self.stage, self.ns)
        if updated > self.stage:
            nodes = self.survey.waypoints[self.stage:updated, :self.ns].ravel()
            self.covered |= self.city.visible[nodes].any(axis=0)
            self.stalled = 0
        elif self.stage < len(self.survey.waypoints):
            self.stalled += 1
        self.stage = updated
        if self.stage == len(self.survey.waypoints) and self.finished_at is None:
            self.finished_at = time_s

    def step(self, offsets):
        motion = self.motion
        targets = planned_goals(self.survey.waypoints, self.survey.options, self.stage, offsets, self.ns)
        motion.blocked += advance_mission(self.city.points, self.city.predecessor, self.city.neighbors,
                                          motion.start, motion.end, motion.elapsed, motion.age,
                                          targets, self.ns, self.stalled)
        return motion.positions()

    def forecast(self, offsets, applied, commit_frames=3, times=TIMES):
        motion = self.motion
        return preview(self.city.points, self.city.predecessor, self.city.neighbors,
                       motion.start, motion.end, motion.elapsed, motion.age, self.survey.waypoints,
                       self.survey.options, self.stage, self.ns, applied, offsets, commit_frames,
                       self.stalled, times)
