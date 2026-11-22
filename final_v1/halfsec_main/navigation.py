"""Clear blocked survey waypoints with checked, single-aircraft edge reservations."""

from math import ceil

import numpy as np
from numba import njit

from fast_halfsec.navigation import DT, Navigator
from final_v1_fast.motion import advance, poses, reserve_ok


STALL_FRAMES = ceil(20. / DT)
RETRY_FRAMES = ceil(5. / DT)


@njit(cache=True)
def route_hop(neighbors, origin, target, occupied):
    """First free edge and length of a shortest path around parked aircraft."""
    if occupied[target]:
        return -1, -1
    by_node = np.full(len(neighbors), -1, np.int32)
    distance = np.zeros(len(neighbors), np.int32)
    queue = np.empty(len(neighbors), np.int32)
    queue[0] = origin
    by_node[origin] = origin
    first, last = 0, 1
    while first < last:
        node = queue[first]
        first += 1
        for other in neighbors[node]:
            if other < 0 or occupied[other] or by_node[other] >= 0:
                continue
            by_node[other] = other if node == origin else by_node[node]
            distance[other] = distance[node] + 1
            if other == target:
                return by_node[other], distance[other]
            queue[last] = other
            last += 1
    return -1, -1


def clearance_move(city, nodes, targets):
    """Find a safe search detour or one parking move that opens a search detour."""
    occupied = np.zeros(len(city.points), dtype=bool)
    occupied[nodes] = True
    unfinished = np.flatnonzero(nodes[:len(targets)] != targets)

    def routes(configuration, free_to_occupy):
        for searcher in unfinished:
            node = configuration[searcher]
            free_to_occupy[node] = False
            hop, length = route_hop(city.neighbors, node, targets[searcher], free_to_occupy)
            free_to_occupy[node] = True
            if hop >= 0:
                yield length, int(searcher), int(hop)

    stationary = np.zeros(len(nodes))
    for _, searcher, hop in sorted(routes(nodes, occupied)):
        if reserve_ok(city.points, nodes, nodes, stationary, searcher, hop):
            return searcher, hop

    alternatives = []
    movable = [*range(len(targets), len(nodes)), *unfinished]
    for aircraft in movable:
        node = nodes[aircraft]
        for park in city.neighbors[node]:
            if park < 0 or occupied[park]:
                continue
            # Leave all survey destinations free for their scheduled aircraft.
            if park in targets or not reserve_ok(city.points, nodes, nodes, stationary, aircraft, park):
                continue
            parked = nodes.copy()
            parked[aircraft] = park
            parked_mask = occupied.copy()
            parked_mask[node], parked_mask[park] = False, True
            for length, searcher, hop in routes(parked, parked_mask):
                if reserve_ok(city.points, parked, parked, stationary, searcher, hop):
                    alternatives.append((length, aircraft < len(targets), int(aircraft), int(park)))
    if alternatives:
        _, _, aircraft, park = min(alternatives)
        return aircraft, park
    return None


class ClearanceNavigator(Navigator):
    """Give a stalled survey cursor a safe route without skipping a waypoint."""

    def __init__(self, city, ns, survey):
        super().__init__(city, ns, survey)
        self.clearing = False
        self.retry_after = STALL_FRAMES
        self.cleared_stages = 0
        self.clearance_edges = 0

    def update(self, time_s):
        before = self.stage
        super().update(time_s)
        if before != self.stage:
            self.cleared_stages += int(self.clearing)
            self.clearing = False
            self.retry_after = STALL_FRAMES
        if self.finished_at is None and self.stalled >= self.retry_after:
            self.clearing = True

    def step(self, offsets):
        if not self.clearing:
            return super().step(offsets)
        motion = self.motion
        current, velocity = motion.positions()
        if np.allclose(current, self.city.points[motion.end], atol=1e-8, rtol=0.) and not velocity.any():
            targets = self.survey.waypoints[self.stage, :self.ns]
            move = clearance_move(self.city, motion.end, targets)
            if move is None:
                self.clearing = False
                self.retry_after = self.stalled + RETRY_FRAMES
                return super().step(offsets)
            aircraft, hop = move
            motion.start[:] = motion.end
            motion.elapsed[:] = 0.
            motion.end[aircraft] = hop
            motion.age[aircraft] = 0.
            self.clearance_edges += 1
        # Integrate reserved edges while keeping every other aircraft parked.
        advance(self.city.points, self.city.predecessor, motion.start, motion.end,
                motion.elapsed, motion.end.copy(), motion.age, self.ns, DT)
        return motion.positions()

    def forecast(self, offsets, applied, commit_frames=3, times=None):
        if not self.clearing:
            if times is None:
                return super().forecast(offsets, applied, commit_frames)
            return super().forecast(offsets, applied, commit_frames, times)
        if times is None:
            from final_v1_fast.controller import TIMES

            times = TIMES
        # Relay decisions pause while the navigation layer clears the next waypoint.
        motion = self.motion
        return np.asarray([
            poses(self.city.points, motion.start, motion.end, motion.elapsed, float(t))[0] for t in times
        ])
