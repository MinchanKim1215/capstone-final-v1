"""Stage an existing relay before a survey passage and verify a live route before entry."""

from dataclasses import asdict, dataclass, replace

import numpy as np
from numba import njit

from cooperative.packets import PacketSim
from fast_halfsec.controller import Option
from fast_halfsec.features import N_CANDIDATE
from fast_halfsec.navigation import planned_goals, settled
from final_v1_fast.controller import TIMES
from final_v1_fast.motion import DT, advance, poses
from swarm_sim.mission import MAX_MISSION

from .mission import MainFlight
from .navigation import ClearanceNavigator
from .release import load_model, map_scope


REPORT_AGE_S = 2.
WAIT_LIMIT_S = 12.


@dataclass(frozen=True)
class Passage:
    begin_stage: int
    entry_stage: int
    end_stage: int
    scout: int
    relay: int
    node: int


def reserve_survey(survey, ns, passage):
    section = slice(passage.begin_stage, passage.end_stage)
    waypoints, options = survey.waypoints.copy(), survey.options.copy()
    if np.any(waypoints[section, :ns] == passage.node):
        raise ValueError('Relay station occupies a search waypoint')
    updated = options[section]
    updated[updated == passage.node] = -1
    relay = passage.relay - ns
    updated[:, relay] = -1
    updated[:, relay, :, 0] = passage.node
    for stage in range(passage.begin_stage, passage.end_stage):
        goals = waypoints[stage]
        for aircraft in np.flatnonzero(goals == passage.node):
            if aircraft == passage.relay:
                continue
            choices = updated[stage - passage.begin_stage, aircraft - ns].ravel()
            unused = next((int(value) for value in choices if value >= 0 and value not in goals), None)
            if unused is None:
                raise ValueError('Relay station has no free fallback target for another relay')
            goals[aircraft] = unused
        goals[passage.relay] = passage.node
    return replace(survey, waypoints=waypoints, options=options)


class TrafficEvidence(PacketSim):
    """Retain the last successful hop ACK beyond a single motion tick."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hops_seen = np.full((self.n, self.n), -1.)

    def interval(self, p_t, node_ids):
        result = super().interval(p_t, node_ids)
        np.maximum(self.hops_seen, self.last_ack, out=self.hops_seen)
        return result


def route_confirmed(now, arrived, anchor, ns, hops, beacons, packets):
    since = max(arrived, now - REPORT_AGE_S)
    ack = (hops >= since - 1e-8) & (hops <= now + 1e-8)
    observed = (beacons >= since - 1e-8) & (beacons <= now + 1e-8)
    edges = ack | ack.T | (observed & observed.T)
    reached = np.zeros(len(edges), bool)
    reached[anchor] = True
    for _ in range(len(edges)):
        reached |= edges[reached].any(axis=0)
    delivered = packets[~np.eye(ns, dtype=bool)]
    return bool(reached[:ns].all() and (delivered >= since - 1e-8).all()
                and (delivered <= now + 1e-8).all())


@njit(cache=True)
def preview_wait(points, predecessor, a, b, elapsed, age, old, new, scout, commit, ns, times):
    start, end, spent, wait = a.copy(), b.copy(), elapsed.copy(), age.copy()
    output = np.empty((len(times), len(a), 2))
    sample = 0
    for frame in range(round(times[-1] / DT) + 1):
        if sample < len(times) and abs(frame * DT - times[sample]) < 1e-7:
            output[sample] = poses(points, start, end, spent)[0]
            sample += 1
        goals = (old if frame < commit else new).copy()
        goals[scout] = b[scout]
        advance(points, predecessor, start, end, spent, goals, wait, ns, DT)
    return output


class PassageNavigator(ClearanceNavigator):
    def __init__(self, city, ns, survey, passages, verify):
        super().__init__(city, ns, survey)
        previous_end = 0
        for item in passages:
            if not (previous_end <= item.begin_stage <= item.entry_stage < item.end_stage <= len(survey.waypoints)
                    and 0 <= item.scout < ns <= item.relay < len(survey.initial)
                    and 0 <= item.node < len(city.points)):
                raise ValueError('Invalid or overlapping relay passage')
            previous_end = item.end_stage
        self.original = survey
        self.verify = verify
        self.active = None
        self.waiting = False
        self.records = []
        self.program = iter((passage, reserve_survey(survey, ns, passage)) for passage in passages)
        self.next_passage = next(self.program, None)

    def update(self, time_s):
        super().update(time_s)
        if self.active is not None and self.stage >= self.active.end_stage:
            self.finish(time_s, 'finished')
        if self.active is None:
            while self.next_passage is not None and self.stage >= self.next_passage[0].end_stage:
                passage = self.next_passage[0]
                self.records.append(dict(asdict(passage), outcome='skipped'))
                self.next_passage = next(self.program, None)
            if self.next_passage is None or self.stage < self.next_passage[0].begin_stage:
                return
            self.active, self.survey = self.next_passage
            self.next_passage = next(self.program, None)
            self.record = dict(asdict(self.active), started_s=time_s, arrived_s=None, entered_s=None,
                               confirmed_s=None, outcome='active')
            self.records.append(self.record)
        passage, record = self.active, self.record
        at_station = self.motion.end[passage.relay] == passage.node and settled(
            self.city.points, self.motion.start, self.motion.end, self.motion.elapsed, passage.relay)
        if not at_station:
            record['arrived_s'] = None
        elif record['arrived_s'] is None:
            record['arrived_s'] = time_s
        if self.stage < passage.entry_stage or record['confirmed_s'] is not None:
            return
        if record['entered_s'] is None:
            record['entered_s'] = time_s
        if record['arrived_s'] is not None and self.verify(time_s, record['arrived_s'], passage.relay):
            record['confirmed_arrival_s'] = record['arrived_s']
            record['confirmed_s'] = time_s
            self.waiting = False
        elif time_s - record['entered_s'] >= WAIT_LIMIT_S - 1e-8:
            self.finish(time_s, 'timed_out')
        else:
            self.clearing = False
            self.waiting = True

    def finish(self, time_s, outcome):
        self.record.update(outcome=outcome, finished_s=time_s)
        self.active, self.waiting = None, False
        self.survey = self.original

    def goals(self, offsets):
        return planned_goals(self.survey.waypoints, self.survey.options, self.stage, offsets, self.ns)

    def step(self, offsets):
        if not self.waiting:
            return super().step(offsets)
        motion = self.motion
        goals = self.goals(offsets)
        goals[self.active.scout] = motion.end[self.active.scout]
        advance(self.city.points, self.city.predecessor, motion.start, motion.end,
                motion.elapsed, goals, motion.age, self.ns, DT)
        return motion.positions()

    def forecast(self, offsets, applied, commit_frames=3, times=None):
        if not self.waiting:
            return super().forecast(offsets, applied, commit_frames, times)
        motion = self.motion
        return preview_wait(self.city.points, self.city.predecessor, motion.start, motion.end,
                            motion.elapsed, motion.age, self.goals(applied), self.goals(offsets),
                            self.active.scout, commit_frames, self.ns, TIMES if times is None else times)


class PassageFlight(MainFlight):
    def __init__(self, model, threshold, passages):
        super().__init__(model, threshold)
        self.passages = tuple(passages)
        self.packet_factory = TrafficEvidence

    def navigator(self, city, ns, survey):
        self.nav = PassageNavigator(city, ns, survey, self.passages, self.verified)
        return self.nav

    def verified(self, now, arrived, relay):
        return route_confirmed(now, arrived, relay, self.nav.ns, self.packets.hops_seen,
                               self.sensor.remote_stamp, self.packets.last_received)

    def options_for(self, nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery):
        if nav.waiting:
            values = np.zeros(N_CANDIDATE, np.float32)
            values[-1] = 1.
            return [Option(requested.copy(), values, 0., None)], 0
        return super().options_for(nav, scenario, now, requested, applied, estimate, state, learned_risk, delivery)


def run(spec, passages, max_seconds=MAX_MISSION):
    with map_scope(spec['region']):
        flight = PassageFlight(*load_model(), passages)
        row, arrays = flight.run(spec, max_seconds)
    row.update(controller='halfsec_staged_passage', planned_passages=[asdict(p) for p in flight.passages],
               staged_passages=flight.nav.records)
    return row, arrays
