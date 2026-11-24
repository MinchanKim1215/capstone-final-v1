"""Execute a preplanned climb with eight aircraft, then verify each descent step.

수평 이동은 통과 비행(navigation_pass, motion_pass)을 쓴다. 고도 구간 진입 확인 전에는 수색기가 진입 지점을
넘어서는 직선 구간을 예약하지 않고, 대기·상승 중인 기체는 제자리에 세운 채 나머지만 통과 비행 규칙으로 움직인다.
"""

from dataclasses import asdict, dataclass
from math import ceil, sqrt

import numpy as np

from fast_halfsec.navigation import DT, settled
from final_v1_fast.controller import TIMES

from .motion_pass import advance, poses
from .navigation_pass import PassNavigator

from .navigation import ClearanceNavigator


BASE_HEIGHT_M = 50.
VERTICAL_SPEED_MPS = 2.
VERTICAL_ACCEL_MPS2 = 1.
COMMAND_DELAY_S = .3
PROOF_WAIT_S = 6.
ENTRY_WAIT_S = 38.
DESCENT_STEP_M = 5.
SAMPLE_S = .5
EPS = 1e-7


@dataclass(frozen=True)
class HeightWindow:
    begin_stage: int
    entry_stage: int
    end_stage: int
    relay: int
    height_m: float


def duration_s(delta_m):
    distance = abs(delta_m)
    raw = max(1.5 * distance / VERTICAL_SPEED_MPS, sqrt(6. * distance / VERTICAL_ACCEL_MPS2))
    return ceil(raw / DT - EPS) * DT


@dataclass(frozen=True)
class VerticalMove:
    start_s: float
    start_m: float
    target_m: float
    kind: str

    @property
    def end_s(self):
        return self.start_s + duration_s(self.target_m - self.start_m)

    def sample(self, now):
        duration = self.end_s - self.start_s
        if duration <= EPS or now >= self.end_s - EPS:
            return self.target_m, 0.
        phase = max(0., (now - self.start_s) / duration)
        distance = self.target_m - self.start_m
        return (self.start_m + distance * phase**2 * (3. - 2. * phase),
                distance * 6. * phase * (1. - phase) / duration)


class HeightNavigator(ClearanceNavigator):
    """The horizontal engine keeps its collision guard while a relay changes height at rest."""

    def __init__(self, city, ns, survey, windows, confirm, descent_safe):
        super().__init__(city, ns, survey)
        n = len(survey.initial)
        self.windows = tuple(windows)
        if len({window.relay for window in windows}) > 1:
            raise ValueError('A flight must use one preassigned height-control relay')
        previous = 0
        for window in windows:
            if not (previous <= window.begin_stage <= window.entry_stage < window.end_stage <= len(survey.waypoints)
                    and ns < window.relay < n and BASE_HEIGHT_M < window.height_m <= 80.):
                raise ValueError('Invalid altitude window')
            previous = window.end_stage
        self.relay = windows[0].relay if windows else None
        self.heights = np.full(n, BASE_HEIGHT_M, dtype=float)
        if windows:
            city.radio_altitudes_m = self.heights
        self.confirm = confirm
        self.descent_safe = descent_safe
        self.last_offsets = np.zeros(n-ns, np.int32)
        self.next_window = 0
        self.window = None
        self.record = None
        self.move = None
        self.pending = None
        self.prove = None
        self.previous_safe = BASE_HEIGHT_M
        self.retry_after_stage = -1
        self.speed = 0.
        self.max_speed = self.max_acceleration = self.vertical_travel = 0.
        self.samples, self.moves, self.windows_log = [], [], []

    def altitude_request(self, height, kind):
        if self.pending is None and self.move is None and abs(self.heights[self.relay]-height) > EPS:
            self.pending = (height, kind)

    def tick_altitude(self, now):
        if self.relay is None:
            return
        previous_height, previous_speed = self.heights[self.relay], self.speed
        motion = self.motion
        if self.pending is not None and settled(self.city.points, motion.start, motion.end, motion.elapsed, self.relay):
            target, kind = self.pending
            self.pending = None
            self.move = VerticalMove(now + COMMAND_DELAY_S, previous_height, target, kind)
            self.moves.append(dict(asdict(self.move), request_s=now, end_s=self.move.end_s, relay=self.relay))
        if self.move is not None:
            height, self.speed = self.move.sample(now)
            self.heights[self.relay] = height
            if now >= self.move.end_s - EPS:
                if self.move.kind == 'descent':
                    self.prove = now
                elif self.move.kind == 'recovery':
                    self.retry_after_stage = self.stage
                self.move = None
        if abs(self.speed) > EPS and not settled(self.city.points, motion.start, motion.end, motion.elapsed, self.relay):
            raise ValueError('The climbing relay moved on the horizontal graph')
        self.max_speed = max(self.max_speed, abs(self.speed))
        self.max_acceleration = max(self.max_acceleration, abs(self.speed-previous_speed) / DT)
        self.vertical_travel += abs(self.heights[self.relay]-previous_height)

    def advance_program(self, now):
        if self.window is not None and self.stage >= self.window.end_stage:
            self.record['outcome'] = 'crossed' if self.record['verified_s'] is not None else 'passed_before_proof'
            self.record['finished_s'] = now
            self.window = self.record = None
        while self.window is None and self.next_window < len(self.windows):
            next_window = self.windows[self.next_window]
            if self.stage < next_window.begin_stage:
                break
            self.next_window += 1
            if self.stage >= next_window.end_stage:
                self.windows_log.append(dict(asdict(next_window), outcome='skipped', started_s=now))
                continue
            self.window = next_window
            self.retry_after_stage = -1
            self.record = dict(asdict(next_window), started_s=now, ready_s=None, entry_request_s=None,
                               verified_s=None, outcome='approach')
            self.windows_log.append(self.record)
        if self.window is not None:
            self.approach(now)
        elif self.relay is not None and self.pending is None and self.move is None:
            self.return_to_base(now)

    def approach(self, now):
        window, record = self.window, self.record
        if record['ready_s'] is None and self.heights[self.relay] >= window.height_m-EPS and self.move is None:
            record['ready_s'] = now
        if record['ready_s'] is None:
            self.altitude_request(window.height_m, 'climb')
        if self.stage < window.entry_stage or record['verified_s'] is not None:
            return
        if record['entry_request_s'] is None:
            record['entry_request_s'] = now
        if record['ready_s'] is not None and self.confirm(now, record['ready_s'], self.relay):
            record['verified_s'], record['outcome'] = now, 'crossing'
        elif now-record['entry_request_s'] >= ENTRY_WAIT_S-EPS:
            record.update(outcome='entry_timeout', finished_s=now)
            self.window = self.record = None

    def return_to_base(self, now):
        if self.prove is not None:
            if self.confirm(now, self.prove, self.relay):
                self.prove = None
            elif now-self.prove >= PROOF_WAIT_S-EPS:
                self.prove = None
                self.altitude_request(self.previous_safe, 'recovery')
            return
        if self.heights[self.relay] <= BASE_HEIGHT_M+EPS or self.stage <= self.retry_after_stage:
            return
        if abs(2.*now-round(2.*now)) > EPS:
            return
        target = max(BASE_HEIGHT_M, self.heights[self.relay]-DESCENT_STEP_M)
        if self.descent_safe(now, self.relay, target):
            self.previous_safe = float(self.heights[self.relay])
            self.altitude_request(target, 'descent')

    def update(self, now):
        super().update(now)
        self.tick_altitude(now)
        self.advance_program(now)
        if abs(2.*now - round(2.*now)) < EPS:
            self.samples.append(self.heights.copy())

    @property
    def entry_waiting(self):
        return self.record is not None and self.record['entry_request_s'] is not None and self.record['verified_s'] is None

    def held(self):
        mask = np.zeros(len(self.heights), bool)
        mask[:self.ns] = self.entry_waiting
        if self.pending is not None or self.move is not None or self.prove is not None:
            mask[self.relay] = True
        return mask

    def run_stop(self):
        """확인이 끝나지 않은 고도 구간(진행 중이거나 다음 구간)의 진입 지점 바로 앞까지만 한 번에 날아간다."""
        stop = super().run_stop()
        pending = []
        if self.window is not None and self.record is not None and self.record['verified_s'] is None:
            pending.append(self.window)
        if self.next_window < len(self.windows):
            pending.append(self.windows[self.next_window])
        for window in pending:
            stop = min(stop, window.entry_stage - 1)
        return stop

    def goals(self, offsets):
        return self.run_targets(offsets)

    def step(self, offsets):
        self.last_offsets[:] = offsets
        mask = self.held()
        if not mask.any():
            return super().step(offsets)
        motion = self.motion
        goals = self.goals(offsets)
        goals[mask] = motion.end[mask]
        motion.blocked += advance(self.city.points, self.city.predecessor, motion.start, motion.end,
                                  motion.elapsed, goals, motion.age, self.ns, DT)
        return motion.positions()

    def predict_holds(self, requested, applied, commit_frames, times, mask):
        return PassNavigator.forecast(self, requested, applied, commit_frames, times, hold=mask)

    def forecast(self, offsets, applied, commit_frames=3, times=None):
        mask = self.held()
        if not mask.any():
            return super().forecast(offsets, applied, commit_frames, times)
        return self.predict_holds(offsets, applied, commit_frames, TIMES if times is None else times, mask)
