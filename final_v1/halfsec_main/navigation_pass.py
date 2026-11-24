"""Pass-through survey navigation for the default half-second controller.

수색기는 직선 구간에서는 멈추지 않고 지점 위를 지나가며, 꺾는 곳에서만 서서 서로를 기다린다.
- 지점 판정: 수색기마다 지난 지점을 기록한다(reached). 멈추지 않아도 지점 위를 지나면 통과다.
  전체 단계(stage)는 세 수색기가 모두 지난 지점까지만 올라간다. 원본의 '모두 지나야 다음' 규칙과 같다.
- 수색기 목표: 세 수색기가 모두 곧게 가는 구간(run_end)의 끝까지. 다만 지금 위치에서 일직선 위에
  아직 안 지난 지점들이 차례로 놓여 있을 때만 멀리 가고, 아니면 다음 지점 하나를 목표로 둔다(지점을 건너뛰지 않음).
- 중계기 목표: 최대 RELAY_LEAD 단계 앞의 대형 위치 (직선 구간에서 수색기를 따라가도록).
- 막혔을 때(20초 진행 없음) 처리는 ClearanceNavigator 원본 동작을 그대로 쓴다.

settled, yield_relay, advance_mission은 fast_halfsec/navigation.py와 같은 코드다. 이동 함수만
motion_pass를 쓰도록 이 파일에 옮겼다 (numba 함수는 불러온 모듈의 이동 함수를 컴파일 때 고정하기 때문).
"""
import numpy as np
from numba import njit

from fast_halfsec.navigation import DT, GUARD, Navigator, search_clearance
from final_v1_fast.controller import TIMES

from .motion_pass import Fleet, advance, duration, poses, profile, reserve_ok

RELAY_LEAD = 3
RUN_STAGE_CAP = 20
NO_STOP = 1 << 30  # run_stop 기본값: 단계 제한 없음


@njit(cache=True)
def settled(points, start, end, elapsed, i):
    if start[i] == end[i]:
        return True
    length = np.sqrt(((points[end[i]] - points[start[i]]) ** 2).sum())
    return elapsed[i] >= duration(length) - 1e-8


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
def passed(points, start, end, elapsed, i, w):
    """수색기 i가 지점 w에 있거나 지나갔는가 (지금 이동 중인 직선 구간 기준)."""
    a = start[i]
    b = end[i]
    if a == b:
        return w == a
    if w == a:
        return True
    d = points[b] - points[a]
    length = np.sqrt((d * d).sum())
    u = d / max(length, 1e-10)
    q = points[w] - points[a]
    projection = q[0] * u[0] + q[1] * u[1]
    off = q - projection * u
    if np.sqrt((off * off).sum()) > 1e-6 or projection < -1e-9 or projection > length + 1e-9:
        return False
    travelled, _ = profile(length, elapsed[i])
    return travelled >= projection - 1e-9


@njit(cache=True)
def progress(points, start, end, elapsed, waypoints, reached, ns):
    """수색기마다 지난 지점을 갱신하고, 셋 모두 지난 단계 수(= 전체 단계)를 돌려준다."""
    count = len(waypoints)
    for i in range(ns):
        while reached[i] + 1 < count:
            r = reached[i] + 1
            w = waypoints[r, i]
            if r > 0 and waypoints[r - 1, i] == w:
                reached[i] = r
                continue
            if passed(points, start, end, elapsed, i, w):
                reached[i] = r
            else:
                break
    low = reached[0]
    for i in range(1, ns):
        low = min(low, reached[i])
    return low + 1


@njit(cache=True)
def run_end(points, waypoints, initial, stage, ns, cap):
    """지금 단계부터 세 수색기가 모두 같은 방향으로 곧게 가는 마지막 단계."""
    last = len(waypoints) - 1
    s = min(stage, last)
    result = s
    direction = np.zeros((ns, 2))
    known = np.zeros(ns, np.bool_)
    for r in range(s, min(last, s + cap) + 1):
        straight = True
        for i in range(ns):
            before = waypoints[r - 1, i] if r > 0 else initial[i]
            v = points[waypoints[r, i]] - points[before]
            length = np.sqrt((v * v).sum())
            if length < 1e-9:
                continue
            u = v / length
            if known[i]:
                if u[0] * direction[i, 0] + u[1] * direction[i, 1] < 0.9999:
                    straight = False
                    break
            else:
                direction[i] = u
                known[i] = True
        if not straight:
            break
        result = r
    return result


@njit(cache=True)
def run_goals(points, waypoints, relay_options, initial, stage, reached, end, offsets, ns, lead, cap):
    """수색기: 지금 위치에서 일직선 위에 아직 안 지난 지점들이 차례로 놓인 데까지 (공동 직선 구간 끝을 넘지 않음).
    일직선이 아니면 다음 지점 하나만 목표로 해서, 지점을 건너뛰지 않는다.
    중계기: 최대 lead 단계 앞의 대형 위치."""
    last = len(waypoints) - 1
    end_stage = run_end(points, waypoints, initial, stage, ns, cap)
    relay_stage = min(min(stage, last) + lead, end_stage)
    targets = waypoints[end_stage].copy()
    targets[ns:] = waypoints[relay_stage, ns:]
    for i in range(ns):
        first = min(reached[i] + 1, last)
        if first > end_stage:
            continue
        node = end[i]
        goal = first
        direction = np.zeros(2)
        known = False
        along = 0.
        for k in range(first, end_stage + 1):
            d = points[waypoints[k, i]] - points[node]
            length = np.sqrt((d * d).sum())
            if length < 1e-9:
                goal = k
                continue
            if not known:
                direction = d / length
                known = True
            cross = abs(d[0] * direction[1] - d[1] * direction[0])
            ahead = d[0] * direction[0] + d[1] * direction[1]
            if cross > 1e-6 or ahead < along - 1e-9:
                break
            along = ahead
            goal = k
        targets[i] = waypoints[goal, i]
    for r in range(len(offsets)):
        for target in relay_options[relay_stage, r, offsets[r]]:
            if target >= 0 and not (targets[:ns + r] == target).any():
                targets[ns + r] = target
                break
    return targets


@njit(cache=True)
def preview_run(points, predecessor, neighbors, start, end, elapsed, age, waypoints, relay_options, initial,
                stage, reached, ns, old_offsets, new_offsets, commit_frames, stalled, times, lead, cap, stop, hold):
    """기존 preview와 같은 순서(지점 판정 → 위치 기록 → 목표 → 이동)로 통과 비행을 미리 계산한다.
    stop: 이 지점 번호를 넘는 직선 구간은 예약하지 않는다. hold: 제자리에 세워 두는 기체 (고도 조정 대기).
    세워 두는 기체가 있으면 정체 처리 없이 예약된 이동만 이어 간다 (제출본 HeightNavigator와 같은 규칙)."""
    a, b, e, wait = start.copy(), end.copy(), elapsed.copy(), age.copy()
    out = np.empty((len(times), len(a), 2))
    sample = 0
    limit = int(np.rint(times[-1] / DT))
    for frame in range(limit + 1):
        if frame:
            stage_next = progress(points, a, b, e, waypoints, reached, ns)
            stalled = stalled + 1 if stage_next == stage else 0
            stage = stage_next
        if sample < len(times) and abs(frame * DT - times[sample]) < 1e-7:
            out[sample], _ = poses(points, a, b, e)
            sample += 1
        if frame < limit:
            offsets = old_offsets if frame < commit_frames else new_offsets
            targets = run_goals(points, waypoints, relay_options, initial, stage, reached, b, offsets, ns, lead,
                                min(cap, stop - stage))
            if hold.any():
                for i in range(len(b)):
                    if hold[i]:
                        targets[i] = b[i]
                advance(points, predecessor, a, b, e, targets, wait, ns, DT)
            else:
                advance_mission(points, predecessor, neighbors, a, b, e, wait, targets, ns, stalled)
    return out


class PassNavigator(Navigator):
    """Navigator와 같은 자리에서 쓰인다. 지점 판정, 목표 선택, 미리보기만 통과 비행용으로 바꾼다."""

    def __init__(self, city, ns, survey):
        super().__init__(city, ns, survey)
        self.motion = Fleet(city, survey.initial, ns)
        self.reached = np.full(ns, -1, np.int64)

    def update(self, time_s):
        motion = self.motion
        updated = progress(self.city.points, motion.start, motion.end, motion.elapsed,
                           self.survey.waypoints, self.reached, self.ns)
        if updated > self.stage:
            nodes = self.survey.waypoints[self.stage:updated, :self.ns].ravel()
            self.covered |= self.city.visible[nodes].any(axis=0)
            self.stalled = 0
        elif self.stage < len(self.survey.waypoints):
            self.stalled += 1
        self.stage = updated
        if self.stage == len(self.survey.waypoints) and self.finished_at is None:
            self.finished_at = time_s

    def run_stop(self):
        """이 지점 번호를 넘는 직선 구간은 한 번에 예약하지 않는다. 하위 항법이 줄일 수 있다 (예: 고도 구간 진입 확인)."""
        return NO_STOP

    def run_targets(self, offsets):
        motion = self.motion
        return run_goals(self.city.points, self.survey.waypoints, self.survey.options,
                         np.asarray(self.survey.initial), self.stage, self.reached, motion.end, offsets,
                         self.ns, RELAY_LEAD, min(RUN_STAGE_CAP, self.run_stop() - self.stage))

    def step(self, offsets):
        motion = self.motion
        targets = self.run_targets(offsets)
        motion.blocked += advance_mission(self.city.points, self.city.predecessor, self.city.neighbors,
                                          motion.start, motion.end, motion.elapsed, motion.age,
                                          targets, self.ns, self.stalled)
        return motion.positions()

    def forecast(self, offsets, applied, commit_frames=3, times=TIMES, hold=None):
        motion = self.motion
        hold = np.zeros(len(motion.start), np.bool_) if hold is None else np.asarray(hold, np.bool_)
        return preview_run(self.city.points, self.city.predecessor, self.city.neighbors,
                           motion.start, motion.end, motion.elapsed, motion.age, self.survey.waypoints,
                           self.survey.options, np.asarray(self.survey.initial), self.stage, self.reached.copy(),
                           self.ns, applied, offsets, commit_frames, self.stalled, np.asarray(times, np.float64),
                           RELAY_LEAD, RUN_STAGE_CAP, self.run_stop(), hold)


def straight_chain(city, a, b):
    """a에서 b까지가 원래 안전 간선만으로 이어진 일직선인가."""
    if city.graph[a, b] > 0:
        return True
    pa, pb = city.points[a], city.points[b]
    u = (pb - pa) / max(1e-10, float(np.linalg.norm(pb - pa)))
    node, steps = int(a), 0
    while node != b and steps < 64:
        nxt = int(city.predecessor[b, node])
        if nxt < 0 or not city.graph[node, nxt] > 0:
            return False
        d = city.points[nxt] - city.points[node]
        if float(np.dot(d, u)) < 0.9999 * float(np.linalg.norm(d)):
            return False
        node, steps = nxt, steps + 1
    return node == b


class StraightChainGraph:
    """비행 후 안전 검사(fast_halfsec/runner.py)는 사용한 구간이 안전 간선인지 city.graph[a, b] > 0으로 본다.
    통과 비행은 여러 칸을 한 구간으로 날므로, 그 구간이 안전 간선의 일직선 연결이면 통과로 본다."""

    def __init__(self, city):
        self._city = city

    def __getitem__(self, key):
        a, b = key
        return 1. if straight_chain(self._city, int(a), int(b)) else 0.


class PassCity:
    """엔진에 넘기는 지도. graph만 StraightChainGraph로 보이고 나머지는 원래 지도 그대로다."""

    def __init__(self, city):
        object.__setattr__(self, '_city', city)
        object.__setattr__(self, 'graph', StraightChainGraph(city))

    def __getattr__(self, name):
        return getattr(self._city, name)

    def __setattr__(self, name, value):
        setattr(self._city, name, value)

    def __copy__(self):
        # 지도 사본을 만들 때는 안쪽 지도를 복사한다 (사본에 고정 중계 위치 등을 바꿔 넣어도 원래 지도는 그대로).
        from copy import copy
        return PassCity(copy(self._city))
