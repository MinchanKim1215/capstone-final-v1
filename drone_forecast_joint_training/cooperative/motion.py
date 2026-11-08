"""최신 수색 루프의 중단 가능한 형태. 보정 0의 기존 궤적 동등성 검사 필수."""

import numpy as np
from swarm_sim.mission import (
    MOVE_DT,
    MAX_MISSION,
    SHARE_SECONDS,
    move_group,
    unique_goals,
)
from swarm_sim.motion import fleet_offsets, swept_minimum
from usable.motion import reserved_goals, recovery_step


def motion_stream(city, design, scenario):
    offset = fleet_offsets(design, scenario.direction)
    if np.any(np.ptp(offset, axis=0) > 380):
        raise ValueError("명목 대형 폭 380m 초과")
    reference = city.route[0].copy()
    cursor = 1
    desired = reference + offset
    nodes = unique_goals(city, desired)
    trajectories = [nodes.copy()]
    desired_log = [desired.copy()]
    phases = [0]
    observed = np.any(city.visible[nodes[: design.n_search]], axis=0)
    age = np.arange(len(nodes), dtype=float) * 1e-3
    goal = -1
    route_end = None
    coverage_end = None
    share_left = None
    recovery = False
    recoveries = 0
    recovery_steps = 0
    escape_left = 0
    escape_goals = None
    since_gain = 0
    escapes = 0
    # 지도/방향에 대해 고정된 탈출 난수. 평가 결과를 보고 시드를 고르지 않습니다.
    rng = np.random.default_rng(
        int(city.sha256[:8], 16) + (scenario.direction == "north_south")
    )
    mindistance = float("inf")
    travel = np.zeros(len(nodes))
    wait = 0
    clamps = 0
    route_length = float(np.linalg.norm(np.diff(city.route, axis=0), axis=1).sum())
    for step in range(1, int(MAX_MISSION / MOVE_DT) + 1):
        phase = 0
        if cursor < len(city.route):
            remaining = scenario.reference_speed_mps * MOVE_DT
            while remaining > 1e-9 and cursor < len(city.route):
                target = city.route[cursor]
                d = np.linalg.norm(target - reference)
                if d <= remaining + 1e-9:
                    reference = target.copy()
                    remaining -= d
                    cursor += 1
                else:
                    reference += (target - reference) * remaining / d
                    remaining = 0
            if cursor == len(city.route):
                route_end = step * MOVE_DT
        else:
            phase = 1
            # 기준경로가끝났어도잔여관측가능셀을실제로찾아가며수색합니다. 공유단계는그뒤입니다.
            missing = city.observable & ~observed
            if not np.any(missing):
                if coverage_end is None:
                    coverage_end = (step - 1) * MOVE_DT
                    share_left = int(SHARE_SECONDS / MOVE_DT)
                phase = 2
            else:
                if (
                    goal < 0
                    or nodes[0] == goal
                    or not np.any(city.visible[goal] & missing)
                ):
                    gains = np.sum(city.visible[:, missing], axis=1)
                    # 새로관측할셀수와이동거리의탐욕절충입니다. 프로젝트자체휴리스틱이며통신예측재배치가아닙니다.
                    score = gains / (10.0 + city.distance[nodes[0]])
                    goal = int(np.argmax(score))
                    recovery = False
                # anchor follows a safe shortest path to a remaining visible cell;
                # other drones keep their geometrical offsets to that anchor.
                reference = (
                    city.points[city.next_node(nodes[0], goal)].copy() - offset[0]
                )
        desired = reference[None, :] + offset
        # 현재 상태에서 계산한 다음 명목 목표. 자기 목표만 Actor에 전달합니다.
        correction = yield {
            "kind": "proposal",
            "nodes": nodes.copy(),
            "nominal": desired.copy(),
            "phase": phase,
            "step": step,
            "reference": reference.copy(),
            "observed_cells": int(observed.sum()),
            "observable_cells": int(city.observable.sum()),
        }
        correction = (
            np.zeros_like(desired)
            if correction is None
            else np.asarray(correction, float)
        )
        if correction.shape != desired.shape or not np.isfinite(correction).all():
            raise ValueError("유한한 [8,2] 보정 필요")
        if np.any(correction[: design.n_search] != 0):
            raise ValueError("수색 역할의 직접 제어 금지")
        desired = desired + correction
        bounded = np.clip(
            desired, -200 + scenario.grid_m / 2, 200 - scenario.grid_m / 2
        )
        clamps += int(np.any(bounded != desired, axis=1).sum())
        goals = unique_goals(city, bounded)
        if phase == 1 and goal >= 0:
            # 잔여 수색의 선두 목표를 다른 드론에 재할당하지 않습니다.
            # 전체 잔여 목표를 유지해야 한 칸 목표 교대에 따른 왕복을 피할 수 있습니다.
            goals = reserved_goals(city, bounded, goal)
        age += 1
        # 오래대기한드론에우선권을주고거리항으로순서를구분합니다. 성공적으로목표에도달하면나이를0으로초기화합니다.
        priorities = age + city.distance[nodes, goals] * 0.001
        if phase == 1:
            # 센서가 달린 선두가 잔여 셀에 도달하도록 우선권을 유지합니다.
            # 통신 품질이나 예측값을 사용한 재배치는 아닙니다.
            priorities[0] = np.max(priorities) + 100.0
        new = move_group(
            nodes,
            goals,
            priorities,
            city.neighbors,
            city.distance,
            city.points,
            scenario.min_separation_m,
        )
        if phase == 1 and len(trajectories) >= 12:
            if (
                len(np.unique(np.asarray(trajectories[-12:]), axis=0)) <= 4
                and not recovery
            ):
                recovery = True
                recoveries += 1
            if recovery:
                new = recovery_step(city, nodes, goal, scenario.min_separation_m)
                recovery_steps += 1
            # 순차 양보도 관측 이득을 내지 못하면 90초간 임시 경유점을 따릅니다.
            # 좁은 막다른 길의 마주보기 해소용이며 안전 제한은 그대로 적용합니다.
            if since_gain >= 60 and escape_left == 0:
                escape_left = 30
                escape_goals = rng.choice(
                    len(city.points), len(nodes), replace=False
                ).astype(np.int32)
                escapes += 1
                since_gain = 0
            if escape_left > 0:
                order = rng.permutation(len(nodes)).astype(float)
                new = move_group(
                    nodes,
                    escape_goals,
                    order,
                    city.neighbors,
                    city.distance,
                    city.points,
                    scenario.min_separation_m,
                )
                escape_left -= 1
                if escape_left == 0:
                    recovery = False
                    goal = -1
        oldp, newp = city.points[nodes], city.points[new]
        minimum, _ = swept_minimum(oldp, newp)
        if minimum < scenario.min_separation_m - 1e-6:
            raise RuntimeError("연속 이동 분리 거리 위반")
        mindistance = min(mindistance, minimum)
        change = np.linalg.norm(newp - oldp, axis=1)
        if np.max(change) > scenario.speed_mps * MOVE_DT + 1e-6:
            raise RuntimeError("최대 속도 위반")
        travel += change
        wait += int(np.sum(change < 1e-9))
        age[new == goals] = 0
        before_count = int(observed.sum())
        nodes = new
        observed |= np.any(city.visible[nodes[: design.n_search]], axis=0)
        since_gain = 0 if int(observed.sum()) > before_count else since_gain + 1
        trajectories.append(nodes.copy())
        desired_log.append(desired.copy())
        phases.append(phase)
        if phase == 2:
            share_left -= 1
        terminal = bool(phase == 2 and share_left <= 0)
        yield {
            "kind": "motion",
            "old": trajectories[-2].copy(),
            "new": nodes.copy(),
            "desired": desired.copy(),
            "phase": phase,
            "done": terminal,
            "time_s": step * MOVE_DT,
            "minimum_separation_m": minimum,
            "observed_cells": int(observed.sum()),
            "observable_cells": int(city.observable.sum()),
            "recovery_steps": recovery_steps,
            "escapes": escapes,
        }
        if terminal:
            break
    # 비행임무완료와공유90%달성은별도입니다. 후자는full_run의mission_success에서검사합니다.
    completed = bool(route_end is not None and np.all(observed[city.observable]))
    if completed and coverage_end is None:
        coverage_end = (len(trajectories) - 1) * MOVE_DT
    arr = np.asarray(trajectories, np.int32)
    # A graph edge is already swept-polygon checked in City; confirm each used edge.
    for a, b in zip(arr[:-1], arr[1:]):
        if np.any((a != b) & (np.asarray(city.graph[a, b]).ravel() == 0)):
            raise RuntimeError("비행 그래프 외 이동")
    return {
        "nodes": arr,
        "desired": np.asarray(desired_log),
        "phase": np.array(phases),
        "duration_s": (len(arr) - 1) * MOVE_DT,
        "mission_completed": completed,
        "route_completion_s": route_end,
        "coverage_completion_s": coverage_end,
        "route_length_m": route_length,
        "minimum_swept_separation_m": mindistance,
        "total_travel_m": float(travel.sum()),
        "search_travel_m": float(travel[: design.n_search].sum()),
        "relay_travel_m": float(travel[design.n_search :].sum()),
        "stationary_agent_frames": wait,
        "boundary_clamp_agent_frames": clamps,
        "obstacle_violations": 0,
        "cycle_recoveries": recoveries,
        "recovery_steps": recovery_steps,
        "temporary_waypoint_escapes": escapes,
    }
