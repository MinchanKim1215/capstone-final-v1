"""Regressions for pass-through motion: long-run reservations, waypoint passing and the edge audit."""

from types import SimpleNamespace
import unittest

import numpy as np

from final_v1_fast import motion as original
from . import motion_pass
from .navigation_pass import StraightChainGraph, passed, progress, run_goals


def line_points(count, spacing=10.):
    return np.array([[spacing * k, 0.] for k in range(count)])


def line_predecessor(count):
    """predecessor[goal, node]: 직선 위에서 goal 쪽으로 한 칸."""
    table = np.full((count, count), -1, np.int64)
    for goal in range(count):
        for node in range(count):
            if node != goal:
                table[goal, node] = node + 1 if goal > node else node - 1
    return table


class ReservationTests(unittest.TestCase):
    def test_parking_on_a_long_run_is_refused_after_the_movers_own_window(self):
        # 기체 1이 0 m에서 100 m까지 긴 직선 이동 중(약 22초). 기체 0이 80 m 지점 옆에서 그 지점으로 오려 한다.
        # 기체 1은 약 17초 뒤에 80 m를 지나므로, '내 이동 + 5초'(약 11초)만 보는 원본 검사는 충돌을 놓친다.
        points = np.vstack([line_points(11), [[80., 20.]]])
        start = np.array([11, 0])
        end = np.array([11, 10])
        elapsed = np.zeros(2)
        self.assertTrue(original.reserve_ok(points, start, end, elapsed, 0, 8))
        self.assertFalse(motion_pass.reserve_ok(points, start, end, elapsed, 0, 8))

    def test_single_edges_keep_the_original_answer(self):
        points = np.array([[0., 0.], [10., 0.], [20., 0.], [0., 20.]])
        start, end, elapsed = np.array([0, 3]), np.array([0, 3]), np.zeros(2)
        for target in (1, 2):
            self.assertEqual(original.reserve_ok(points, start, end, elapsed, 0, target),
                             motion_pass.reserve_ok(points, start, end, elapsed, 0, target))

    def test_a_drone_flies_a_straight_run_in_one_reservation(self):
        count = 8
        points, predecessor = line_points(count), line_predecessor(count)
        start, end = np.array([0]), np.array([0])
        elapsed, age = np.zeros(1), np.zeros(1)
        motion_pass.advance(points, predecessor, start, end, elapsed, np.array([7]), age, 1, .1)
        self.assertEqual(end[0], 7)


class WaypointTests(unittest.TestCase):
    def test_waypoints_count_as_passed_while_flying_over_them(self):
        points = line_points(6)
        start, end = np.array([0]), np.array([5])
        travelled = 25.
        elapsed = np.array([2. + (travelled - 5.) / 5.])  # 가속 2초(5 m) 후 5 m/s
        self.assertTrue(passed(points, start, end, elapsed, 0, 2))
        self.assertFalse(passed(points, start, end, elapsed, 0, 3))
        waypoints = np.arange(1, 6, dtype=np.int32)[:, None]
        reached = np.full(1, -1, np.int64)
        self.assertEqual(progress(points, start, end, elapsed, waypoints, reached, 1), 2)

    def test_an_off_route_searcher_heads_for_its_next_waypoint_first(self):
        points = np.vstack([line_points(6), [[30., 10.]]])
        waypoints = np.arange(1, 6, dtype=np.int32)[:, None]
        initial = np.array([0], np.int32)
        options = np.full((5, 0, 1, 12), -1, np.int32)
        offsets = np.zeros(0, np.int64)
        on_route = run_goals(points, waypoints, options, initial, 0, np.full(1, -1, np.int64),
                             np.array([0]), offsets, 1, 3, 20)
        self.assertEqual(on_route[0], 5)
        off_route = run_goals(points, waypoints, options, initial, 0, np.full(1, -1, np.int64),
                              np.array([6]), offsets, 1, 3, 20)
        self.assertEqual(off_route[0], 1)


class AuditTests(unittest.TestCase):
    def test_straight_chains_of_safe_edges_pass_and_bent_ones_fail(self):
        points = np.array([[0., 0.], [10., 0.], [20., 0.], [20., 10.]])
        graph = np.zeros((4, 4))
        for a, b in ((0, 1), (1, 2), (2, 3)):
            graph[a, b] = graph[b, a] = 1.
        predecessor = np.full((4, 4), -1, np.int64)
        path = [0, 1, 2, 3]
        for goal in path:
            for node in path:
                if node != goal:
                    predecessor[goal, node] = node + 1 if goal > node else node - 1
        view = StraightChainGraph(SimpleNamespace(points=points, graph=graph, predecessor=predecessor))
        self.assertGreater(view[0, 1], 0)
        self.assertGreater(view[0, 2], 0)
        self.assertEqual(view[0, 3], 0)


if __name__ == '__main__':
    unittest.main()
