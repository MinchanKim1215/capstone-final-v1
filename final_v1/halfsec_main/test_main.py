"""Regressions for waypoint clearance, recovery shields and delivery stress."""

from types import SimpleNamespace
import unittest

import numpy as np

from fast_halfsec.navigation import Survey
from final_v1_fast.controller import TIMES

from .mission import StressCommands, recovers_safely, steady_reconnect
from .navigation import clearance_move
from .planned_eight_motion import HeightWindow
from .submission import SubmissionNavigator


class ClearanceTests(unittest.TestCase):
    def test_detours_around_a_parked_relay(self):
        city = SimpleNamespace(
            points=np.array([[0., 0.], [10., 0.], [20., 0.], [0., 10.], [10., 10.]]),
            neighbors=np.array([[1, 3, -1], [0, 2, 4], [1, 4, -1], [0, 4, -1], [3, 1, 2]]),
        )
        self.assertEqual(clearance_move(city, np.array([0, 1]), np.array([2])), (0, 3))

    def test_clears_an_occupied_corridor_by_parking_the_relay(self):
        city = SimpleNamespace(
            points=np.array([[0., 0.], [10., 0.], [20., 0.], [10., 10.]]),
            neighbors=np.array([[1, -1, -1], [0, 2, 3], [1, -1, -1], [1, -1, -1]]),
        )
        self.assertEqual(clearance_move(city, np.array([0, 1]), np.array([2])), (1, 3))


class RecoveryTests(unittest.TestCase):
    def test_reconnection_requires_two_half_second_samples(self):
        transient = np.where(np.isclose(TIMES, .5), 1., -1.)
        stable = np.where(TIMES >= 1.5, 1., -1.)
        self.assertEqual(steady_reconnect(transient), np.inf)
        self.assertEqual(steady_reconnect(stable), 1.5)

    def test_rejects_a_new_cut_or_an_existing_pair_becoming_unreachable(self):
        baseline = np.where(TIMES >= 3., 1., -1.)
        candidate = np.where(TIMES >= 1.5, 1., -1.)
        healthy = np.ones((len(TIMES), 1))
        reference = (baseline, healthy, 3., 3.)
        self.assertTrue(recovers_safely(reference, (candidate, healthy, 1.5, 1.5)))
        lost_pair = healthy.copy()
        lost_pair[TIMES == 1.5] = -.01
        self.assertFalse(recovers_safely(reference, (candidate, lost_pair, 1.5, 1.5)))
        repeats = np.where(TIMES >= 5., -1., candidate)
        self.assertFalse(recovers_safely(reference, (repeats, healthy, 1.5, 1.5)))


class CommandTests(unittest.TestCase):
    def test_forced_loss_does_not_rewrite_radio_probabilities(self):
        channel = StressCommands(4, 1, np.zeros(3, dtype=int), 1, .1, 180.)
        channel.requested[:] = [1, 2, 3]
        probability = np.ones((4, 4))
        channel.transmit(0, 36., probability)
        channel.deliver(36.1)
        np.testing.assert_array_equal(channel.applied, [1, 0, 3])
        np.testing.assert_array_equal(probability, np.ones((4, 4)))


class AltitudeTransitionTests(unittest.TestCase):
    def navigator(self, confirm):
        initial = np.arange(5, dtype=np.int32)
        survey = Survey(initial, np.tile(initial, (2, 1)), np.zeros((2, 2, 1, 1), np.int32), 12., 1)
        city = SimpleNamespace(visible=np.ones((5, 1), bool))
        nav = SubmissionNavigator(city, 3, survey, [HeightWindow(0, 1, 2, 4, 75.)], confirm,
                                  lambda *args: True)
        nav.heights[4], nav.previous_safe = 55., 60.
        nav.prove = 10.
        return nav

    def test_next_window_resolves_descent_proof_before_climbing(self):
        nav = self.navigator(lambda now, since, relay: now >= 11. and since == 10. and relay == 4)
        nav.advance_program(10.5)
        self.assertEqual(nav.prove, 10.)
        self.assertIsNone(nav.pending)
        nav.advance_program(11.)
        self.assertIsNone(nav.prove)
        self.assertEqual(nav.pending, (75., 'climb'))

    def test_missing_descent_proof_restores_safe_height_first(self):
        nav = self.navigator(lambda *args: False)
        nav.advance_program(16.)
        self.assertIsNone(nav.prove)
        self.assertEqual(nav.pending, (60., 'recovery'))


if __name__ == '__main__':
    unittest.main()
