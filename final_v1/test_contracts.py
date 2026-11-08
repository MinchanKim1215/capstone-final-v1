"""Regression checks for event bin edges and the command fallback."""
import unittest

import numpy as np

from .causes import pattern
from .policy import timing_labels
from .transport import CommandChannel


class ProtocolTests(unittest.TestCase):
    def test_patterns_keep_short_history_and_mixed(self):
        self.assertEqual(pattern(np.r_[np.repeat(14.,4),np.repeat(8.,8),-3.]),"mixed")
        self.assertEqual(pattern(np.r_[np.repeat(14.,12),-3.]),"abrupt")
        self.assertEqual(pattern(np.linspace(5.,-.5,13)),"gradual")
        self.assertEqual(pattern(np.array([.5,-.5])),"insufficient_history")

    def test_outage_time_edges(self):
        for lead, expected in ((.5,1),(2.5,1),(3.,2),(4.5,2),(5.,3),(6.5,3),(7.,0)):
            net = np.ones(100,dtype=bool)
            self.assertEqual(timing_labels(np.array([21.]),np.array([21.+lead]),net).item(),expected)
            net[41] = False
            self.assertEqual(timing_labels(np.array([21.]),np.array([21.+lead]),net).item(),0)

    def test_missed_command_ages_into_local_then_nominal(self):
        channel = CommandChannel(5, 4, 2, 300)
        observation = dict(action_mask=np.ones((2,25),dtype=bool),teacher=np.zeros((2,25)))
        observation["teacher"][1,2] = 1.
        requested = np.array([3,4])
        rf = np.array([True,True])
        first = channel.apply(1,np.ones((4,4)),requested,observation,rf,"local")
        self.assertEqual(first[0].tolist(),[3,4])
        missed = channel.apply(2,np.zeros((4,4)),requested,observation,rf,"local")
        self.assertEqual(missed[4].tolist(),["received","hold_one_cycle"])
        self.assertEqual(missed[0].tolist(),[3,4])
        stale = channel.apply(3,np.zeros((4,4)),requested,observation,rf,"local")
        self.assertEqual(stale[0].tolist(),[3,2])
        self.assertEqual(stale[4].tolist(),["received","local_radio"])
        disconnected = channel.apply(4,np.zeros((4,4)),requested,observation,~rf,"local")
        self.assertEqual(disconnected[0].tolist(),[3,0])


if __name__ == "__main__":
    unittest.main()
