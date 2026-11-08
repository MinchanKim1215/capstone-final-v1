"""Regressions for observation timestamps and causal finite differences."""
import unittest

import numpy as np

from .features import NAMES,observed_features


class ObservedChanges(unittest.TestCase):
    def arrays(self):
        x = np.zeros((20,2,48),dtype=np.float32)
        m = np.zeros_like(x,dtype=bool)
        for field in (9,10,29,30,32,41,46):
            m[:,0,field] = True
        x[:,0,46] = 1.
        x[:,0,[10,30,32]] = .1
        x[:,0,9] = 20.-np.arange(20)
        x[:,0,29] = 20.-np.arange(20)
        x[:,0,41] = 10.-np.arange(20)
        return x,m

    def test_half_second_drop_uses_two_new_measurements(self):
        x,m = self.arrays()
        out = observed_features(x,m,1)
        col = NAMES.index("ack_inbound_0.5s_max_drop")
        self.assertTrue(np.isnan(out[0,col]))
        np.testing.assert_allclose(out[1:,col],1.)
        rate = NAMES.index("ack_inbound_0.5s_min_rate")
        np.testing.assert_allclose(out[1:,rate],-2.)

    def test_held_measurement_does_not_become_zero_slope(self):
        x,m = self.arrays()
        x[:,0,[10,30,32]] = ((np.arange(20)+1)*.5)[:,None]-.1
        out = observed_features(x,m,1)
        for channel in ("beacon_inbound","beacon_bidirectional","ack_inbound"):
            self.assertTrue(np.isnan(out[:,NAMES.index(channel+"_0.5s_max_drop")]).all())

    def test_future_and_unobserved_values_cannot_change_features(self):
        x,m = self.arrays()
        original = observed_features(x,m,1)
        np.testing.assert_array_equal(original[:12],observed_features(x[:12],m[:12],1))
        hidden = x.copy()
        hidden[~m] = 100000.
        np.testing.assert_array_equal(original,observed_features(hidden,m,1))


if __name__ == "__main__":
    unittest.main()
