"""An already-high20s risk must yield the first evaluated alarm at21s."""
import unittest
import numpy as np
from .runtime_aligned import AlignedStream


class AlarmStart(unittest.TestCase):
    def test_physical_latch_starts_at_the_existing_metric_boundary(self):
        class Fake:
            def __init__(self):self.active=np.zeros(1,bool);self.qualified=np.zeros(1,bool)
            def observe(self,t):
                first=not bool(self.qualified[0]);self.active[:]=True;self.qualified[:]=True
                return dict(links=[dict(risk_5s=.9,advisory_first_alert=first,qualified_first_alert=first,trusted_alarm=False)])
        stream=AlignedStream(Fake(),'link');early=stream.observe(20);start=stream.observe(21)
        self.assertFalse(early['links'][0]['qualified_first_alert']);self.assertTrue(start['links'][0]['qualified_first_alert'])
        self.assertEqual(early['links'][0]['risk_5s'],start['links'][0]['risk_5s'])

    def test_service_latch_starts_at_the_same_boundary(self):
        class Fake:
            active=False
            def observe(self,t):
                first=not self.active;self.active=True
                return dict(service_risk_3to7s=.9,first_alert=first,trusted_alarm=False)
        stream=AlignedStream(Fake(),'service');early=stream.observe(20);start=stream.observe(21)
        self.assertFalse(early['first_alert']);self.assertTrue(start['first_alert']);self.assertEqual(early['service_risk_3to7s'],start['service_risk_3to7s'])


if __name__=='__main__':unittest.main()
