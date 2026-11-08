"""Meaningful safety, causal-clock and air-capacity regression checks."""
import unittest
from types import SimpleNamespace
import numpy as np
from .motion import profile,duration,poses,advance,forecast
from .sensing import Sensors,Commands
from .runner import intervals
from .source import fine_shadow
from adaptive_joint_expanded.baseline import paired_shadow


class Contracts(unittest.TestCase):
    def test_acceleration_limited_turn_and_stop(self):
        for length in (10.,np.sqrt(200.)):
            t=np.arange(0,duration(length)+.10001,.1)
            distance,speed=np.array([profile(length,x) for x in t]).T
            self.assertAlmostEqual(distance[-1],length)
            self.assertLessEqual(speed.max(),5.+1e-9)
            self.assertLessEqual(np.abs(np.diff(speed)/.1).max(),2.5+1e-8)
            self.assertAlmostEqual(speed[-1],0.)

    def test_crossing_edge_reservation(self):
        points=np.array([[-10.,0.],[10.,0.],[0.,-10.],[0.,10.]])
        predecessor=np.full((4,4),-1,np.int32)
        predecessor[1,0]=1;predecessor[3,2]=3
        a=np.array([0,2],np.int32);b=a.copy();e=np.zeros(2);age=np.zeros(2)
        goals=np.array([1,3],np.int32)
        separation=99.
        for _ in range(100):
            advance(points,predecessor,a,b,e,goals,age,1)
            xy,_=poses(points,a,b,e)
            separation=min(separation,float(np.linalg.norm(xy[0]-xy[1])))
        self.assertGreaterEqual(separation,5.)
        np.testing.assert_array_equal(b,[1,3])

    def test_delayed_received_measurements_only(self):
        sensor=Sensors(3,1,12,.1)
        p=np.ones((3,3));power=np.full((3,3),-70.);sinr=np.full((3,3),30.)
        sensor.transmit(0,0.,p,power,sinr)
        self.assertFalse(sensor.estimate(0.).local_rf)
        sensor.deliver(.1)
        self.assertTrue(sensor.estimate(.1).local_rf)
        np.testing.assert_allclose(sensor.estimate(.1).power[[0,2],1],-70.)
        sensor.deliver(2.2)
        self.assertFalse(sensor.estimate(2.2).local_rf)

    def test_command_latency_and_no_link_loss(self):
        command=Commands(3,1,np.zeros((2,2)),5,.3)
        command.requested[:]=20.
        command.transmit(0,0.,np.zeros((3,3)))
        command.deliver(.2);self.assertFalse(command.applied.any())
        command.deliver(.3)
        np.testing.assert_array_equal(command.applied[0],[20.,20.])
        np.testing.assert_array_equal(command.applied[1],[0.,0.])

    def test_persistent_definition_in_seconds(self):
        self.assertEqual(intervals(np.array([True]+[False]*9+[True])),[])
        self.assertEqual(intervals(np.array([True]+[False]*10+[True])),[[.1,1.1]])

    def test_periodic_airtime_reservation(self):
        for frame in range(30):
            for count in (3,11,13):
                offset=(frame*20)%200
                reserved=[slot for slot in range(20) if (frame*20+slot)%200<offset+count]
                self.assertEqual(reserved,list(range(count)))

    def test_fine_shadow_is_stochastic_and_keeps_original_endpoints(self):
        case=dict(start_s=33.,spec=dict(seed=123),fine_seed=987)
        fine=fine_shadow(case,3)
        coarse=paired_shadow(123,116,3,.5,1.5,4.)[66:115]
        np.testing.assert_array_equal(fine[::5],coarse)
        linear=.8*coarse[0]+.2*coarse[1]
        self.assertGreater(np.abs(fine[1]-linear).max(),1e-3)

    def test_prospective_motion_matches_real_step_without_mutating_start(self):
        points=np.array([[0.,0.],[10.,0.],[20.,0.]])
        pred=np.array([[-1,0,1],[1,-1,1],[1,2,-1]],np.int32)
        a=np.array([0],np.int32);b=a.copy();e=np.zeros(1);age=np.zeros(1)
        goals=np.tile([2],(51,1)).astype(np.int32)
        times=np.arange(0,5.01,.1)
        expected,_=forecast(points,pred,a,b,e,age,goals,0,times)
        np.testing.assert_array_equal(a,[0]);np.testing.assert_array_equal(e,[0.])
        actual=[poses(points,a,b,e)[0]]
        for k in range(50):
            advance(points,pred,a,b,e,goals[k],age,0)
            actual.append(poses(points,a,b,e)[0])
        np.testing.assert_allclose(expected,actual,atol=1e-10)

    def test_future_and_masked_rf_never_enter_current_estimate(self):
        sensor=Sensors(3,1,12,.1)
        sensor.remote_power[0,1]=1e9;sensor.remote_sinr[0,1]=-1e9;sensor.remote_stamp[0,1]=10.
        self.assertFalse(sensor.estimate(0.).local_rf)
        self.assertTrue(np.isnan(sensor.estimate(0.).power[0,1]))

    def test_hybrid_rate_changes_wait_for_delivered_command(self):
        command=Commands(4,2,np.zeros((2,2)),5,.3)
        command.requested_fast[:]=False;command.requested_fast[[1,3]]=True
        command.transmit(0,0.,np.ones((4,4)))
        command.deliver(.2)
        np.testing.assert_array_equal(np.flatnonzero(command.fast_mask),[0,2])
        command.deliver(.3)
        np.testing.assert_array_equal(np.flatnonzero(command.fast_mask),[1,3])


if __name__=='__main__':
    unittest.main()
