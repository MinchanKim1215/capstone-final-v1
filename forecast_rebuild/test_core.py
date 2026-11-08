import unittest
import numpy as np
from .targets import labels,onsets
from .features import build,sequence
from .sensors import received_energy


class TargetTests(unittest.TestCase):
    def trajectory(self):
        a=np.ones((100,3,3),bool)
        a[:,np.arange(3),np.arange(3)]=True
        return a

    def test_first_and_repeated_onsets_have_different_operational_targets(self):
        a=self.trajectory();a[42:44,1,0]=False;a[50:53,1,0]=False
        y=labels(a,[40],1)
        self.assertEqual(y['next_bin'][0,0,0],2)
        self.assertEqual(y['targets'][0,0,0,5],1) # second event at +5 s

    def test_reoutage_while_currently_down_is_a_valid_any_event_target(self):
        a=self.trajectory();a[36:43,1,0]=False;a[48:51,1,0]=False
        y=labels(a,[40],1)
        self.assertFalse(y['currently_up'][0,0,0])
        self.assertTrue(y['valid'][0,0,0])
        self.assertEqual(y['targets'][0,0,0,5],1)

    def test_single_bad_frame_is_not_a_persistent_outage(self):
        a=self.trajectory();a[46,1,0]=False
        self.assertFalse(onsets(a).any())

    def test_half_open_3to7_boundary(self):
        a=self.trajectory();a[54:56,1,0]=False
        y=labels(a,[40],1)
        self.assertEqual(y['targets'][0,0,0,5],0) # exactly 7 seconds excluded
        self.assertEqual(y['targets'][0,0,0,3],1)

    def test_unknown_tail_and_self_never_become_negative_training_labels(self):
        a=self.trajectory();y=labels(a,[40,84],1)
        self.assertFalse(y['valid'][1].any())
        self.assertFalse(y['valid'][:,0,1].any())


class CausalityTests(unittest.TestCase):
    def test_peer_energy_requires_received_beacon_and_uses_previous_sample(self):
        own=np.arange(18).reshape(6,3).astype(float);valid=np.ones_like(own,bool)
        counts=np.zeros((6,2,3),int);counts[2,0,0]=1
        e=received_energy(own,valid,counts,1)
        self.assertTrue(np.isnan(e[1,0,0,1]))
        self.assertEqual(e[2,0,0,1],own[1,0])
        self.assertEqual(e[2,0,0,2],1.)
        self.assertEqual(e[2,0,0,5],.5)
        self.assertNotEqual(e[2,0,0,1],own[2,0])

    def test_future_sensor_changes_cannot_change_previous_observations(self):
        a=np.zeros((20,3));m=np.ones_like(a,bool);counts=np.ones((20,2,3),int)
        first=received_energy(a,m,counts,1)
        a[10:]=100
        second=received_energy(a,m,counts,1)
        np.testing.assert_equal(first[:10],second[:10])

    def test_future_rf_frames_cannot_change_features_or_sequence(self):
        rng=np.random.default_rng(4)
        values=rng.normal(size=(70,2,3,48)).astype(np.float32)
        masks=np.ones_like(values,bool);intents=np.zeros((70,2,3,9),np.float32)
        ends=np.array([40]);geometry=np.zeros((1,2,3,46),np.float32);geometry[...,-1]=1
        rows=np.array([[0,0,0],[0,1,0]])
        a=build(values,masks,intents,ends,geometry,rows)
        sa=sequence(values,masks,intents,ends,rows)
        values[40:]=1e9;intents[40:]=1e9
        np.testing.assert_equal(a,build(values,masks,intents,ends,geometry,rows))
        np.testing.assert_equal(sa,sequence(values,masks,intents,ends,rows))
        self.assertEqual(sa.shape,(2,40,111))


class AlarmTests(unittest.TestCase):
    def test_sensor_alarm_requires_fresh_physically_ordered_measurements(self):
        from types import SimpleNamespace
        from .calibration import alarm_support
        energy=np.array([[[[-95.,-96.,1.,1.,1.,.5]]]],np.float32)
        ep=SimpleNamespace(legacy_support=np.ones((1,1,1),bool),energy=energy,ends=np.array([1]))
        self.assertTrue(alarm_support(ep,True).item())
        for column,value in [(5,3.),(5,np.nan),(2,.5),(2,4.),(3,0.),(4,0.)]:
            old=energy[0,0,0,column];energy[0,0,0,column]=value
            self.assertFalse(alarm_support(ep,True).item())
            energy[0,0,0,column]=old

    def test_strict_alarm_counts_match_preserved_protocol(self):
        from .metrics import alert_counts
        from adaptive_joint_expanded.metrics import alert_counts as original
        rng=np.random.default_rng(78);links=rng.random((240,4,4))>.06
        ends=np.arange(40,240,2);score=rng.random((len(ends),2,4))
        support=np.ones_like(score,bool);support[:,0,2]=False;support[:,1,3]=False
        p=np.zeros((*score.shape,4));p[...,2]=score
        old=original(dict(links=links,aux_endpoints=ends,aux_support=support,aux_probabilities=p),2,enter=.7)
        new=alert_counts(links,ends,score,support,2,.7)
        self.assertEqual(old,new)

    def test_empty_or_weak_alarm_set_cannot_pass_acceptance(self):
        from .calibration import meets
        base=dict(first_alerts=100,precision=.95,event_recall_3s=.85,false_or_late_per_drone_hour=.5)
        self.assertTrue(meets(base))
        for changed in [dict(first_alerts=0,precision=None),dict(event_recall_3s=.1),dict(precision=.6),dict(false_or_late_per_drone_hour=10)]:
            self.assertFalse(meets(dict(base,**changed)))


if __name__=='__main__':unittest.main()
