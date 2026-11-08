import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from adaptive_joint_expanded.contracts import (DEFAULT,DEPLOYMENTS,deployment_mask,entry_for,
                                      plan,public_context,atomic_json,read_json,seed_all)
from adaptive_joint_expanded.targets import timing_labels,probabilities,binary_targets
from adaptive_joint_expanded.experiment import development_splits,validate_config
from adaptive_joint_expanded.metrics import alert_counts,summarize_alerts
from adaptive_joint_expanded.model import Controller
from adaptive_joint_expanded import ARTIFACTS


class TimingTests(unittest.TestCase):
    def test_first_persistent_event_and_half_open_boundaries(self):
        links=np.ones((60,2,2),bool);links[18:20,1,0]=False
        y,valid=timing_labels(links,[10],1)
        self.assertEqual(y[0,0,0],8) # precisely 4 s ahead
        self.assertTrue(valid[0,0,0]);self.assertFalse(valid[0,0,1])
        self.assertFalse(binary_targets(y)[0,0,0,1])
        self.assertTrue(binary_targets(y)[0,0,0,2])
        links[14:16,1,0]=False
        y,_=timing_labels(links,[10],1)
        self.assertEqual(y[0,0,0],4) # first event, not the later 4 s event

    def test_single_frame_drop_is_not_a_persistent_event(self):
        links=np.ones((60,2,2),bool);links[12,1,0]=False
        y,_=timing_labels(links,[10],1)
        self.assertEqual(y[0,0,0],16)

    def test_unknown_confirmation_tail_is_not_a_negative_label(self):
        links=np.ones((30,2,2),bool)
        _,valid=timing_labels(links,[14],1)
        self.assertFalse(valid.any())
        links[9,1,0]=False
        _,valid=timing_labels(links,[10],1)
        self.assertFalse(valid[0,0,0])

    def test_bins_aggregate_consistently(self):
        logits=torch.full((1,17),-100.)
        logits[0,8]=0
        p=probabilities(logits)
        self.assertAlmostEqual(float(p[0,0]),1.)
        self.assertAlmostEqual(float(p[0,1]),0.)
        self.assertAlmostEqual(float(p[0,2]),1.)


class BoundaryTests(unittest.TestCase):
    def test_initial_sensor_roles_follow_the_selected_search_count(self):
        from adaptive_joint_expanded.world import World
        from adaptive_joint_expanded.observation import Observer
        for d in DEPLOYMENTS:
            env=World(region='ansan_bono',deployment=d.key,observer=Observer(),max_steps=1)
            for i in range(d.n):
                for j in range(d.n):
                    if i!=j:
                        self.assertEqual(env.last_x[i,j,3],i>=d.ns)
                        self.assertEqual(env.last_x[i,j,4],j>=d.ns)

    def test_fleet_budget_masks_the_unavailable_deployments(self):
        self.assertEqual(deployment_mask(6,3).tolist(),[False,True,False,True])
        self.assertEqual(deployment_mask(10,6).tolist(),[False,False,True,False])
        with self.assertRaises(ValueError):deployment_mask(5,3)

    def test_test_geometry_is_sealed(self):
        key=next(m['key'] for m in plan()['maps'] if m['joint_split']=='test')
        with self.assertRaises(ValueError):entry_for(key)

    def test_geography_groups_do_not_cross_development_folds(self):
        folds=development_splits()
        self.assertEqual(sum(map(len,folds.values())),sum(m['joint_split']=='validation' for m in plan()['maps']))
        self.assertLessEqual(max(map(len,folds.values()))-min(map(len,folds.values())),1)
        seen={}
        for fold,keys in folds.items():
            for m in plan()['maps']:
                if m['key'] in keys:
                    group=m['joint_spatial_group']
                    self.assertEqual(seen.setdefault(group,fold),fold)

    def test_selector_cannot_observe_seed_jammer_or_outcomes(self):
        entry=next(m for m in plan()['maps'] if m['joint_split']=='train')
        first=public_context(entry,'east_west')
        poisoned=copy.deepcopy(entry)
        poisoned.update(seed=999,jammer=999,outage_total_s=999,pdr=0.,joint_split='test',key='secret')
        np.testing.assert_array_equal(first,public_context(poisoned,'east_west'))

    def test_atomic_json_does_not_accept_nonfinite_performance(self):
        root=ARTIFACTS/'unit_tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            self.assertEqual(Path(folder).resolve().parent,root.resolve())
            path=Path(folder)/'record.json'
            atomic_json(path,dict(ready=True))
            with self.assertRaises(ValueError):atomic_json(path,dict(value=float('nan')))
            self.assertEqual(read_json(path),dict(ready=True))

    def test_invalid_optimizer_and_budget_settings_fail(self):
        for change in [dict(actor_lr=0),dict(batch_episodes=3),dict(available_drones=5),dict(epochs=0)]:
            with self.assertRaises(ValueError):validate_config(dict(DEFAULT,**change))

    def test_argmax_evaluation_data_cannot_enter_ppo(self):
        from adaptive_joint_expanded.learning import update
        with self.assertRaises(ValueError):
            update(None,None,[(None,dict(deterministic_policy=True))],DEFAULT,1)


class AlarmTests(unittest.TestCase):
    def arrays(self):
        links=np.ones((200,2,2),bool)
        links[70:73,1,0]=False # 35 s
        endpoints=np.arange(40,180,2)
        p=np.zeros((len(endpoints),1,2,4),np.float32)
        support=np.ones((len(endpoints),1,2),bool);support[:,:,1]=False
        p[np.flatnonzero(endpoints==60)[0],0,0,2]=.9 # 30 s; 5 s lead
        return dict(links=links,aux_endpoints=endpoints,aux_probabilities=p,aux_support=support)

    def test_no_alert_precision_is_undefined_and_no_recall_credit(self):
        a=self.arrays();a['aux_probabilities'][:]=0
        result=summarize_alerts([alert_counts(a,1)])
        self.assertIsNone(result['timely_first_alert_precision'])
        self.assertEqual(result['event_recall_3s'],0.)

    def test_one_and_three_second_exposure_matches(self):
        a=self.arrays();one=alert_counts(a,1,cadence=1);three=alert_counts(a,1,cadence=3)
        self.assertEqual(one['drone_hours'],three['drone_hours'])
        self.assertEqual(one['event_hits_3s'],1)
        self.assertEqual(three['event_hits_3s'],1)

    def test_future_label_validity_cannot_turn_a_live_alert_off(self):
        a=self.arrays();before=alert_counts(a,1)
        a['aux_valid']=np.zeros_like(a['aux_support'])
        self.assertEqual(before,alert_counts(a,1))

    def test_summary_streams_episodes_without_consuming_them_twice(self):
        from adaptive_joint_expanded.metrics import summarize
        a=self.arrays()
        a['aux_labels'],a['aux_valid']=timing_labels(a['links'],a['aux_endpoints'],1)
        row={k:1. for k in ['search_connected_mean','pdr','timely_pdr','outage_total_s','outage_max_s',
                             'duration_s','delay_mean_s','delay_p95_s','relay_travel_m','observable_coverage']}
        row.update(region='synthetic-test',search_drones=1,total_drones=2,smoke_only=True,mission_completed=True,audit=dict(passed=True))
        result=summarize(iter([(a,row)]))
        self.assertGreater(result['prediction']['next_onset_3to7s']['samples'],0)
        self.assertEqual(result['alarms_at_threshold']['1']['first_alerts'],1)

    def test_an_unmeasured_legacy_forecast_is_not_reported_as_zero_recall(self):
        from adaptive_joint_expanded.metrics import summarize
        a=self.arrays()
        a['aux_endpoints']=np.zeros(0,int)
        a['aux_probabilities']=np.zeros((0,1,2,4),np.float32)
        a['aux_support']=np.zeros((0,1,2),bool)
        row={k:1. for k in ['search_connected_mean','pdr','timely_pdr','outage_total_s','outage_max_s',
                             'duration_s','delay_mean_s','delay_p95_s','relay_travel_m','observable_coverage']}
        row.update(region='legacy-synthetic-test',search_drones=1,total_drones=2,smoke_only=True,mission_completed=True,audit=dict(passed=True))
        result=summarize([(a,row)])
        self.assertIsNone(result['alarms_at_threshold'])
        self.assertFalse(result['compatible_prediction_stream_measured'])


if __name__=='__main__':unittest.main()
