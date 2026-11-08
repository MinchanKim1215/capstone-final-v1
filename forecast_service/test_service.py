"""Deadline labels must never settle packets that are not due yet."""
import unittest
import numpy as np
from .replay import service_health


class DeadlineLabels(unittest.TestCase):
    def test_future_packets_do_not_change_current_service_state(self):
        born=np.arange(0,4,1/30);s=np.ones(len(born),np.int8);d=np.full(len(born),.02)
        baseline=service_health(born,s,d,8);s[born>1.75]=2;d[born>1.75]=0
        changed=service_health(born,s,d,8)
        np.testing.assert_equal(baseline['timely_pdr_1s'][:5],changed['timely_pdr_1s'][:5])
        self.assertTrue((baseline['service_healthy']).all());self.assertFalse(changed['service_healthy'][-1])

    def test_late_delivery_counts_as_deadline_failure(self):
        born=np.arange(0,3,1/30);s=np.ones(len(born),np.int8);d=np.full(len(born),.3)
        value=service_health(born,s,d,6)
        self.assertTrue(value['service_valid'][-1]);self.assertFalse(value['service_healthy'][-1]);self.assertEqual(value['timely_pdr_1s'][-1],0)

    def test_service_view_counts_one_event_not_a_duplicate_directed_pair(self):
        from forecast_rebuild.targets import labels
        from forecast_extension.metrics import alert_counts
        links=np.zeros((200,2,2),bool);links[:,1,0]=True;links[100:120,1,0]=False
        ends=np.arange(40,200,2);truth=labels(links,ends,1)
        score=np.zeros((len(ends),1,2),np.float32);score[np.flatnonzero(ends==90),0,0]=.9
        support=np.zeros(score.shape,bool);support[:,0,0]=True
        value=alert_counts(links,ends,score,support,1,.5)
        self.assertEqual(value['eligible_directed_link_events'],1);self.assertEqual(value['event_hits_3s'],1)
        self.assertEqual(value['first_alerts'],1);self.assertEqual(value['timely_correct_first_alerts'],1)
        self.assertEqual(truth['targets'][np.flatnonzero(ends==90)[0],0,0,5],1)


if __name__=='__main__':unittest.main()
