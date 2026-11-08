import unittest
from types import SimpleNamespace
import numpy as np
from .diagnose import observability,severity
from .metrics import alert_counts


class DiagnosticTests(unittest.TestCase):
    def test_no_positive_flight_does_not_turn_into_zero_ap(self):
        from .analyze_results import macro_group
        empty=dict(first_alerts=0,timely_correct_first_alerts=0,eligible_directed_link_events=0,event_hits_3s=0,drone_hours=1.)
        cases=[dict(region='one',next5={'ap':None},any3to7={'ap':None},alerts=empty),
               dict(region='two',next5={'ap':.6},any3to7={'ap':.4},alerts=empty)]
        outcome=macro_group(cases)
        self.assertEqual(outcome['macro_flight_ap5'],.6)
        self.assertEqual(outcome['macro_flight_ap37'],.4)
        self.assertEqual(outcome['ap5_defined_flights'],1)
        self.assertEqual(outcome['flights'],2)

    def test_duration_groups_retain_the_strict_event_denominator(self):
        links=np.ones((200,3,3),bool);links[60:72,1,0]=False;links[90:92,1,0]=False
        ends=np.arange(40,200,2);support=np.ones((len(ends),2,3),bool)
        support[:,0,1]=False;support[:,1,2]=False
        energy=np.ones((200,2,3,6),np.float32);energy[...,0:2]=-95;energy[...,5]=.5
        ep=SimpleNamespace(links=links,ends=ends,legacy_support=support,available=support,energy=energy,
                           values=np.zeros((200,2,3,48)),record={'n_search':1})
        score=np.zeros_like(support,dtype=np.float32);score[ends==52,0,0]=.9;score[ends==80,0,0]=.9
        groups=severity(ep,score);strict=alert_counts(links,ends,score,support,1)
        self.assertEqual(sum(r['events'] for r in groups.values()),strict['eligible_directed_link_events'])
        self.assertEqual(sum(r['first_alert_hits_3s'] for r in groups.values()),strict['event_hits_3s'])
        self.assertEqual(groups['duration_1to2s']['events'],1)
        self.assertEqual(groups['duration_at_least5s']['events'],1)
        self.assertEqual(observability(ep)['legacy'],2)


if __name__=='__main__':unittest.main()
