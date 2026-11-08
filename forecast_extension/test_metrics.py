import unittest
import numpy as np
from .metrics import alert_counts
from forecast_rebuild.metrics import alert_counts as preserved


class ExactFirstAlertRule(unittest.TestCase):
    def test_float32_score_with_float64_quantile_boundary(self):
        links=np.ones((160,3,3),bool);ends=np.arange(40,160,2)
        for value in (.25,.5,.8):
            p=np.full((len(ends),2,3),value,np.float32);m=np.ones_like(p,bool)
            for threshold in (float(np.nextafter(float(p[0,0,0]),1.)),float(p[0,0,0]),float(np.nextafter(float(p[0,0,0]),0.))):
                self.assertEqual(alert_counts(links,ends,p,m,1,threshold),preserved(links,ends,p,m,1,threshold))

    def test_random_missing_hysteresis_and_boundaries_match_frozen_rule(self):
        rng=np.random.default_rng(591)
        for case in range(30):
            frames=int(rng.integers(40,600));n=int(rng.integers(3,8));ns=int(rng.integers(1,n))
            links=rng.random((frames,n,n))>.2
            for q in range(30,frames-3,30):links[q:q+2]=False
            ends=np.arange(40,frames,2);score=rng.random((len(ends),n-ns,n));support=rng.random(score.shape)>.1
            score[rng.random(score.shape)<.03]=np.nan
            threshold=(.01,.25,.5,.8,.99)[case%5]
            self.assertEqual(alert_counts(links,ends,score,support,ns,threshold),preserved(links,ends,score,support,ns,threshold))


if __name__=='__main__':unittest.main()
