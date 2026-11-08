import unittest
import numpy as np
from .belief import shift_positions,noisy_energy


class ObservationConsistency(unittest.TestCase):
    def test_unknown_peer_commands_remain_unknown_under_gps_error(self):
        v=np.zeros((50,2,4,48),np.float16);m=np.ones_like(v,bool);c=np.zeros((50,2,4,9),np.float16)
        c[...,0]=10;c[...,6]=10;c[:,0,2]=0;c[:,1,3]=0
        c[20:,0,0,8]=1;c[20:,0,0,2]=12;c[20:,0,0,7]=12
        bias=np.array([[1,2],[2,3],[3,4],[4,5]],float)
        a,b=shift_positions(v,m,c,2,bias)
        unknown=c[...,8]==0
        np.testing.assert_equal(b[...,2:4][unknown],c[...,2:4][unknown])
        np.testing.assert_equal(b[...,7][unknown],c[...,7][unknown])
        np.testing.assert_equal(b[:,0,2],c[:,0,2]);np.testing.assert_equal(b[:,1,3],c[:,1,3])

    def test_future_frames_do_not_change_past_noisy_energy(self):
        e=np.ones((30,2,4,6),np.float32);e[...,:2]=-100;e[...,5]=.5
        a=noisy_energy(e,2,np.random.default_rng(7),'noise')
        e[15:,...,:2]=10
        b=noisy_energy(e,2,np.random.default_rng(7),'noise')
        np.testing.assert_equal(a[:15],b[:15])
        missing=noisy_energy(e,2,np.random.default_rng(7),'missing')
        self.assertTrue(np.isnan(missing[...,:3]).all());self.assertFalse(missing[...,3:5].any())


if __name__=='__main__':unittest.main()
