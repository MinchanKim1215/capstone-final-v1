"""v3 통신 우선 비용. v2 물리·이동·패킷 엔진은 그대로 재사용한다."""
import numpy as np
from cooperative.world import ClosedLoopWorld
from cooperative.policy import critic_state
from cooperative.candidates import OFFSETS

TERM_NAMES = ('outage_s','burst','packet_quality','extra_motion','search_lag','protection','time','failure')
REWARD_SPEC = dict(outage_per_second=1.0, burst_excess_weight=.05,
                   burst_grace_s=1.,burst_ramp_s=5.,packet_quality=.5,
                   extra_motion_cap=.003,search_lag=.01,protection=.01,time=.002,
                   failure=10.,positive_alive_reward=False,
                   safety='unchanged shared hard movement constraints; no pre-correction H penalty')


def reward_terms(connected, age_s, quality, extra_motion, lag, protection, failure=False):
    connected=np.asarray(connected)
    if (connected.shape!=(6,) or not np.isfinite(connected).all() or
        not np.isin(connected,[0,1]).all() or not np.isfinite([age_s,quality,extra_motion,lag]).all() or
        not 0<=quality<=1+1e-9):
        raise ValueError('6 RF samples and bounded packet quality required')
    connected=connected.astype(bool);quality=float(np.clip(quality,0,1))
    exposure=0.
    for up in connected:
        age_s=0. if up else age_s+.5
        if not up:exposure+=.5*np.clip((age_s-1.)/5.,0.,1.)
    terms=np.array([-.5*(~connected).sum(),-.05*exposure,
                    -.5*(1-quality),-.003*np.clip(extra_motion,0,1),
                    -.01*np.clip(lag,0,4),-.01*bool(protection),-.002,
                    -10.*bool(failure)],float)
    return terms,float(age_s)


class ContinuityWorld(ClosedLoopWorld):
    def __init__(self,*args,**kwargs):
        self.reward_cursor=0;self.outage_age_s=0.
        self.previous_actions=np.zeros(5,dtype=int)
        super().__init__(*args,**kwargs)

    def _settle(self,final=False):
        super()._settle(final)
        while self.reward_cursor in self.settled:
            k=self.reward_cursor;row=self.settled[k]
            terms,self.outage_age_s=reward_terms(
                np.asarray(self.stats[k*6:(k+1)*6])[:,0],self.outage_age_s,
                row['Q'],row['extra_move'],row['search_lag'],row['protection'])
            row['reward_terms']=terms;row['reward']=float(terms.sum())
            self.reward_cursor+=1

    def metrics(self,event):
        # The base engine already adds -10 at truncation, after final settlement.
        if not event['done']:
            self.settled[self.k-1]['reward_terms'][-1]=-10.
        result=super().metrics(event)
        terms=np.stack([self.settled[k]['reward_terms'] for k in range(self.k)])
        rewards=np.array([self.settled[k]['reward'] for k in range(self.k)])
        if not np.allclose(terms.sum(1),rewards,atol=1e-12,rtol=0):
            raise ValueError('Reward component conservation failed')
        result.update(reward_objective='communications-priority-v3',
                      reward_terms_sum=dict(zip(TERM_NAMES,terms.sum(0).tolist())),
                      reward_terms_mean=dict(zip(TERM_NAMES,terms.mean(0).tolist())))
        return result


def training_state(env,use_risk=True):
    state=critic_state(env,use_risk)
    state[171:181]=(OFFSETS[env.previous_actions]/40).ravel()
    age=0.
    for row in reversed(env.stats[-20:]):
        if row[0]:break
        age+=.5
    state[181]=min(age/10,1.)
    return state
