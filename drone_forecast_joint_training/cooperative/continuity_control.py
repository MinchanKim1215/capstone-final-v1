"""관측만으로 계산하는 통신 우선 후보 점수와 작은 행동 전환 관성."""
import numpy as np
from scipy.special import expit
from cooperative.candidates import observe_policy,OFFSETS

PROFILES=('linear','probability','persistent')
CONTROL_SPEC=dict(profiles=list(PROFILES),linear_displacement_cost=.02,
                  linear_travel_cost=.01,probability_temperature_db=2.,
                  switching_cost=.03,nominal_tie=.03,hold_tie=.05,
                  probability_score='10*(.65*sigmoid(b3/2)+.35*sigmoid(b6/2)) + .15*tanh(weighted/3) + .1*sigmoid(mean6/4)',
                  unavailable_peer_fallback='original local score, no invented missing observations',
                  uses_simulator_future=False)


def transform_observation(obs,previous,profile):
    if profile not in PROFILES:raise ValueError('Unknown priority profile')
    context,features,mask,old=obs
    b3,b6,mean6,weighted=[features[...,j].astype(float)*10 for j in (0,1,3,5)]
    displacement=features[...,6];travel=features[...,8]
    if profile in ('linear','persistent'):
        score=old+.14*displacement+.02*travel
    else:
        score=10*(.65*expit(b3/2)+.35*expit(b6/2))+.15*np.tanh(weighted/3)+.1*expit(mean6/4)-.02*displacement-.01*travel
    # A complete group prediction cannot be inferred when search peers are missing.
    score=np.where(features[:,0,9,None]<.999,old,score)
    distance=np.linalg.norm(OFFSETS[None]-OFFSETS[np.asarray(previous),None],axis=-1)/80.
    score=score-.03*distance
    score=np.where(mask,score,-1e6)
    return context,features,mask,score


def observe(env,use_risk=True,profile='linear'):
    return transform_observation(observe_policy(env,True,use_risk),env.previous_actions,profile)


def choose(obs,previous,persistent=False):
    mask=obs[2];score=obs[3];actions=score.argmax(-1)
    if persistent:
        for i in range(5):
            best=actions[i];old=int(previous[i])
            if mask[i,0] and score[i,0]>=score[i,best]-.03:actions[i]=0
            elif mask[i,old] and score[i,old]>=score[i,best]-.05:actions[i]=old
    return actions
