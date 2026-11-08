"""모델 점수와 알림을 분리한 검증 전용 정책 선택. 신경망 재학습 없음.

첫 알림의 정밀도 >=0.8, 드론·시간당 오경보/늦은 알림 <=5를 프로젝트
운용 목표로 고정합니다. 원래 평가 자료는 정책 적합에 사용하지 않습니다.
높은 진입 문턱과 낮은 해제 문턱의 히스테리시스는 첫 경고를 지연하지
않으면서 문턱 부근의 반복 알림을 줄이기 위한 별도 상태 처리입니다.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import json
import numpy as np
from scipy.special import expit
from numba import njit
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from usable.data import collection,index_episode,load_observation,validation_partition
from usable.evaluate import RESULT,RUNS,load,calibrate,predict_episode
from upgrade.train import configure

GOALS={'minimum_first_alert_precision':.80,'maximum_false_alerts_per_drone_hour':5.,'minimum_validation_alerts':30}
ENTER=np.r_[np.linspace(.05,.95,37),.96,.97,.98,.99,.995,.999]
RATIOS=[.25,.5,.75,1.]


def arrays_for_record(r,index,score):
    with np.load(ROOT/r['path']/'truth.npz') as z:up=z['link_available']
    nt,n,_=up.shape;ends=np.arange(39,nt-20,2);lookup={int(k):q for q,k in enumerate(ends)}
    prob=np.full((len(ends),n*n),-1.,np.float32);y=np.zeros_like(prob,np.bool_)
    rows=np.array([lookup[int(k)] for k in index[:,0]],int);cols=index[:,1]*n+index[:,2];prob[rows,cols]=score
    anynew=np.zeros_like(prob,np.bool_);event3=np.full(prob.shape,-1,np.int32);events=0
    for i in range(n):
        for j in range(n):
            if i==j:continue
            bad=~up[:,i,j];boundary=np.diff(np.r_[False,bad,False].astype(np.int8))
            for start,stop in zip(np.flatnonzero(boundary==1),np.flatnonzero(boundary==-1)):
                # 모든 미래 위험 창의 라벨: 현재부터 이어진 단절도 포함합니다.
                for q,k in enumerate(ends):
                    if min(stop,k+11)-max(start,k+1)>=2:y[q,i*n+j]=True
                if stop-start<2 or start*.5<20:continue
                e=events;events+=1
                candidates=np.flatnonzero((ends<start)&(start+1<=ends+10))
                anynew[candidates,i*n+j]=True
                lead=(start-ends[candidates]-1)*.5
                event3[candidates[lead>=3],i*n+j]=e
    return prob,y,anynew,event3,events


@njit(cache=True)
def simulate_policy(prob,y,anynew,event3,n_events,enter,exit):
    active=np.zeros(prob.shape[1],np.bool_);hit=np.zeros(n_events,np.bool_);hit_onset=np.zeros(n_events,np.bool_)
    alerts=0;correct=0;latefalse=0;warn_tp=0;warn_fp=0;positive=0
    states=np.zeros(prob.shape,np.bool_)
    for t in range(len(prob)):
        for link in range(prob.shape[1]):
            p=prob[t,link]
            if p<0:active[link]=False;continue
            positive+=int(y[t,link])
            old=active[link]
            if not old and p>=enter:active[link]=True
            elif old and p<exit:active[link]=False
            if active[link]:
                states[t,link]=True;warn_tp+=int(y[t,link]);warn_fp+=int(not y[t,link])
                if event3[t,link]>=0:hit[event3[t,link]]=True
                if not old:
                    alerts+=1
                    if anynew[t,link]:correct+=1
                    else:latefalse+=1
                    if event3[t,link]>=0:hit_onset[event3[t,link]]=True
    return np.array([alerts,correct,latefalse,np.sum(hit),np.sum(hit_onset),warn_tp,warn_fp,positive,n_events],np.int64),states


def score_one(r,folder):
    configure(11,2);model,state,_=load(folder);cal=calibrate(folder);_,p=collection()
    ix=index_episode(r,p);x,m=load_observation(r);k,i,j=ix['index'].T
    eligible=m[k,i,j,46]&(x[k,i,j,46]>.5)&m[k,i,j,41]&(x[k,i,j,41]>0)
    index=ix['index'][eligible];target=folder/'alarm_calibration_scores';target.mkdir(exist_ok=True)
    path=target/(r['id']+'.npz')
    if path.exists():
        with np.load(path) as z:score=z['probability']
    else:
        score=expit(predict_episode(model,state,r,index)/cal['temperature']).astype(np.float32)
        np.savez_compressed(path,index=index,probability=score,checkpoint_sha256=sha256_file(folder/'best.pt'))
    a=arrays_for_record(r,index,score)
    print('alarm calibration',r['id'],len(index),flush=True)
    return r['id'],r['metadata']['duration_s'],a


def summarize(v,duration):
    alerts,correct,false,hit,hiton,tp,fp,positive,total=[int(x) for x in v]
    return {'first_alerts':alerts,'correct_first_alerts':correct,'false_or_late_first_alerts':false,
            'first_alert_precision':correct/alerts if alerts else None,
            'false_alerts_per_drone_hour':false/(duration/3600*8),
            'event_recall_3s':hit/max(total,1),'first_alert_event_recall_3s':hiton/max(total,1),
            'warning_window_precision':tp/max(tp+fp,1),'warning_window_recall':tp/max(positive,1),
            'all_new_directed_events':total,'warning_window_tp':tp,'warning_window_fp':fp}


def main():
    sel=read_json(RESULT/'selection.json');folder=RUNS/(sel['selected_variant']+'_s11');manifest,_=collection()
    _,validation=validation_partition([r for r in manifest['records'] if r['split']=='val'])
    plan={'goals':GOALS,'enter_thresholds':ENTER.tolist(),'exit_ratios':RATIOS,
          'selection':'maximize first-alert event recall >=3s subject to goals; then event recall >=3s, then fewer false alerts',
          'data':'21 validation-calibration whole flights only; original 48 test flights excluded',
          'model_sha256':sha256_file(folder/'best.pt'),'source_sha256':sha256_file(ROOT/'usable/alarm_policy.py'),
          'context':'notification design after initial evaluation; new random confirmation flights planned before final use'}
    planpath=ROOT/'configs/usable_alarm_policy.json'
    if planpath.exists() and read_json(planpath)!=plan:raise ValueError('알림 선택 계획 변경 금지')
    write_json(planpath,plan)
    records=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(score_one,r,folder) for r in validation]
        for f in as_completed(jobs):records.append(f.result())
    duration=sum(x[1] for x in records);rows=[]
    for enter in ENTER:
        for ratio in RATIOS:
            counts=np.zeros(9,np.int64)
            for _,_,a in records:counts+=simulate_policy(*a,enter,enter*ratio)[0]
            summary=summarize(counts,duration)
            qualifies=bool(summary['first_alerts']>=GOALS['minimum_validation_alerts'] and summary['first_alert_precision']>=GOALS['minimum_first_alert_precision'] and summary['false_alerts_per_drone_hour']<=GOALS['maximum_false_alerts_per_drone_hour'])
            rows.append({'enter_threshold':float(enter),'exit_threshold':float(enter*ratio),'exit_ratio':ratio,'qualifies':qualifies,**summary})
    valid=[r for r in rows if r['qualifies']]
    chosen=max(valid,key=lambda r:(r['first_alert_event_recall_3s'],r['event_recall_3s'],-r['false_alerts_per_drone_hour'])) if valid else None
    result={'enabled':chosen is not None,'selected':chosen,'goals':GOALS,'all_candidates':rows,'checkpoint_sha256':sha256_file(folder/'best.pt'),
            'fit_source':'disjoint validation-calibration flights only','initial_warning_delay_s':0.,
            'state_rule':'enter at p>=high; remain until p<low; reset when observations unsupported',
            'counts_note':'directed observer links; first alert requires NEW >=1s outage fully contained in next 5s',
            'literature_basis':'NHDP RFC6130 section 5.3.3 hysteresis concept; thresholds and operating goals are project choices'}
    write_json(folder/'alarm_policy.json',result);write_json(RESULT/'alarm_policy_selection.json',result)
    print('selected policy',json.dumps(chosen,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
