"""실행 가능한 관측만으로 경고를 만든 뒤 미래 정답과 대조하는 평가.

true current availability는 후보 생성/모델 입력/경고 판정에 사용하지 않습니다.
오프라인 현재 가용 링크 성능과 실제 경고 모드 성능을 구분합니다.
"""
import json
import numpy as np
from scipy.special import expit
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from upgrade.evaluate import event_rows,summarize_events
from swarm_sim.forecast_evaluation import binary_metrics
from usable.data import collection,index_episode,load_observation
from usable.evaluate import load,calibrate,predict_episode,RESULT,RUNS
from upgrade.train import configure


def alarm_onsets(index,score,threshold,n):
    """1초 결정에서 연속 경고의 첫 시각만 별도 알림으로 집계합니다."""
    out=[]
    for i in range(n):
        for j in range(n):
            if i==j:continue
            q=np.flatnonzero((index[:,1]==i)&(index[:,2]==j));k=index[q,0];active=score[q]>=threshold
            previous=np.r_[False,active[:-1]&(np.diff(k)==2)]
            out.extend((int(k[t]),i,j) for t in np.flatnonzero(active&~previous))
    return out


def onset_metrics(onsets,available,dt=.5):
    tp=0;late=0;leads=[]
    for k,i,j in onsets:
        bad=~available[:,i,j];starts=np.flatnonzero(np.diff(np.r_[False,bad].astype(np.int8))==1)
        candidates=[]
        for start in starts:
            # 학습 정답과 같은 5초 창 안에 최소 1초 지속까지 포함해야 합니다.
            if start+1<len(bad) and bad[start+1] and 0<=(start-k-1)*dt and (start-k-1)*dt+1.<=5:
                candidates.append((start-k-1)*dt)
        if candidates:tp+=1;leads.append(max(candidates))
        elif not available[k,i,j]:late+=1
    return {'onset_count':len(onsets),'onsets_with_new_outage_within_5s':tp,'false_or_late_onsets':len(onsets)-tp,
            'already_unavailable_onsets_without_future_new_outage':late,
            'onset_precision':tp/len(onsets) if onsets else None,
            'successful_onset_lead_median_s':float(np.median(leads)) if leads else None}


def main():
    configure(11,2);selection=read_json(RESULT/'selection.json');name=selection['selected_variant']+'_s11'
    folder=RUNS/name;model,state,_=load(folder);cal=calibrate(folder);manifest,p=collection()
    all_y=[];all_prob=[];all_region=[];balanced=[];tight=[];rows=[];total_s=0;total_candidates=0;total_observed=0
    dest=folder/'runtime';dest.mkdir(exist_ok=True)
    totals={mode:{'onset_count':0,'onsets_with_new_outage_within_5s':0,'false_or_late_onsets':0,'already_unavailable_onsets_without_future_new_outage':0} for mode in ('balanced','conservative')}
    for r in [e for e in manifest['records'] if e['split']=='test']:
        ix=index_episode(r,p);idx=ix['index'];x,m=load_observation(r);k,i,j=idx.T
        # 세 항목 모두 성공적으로 수신한 과거 관측. 정답 배열을 조회하지 않습니다.
        eligible=m[k,i,j,46]&(x[k,i,j,46]>.5)&m[k,i,j,41]&(x[k,i,j,41]>0)
        q=np.flatnonzero(eligible);index=idx[q];total_observed+=len(idx);total_candidates+=len(index)
        path=dest/(r['id']+'.npz')
        if path.exists():
            with np.load(path) as z:
                if str(z['checkpoint_sha256'])!=sha256_file(folder/'best.pt'):raise ValueError('실행 평가 캐시의 가중치 불일치')
                np.testing.assert_array_equal(index,z['index']);prob=z['probability']
        else:
            prob=expit(predict_episode(model,state,r,index)/cal['temperature']).astype(np.float32)
            np.savez_compressed(path,index=index,y=ix['y'][q],probability=prob,checkpoint_sha256=sha256_file(folder/'best.pt'))
        with np.load(ROOT/r['path']/'truth.npz') as z:available=z['link_available']
        events=event_rows(r,index,prob,cal['threshold_balanced'],p);balanced.extend(events)
        ep={'id':r['id'],'candidate_decisions':len(index),'observed_decisions':len(idx)}
        for mode,cut in [('balanced',cal['threshold_balanced']),('conservative',cal['threshold_conservative'])]:
            if cut is None:continue
            if mode=='conservative':tight.extend(event_rows(r,index,prob,cut,p))
            onsets=alarm_onsets(index,prob,cut,available.shape[1]);v=onset_metrics(onsets,available)
            ep[mode]=v
            for key in totals[mode]:totals[mode][key]+=v[key]
        all_y.append(ix['y'][q]);all_prob.append(prob);all_region.extend([r['region']]*len(q));rows.append(ep)
        total_s+=r['metadata']['duration_s'];print('runtime',r['id'],len(q),flush=True)
    y=np.concatenate(all_y);prob=np.concatenate(all_prob)
    for mode in totals:
        c=totals[mode];c['onset_precision']=c['onsets_with_new_outage_within_5s']/max(c['onset_count'],1)
        c['false_or_late_onsets_per_drone_hour']=c['false_or_late_onsets']/(total_s/3600*8)
    out={'run':name,'eligibility':'history of >=3 received beacons; 20s history; both measured directions fresh <=2s; measured margin >0; no truth gate',
         'total_beacon_observed_decisions':total_observed,'proactive_candidate_decisions':total_candidates,
         'balanced_windows':binary_metrics(y,prob,cal['threshold_balanced']),
         'conservative_windows':binary_metrics(y,prob,cal['threshold_conservative']) if cal['threshold_conservative'] is not None else None,
         'balanced_events':summarize_events(balanced),'conservative_events':summarize_events(tight) if tight else None,
         'alarm_onsets':totals,'episodes':rows,'mission_hours':total_s/3600,
         'episode_precision_definition':'first warning matches a NEW outage with at least 1s duration fully contained in the next 5s window; late alerts not credited',
         'warning_frequency_note':'directed link notifications; two directions correlated; no debounce/cooldown'}
    write_json(RESULT/'runtime_evaluation.json',out)


if __name__=='__main__':main()
