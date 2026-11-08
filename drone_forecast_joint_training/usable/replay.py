"""학습된 예측기를 실제 observe 인터페이스로 연결한 전체 임무 재생 자료."""
import json,time,csv
from pathlib import Path
import numpy as np
from swarm_sim.regions import ROOT,read_json,write_json
from usable.data import collection,load_observation
from usable.evaluate import RESULT,RUNS
from usable.predictor import LinkRiskPredictor

STATUS={'normal':1,'watch':2,'warning':3,'stale_observations':4,'currently_degraded':5,'insufficient_observations':6}


def create_case(record,model_folder):
    x,m=load_observation(record);n=x.shape[1]
    state=__import__('torch').load(model_folder/'best.pt',map_location='cpu',weights_only=False)
    if max(state['feature_indices'])>=48:
        with np.load(ROOT/record['path']/'spatial_observations.npz') as z:
            x=np.concatenate((x,z['features']),axis=-1);m=np.concatenate((m,z['mask']),axis=-1)
    predictor=LinkRiskPredictor(model_folder/'best.pt',threads=1)
    with np.load(ROOT/record['path']/'truth.npz') as z:
        xyz=z['positions_start_m'];available=z['link_available'];jammer=z['jammer_active'];connected=z['search_connected']
    # 0초부터 임무 종료까지 전 구간. 마지막 위치는 마지막 속도로 0.5초 보간합니다.
    positions=[np.round(xyz[0,:,:2],1).tolist()];probabilities=[[None]*(n*n)];statuses=[[0]*(n*n)]
    gt=[available[0].astype(int).ravel().tolist()];jams=[int(jammer[0])];connect=[int(connected[0])];rows=[]
    started=time.perf_counter();mismatch=0
    for t in range(len(x)):
        output=predictor.observe((t+1)*.5,x[t],m[t])
        if (t+1)%2:continue
        pp=[None]*(n*n);ss=[0]*(n*n)
        for link in output['links']:
            i,j=link['source'],link['peer'];index=i*n+j
            pp[index]=round(link['risk_5s'],5) if link['risk_5s'] is not None else None
            ss[index]=STATUS[link['status']]
            rows.append({'time_s':(t+1)*.5,**link})
        # 대각선 자기 링크는 결측이므로, 각 드론의 유효한 이웃 행에서 자기 좌표를 읽습니다.
        end=x[t,np.arange(n),(np.arange(n)+1)%n,12:14]
        positions.append(np.round(end,1).tolist());probabilities.append(pp);statuses.append(ss)
        gt.append(available[t].astype(int).ravel().tolist());jams.append(int(jammer[t]));connect.append(int(connected[t]))
    summary={'id':record['id'],'region':record['region'],'duration_s':record['metadata']['duration_s'],
             'rendered_start_s':0,'rendered_end_s':len(positions)-1,'frames':len(positions),
             'complete_mission':record['metadata']['mission']['mission_completed'],
             'stream_inference_elapsed_s':time.perf_counter()-started,'stream_observe_calls':len(x)}
    if summary['rendered_end_s']!=summary['duration_s']:raise RuntimeError('전체 임무 재생 시간 불일치')
    dest=RESULT/'replays';dest.mkdir(exist_ok=True)
    with (dest/(record['id']+'_risk.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'summary':summary,'positions':positions,'risk':probabilities,'status':statuses,
            'truth':gt,'jammer':jams,'connected':connect},summary


def main():
    manifest,p=collection();selected=read_json(RESULT/'selection.json')['selected_variant'];folder=RUNS/(selected+'_s11')
    # 최고 성능 영상 선택 없음: 기존 테헤란, 기존 취약 사례, 다른 평가 지역의 고정 사례.
    cases=[('teheran','east_west','intermittent'),('seoul_20261023_03','north_south','intermittent'),
           ('bucheon_jungdong_east','east_west','continuous')]
    data=[];summaries=[]
    for key,direction,kind in cases:
        r=next(r for r in manifest['records'] if r['region']==key and r['metadata']['direction']==direction and r['metadata']['kind']==kind)
        replay,summary=create_case(r,folder);entry=next(e for e in p['maps'] if e['key']==key);scene=read_json(ROOT/entry['path'])
        replay.update(name=entry['name'],direction=direction,kind=kind,buildings=scene['buildings'],no_fly_zones=scene['no_fly_zones'])
        data.append(replay);summaries.append(summary);print('replay',summary,flush=True)
    payload={'cases':data,'n_drones':8,'n_search':3,'calibration':read_json(folder/'calibration.json'),
             'alarm_policy':read_json(folder/'alarm_policy.json') if (folder/'alarm_policy.json').exists() else None}
    template=(ROOT/'usable/replay_template.html').read_text()
    html=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    dest=ROOT/'docs';dest.mkdir(exist_ok=True);(dest/'통신위험_전체임무_재생.html').write_text(html,encoding='utf-8')
    write_json(RESULT/'replay_summary.json',summaries)


if __name__=='__main__':main()
