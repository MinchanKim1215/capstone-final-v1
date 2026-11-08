"""알림 정책 고정 후 사용할 추가 8개 난수 비행. 새로운 지역 맹검과 구분."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from dataclasses import asdict
import argparse,json
import numpy as np
from scipy.special import expit
from swarm_sim.config import Design,Scenario
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from swarm_sim.forecast import atomic_npz
from upgrade.map_setup import digest
from upgrade.train import configure
from upgrade.evaluate import grouped
from usable.collect import prepare_city_without_unused_radio_cache,trajectory,collect_one
from usable.spatial import features
from usable.data import index_episode,load_observation
from usable.evaluate import RESULT,RUNS,load,calibrate,predict_episode
from usable.alarm_policy import arrays_for_record,simulate_policy,summarize


def plan():
    p=read_json(ROOT/'configs/usable_protocol.json');jobs=[]
    for q,key in enumerate(p['splits']['test']):
        direction=['east_west','north_south'][q%2];kind=['clear','continuous','intermittent'][q%3]
        seed=int(digest(['prospective-confirmation8-v1',2026102503,key,direction,kind])[:8],16)%(2**31-2000004)
        jobs.append({'region':key,'direction':direction,'kind':kind,'seed':seed})
    out={'format':'new-random-confirmation8-v1','jobs':jobs,'model_sha256':sha256_file(RUNS/(read_json(RESULT/'selection.json')['selected_variant']+'_s11')/'best.pt'),
         'selection_uses_confirmation':False,'geography':'same known evaluation cities, new shadow/beacon/jammer/packet randomness',
         'timing':'data plan before observing any confirmation labels or metrics; evaluate after alarm policy frozen'}
    out['protocol_sha256']=digest(out);path=ROOT/'configs/usable_confirmation8.json'
    if path.exists() and read_json(path)!=out:raise ValueError('추가 확인 계획 동결 위반')
    write_json(path,out);return out,p


def collect_job(job,protocol,p):
    entry=next(e for e in p['maps'] if e['key']==job['region']);s=Scenario(duration_s=7200,motion_dt_s=3,direction=job['direction']).validate()
    design=Design(**p['design']);city=prepare_city_without_unused_radio_cache(ROOT/entry['path'],s,ROOT/'cache/usable')
    a,meta=trajectory(city,design,s,ROOT/'cache/usable')
    id=f"confirmation_{job['region']}_{job['direction']}_{job['kind']}_s{job['seed']}"
    dest=ROOT/'datasets/confirmation8/episodes'/id;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'episode.json').exists():item=read_json(dest/'episode.json')
    else:item=collect_one(city,a,meta,design,s,job['seed'],job['kind'],dest,protocol,'confirmation')
    if not (dest/'spatial_observations.npz').exists():
        with np.load(dest/'observations.npz') as z:x,m=z['features'],z['mask']
        f,fm=features(city,x,m);atomic_npz(dest/'spatial_observations.npz',features=f,mask=fm,source_sha256=sha256_file(ROOT/'usable/spatial.py'))
    print('confirmation collected',id,flush=True)
    return {'id':id,'region':job['region'],'split':'confirmation','path':dest.relative_to(ROOT).as_posix(),'metadata':item}


def collect_all():
    conf,p=plan();records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(collect_job,j,conf,p) for j in conf['jobs']]):records.append(f.result())
    out={'complete':True,'protocol':conf,'records':sorted(records,key=lambda r:r['id'])}
    write_json(ROOT/'datasets/confirmation8/collection_manifest.json',out)


def evaluate_one(r,folder,policy):
    configure(11,2);model,state,_=load(folder);cal=calibrate(folder);p=read_json(ROOT/'configs/usable_protocol.json')
    ix=index_episode(r,p);x,m=load_observation(r);k,i,j=ix['index'].T
    eligible=m[k,i,j,46]&(x[k,i,j,46]>.5)&m[k,i,j,41]&(x[k,i,j,41]>0)
    union=eligible|ix['current_available'];index=ix['index'][union]
    path=folder/'confirmation_scores';path.mkdir(exist_ok=True);file=path/(r['id']+'.npz')
    if file.exists():
        with np.load(file) as z:prob=z['probability']
    else:
        prob=expit(predict_episode(model,state,r,index)/cal['temperature']).astype(np.float32)
        np.savez_compressed(file,index=index,probability=prob,checkpoint_sha256=sha256_file(folder/'best.pt'))
    op=eligible[union];a=arrays_for_record(r,index[op],prob[op]);chosen=policy['selected']
    new=simulate_policy(*a,chosen['enter_threshold'],chosen['exit_threshold'])[0] if chosen else np.zeros(9,np.int64)
    old=simulate_policy(*a,cal['threshold_conservative'],cal['threshold_conservative'])[0]
    current=ix['current_available'][union]
    print('confirmation evaluated',r['id'],len(index),flush=True)
    return r['region'],ix['y'][union][current],prob[current],new,old,r['metadata']['duration_s']


def evaluate_all():
    conf=read_json(ROOT/'datasets/confirmation8/collection_manifest.json');name=read_json(RESULT/'selection.json')['selected_variant']+'_s11';folder=RUNS/name
    policy=read_json(folder/'alarm_policy.json');cal=calibrate(folder);outputs=[]
    # 평가 전에 가중치와 정책의 해시를 명시적으로 고정합니다.
    frozen={'model_sha256':sha256_file(folder/'best.pt'),'alarm_policy_sha256':sha256_file(folder/'alarm_policy.json'),
            'confirmation_protocol_sha256':conf['protocol']['protocol_sha256'],'before_confirmation_metrics':True}
    write_json(RESULT/'confirmation_frozen_inputs.json',frozen)
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(evaluate_one,r,folder,policy) for r in conf['records']]):outputs.append(f.result())
    y=np.concatenate([x[1] for x in outputs]);prob=np.concatenate([x[2] for x in outputs]);region=np.concatenate([np.full(len(x[1]),x[0]) for x in outputs]);duration=sum(x[5] for x in outputs)
    result={'format':'prospective-random-confirmation-results-v1','frozen_inputs':frozen,'flights':len(outputs),
            'classification':grouped(y,prob,region,cal['threshold_conservative']),
            'new_alarm_policy':summarize(np.sum([x[3] for x in outputs],axis=0),duration),
            'previous_point_warning':summarize(np.sum([x[4] for x in outputs],axis=0),duration),
            'mission_hours':duration/3600,'geography':'same eight known cities; new random realizations; no new-city generalization claim',
            'used_for_fitting':False}
    write_json(RESULT/'confirmation_results.json',result);print(json.dumps(result['new_alarm_policy'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','evaluate']);a=p.parse_args()
    collect_all() if a.stage=='collect' else evaluate_all()
