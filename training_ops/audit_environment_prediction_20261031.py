"""Offline RF counterfactual diagnostics on saved development trajectories.

No model, map, campaign setting or queued training source is modified. Hidden
RF truth is used only to explain already-recorded events, never as model input.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import multiprocessing as mp
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from adaptive_joint.contracts import read_json,sha,atomic_json
from adaptive_joint.world import World
from cooperative.packets import hop_probabilities

FOLDER=ROOT/'adaptive_joint/artifacts/runs/adaptive_v6_20261030/validation/u0000144/rollouts'
OUT=ROOT/'training_ops/reports/environment_cause_audit_20261031.json'


def case(meta):
    meta=Path(meta);row=read_json(meta);path=meta.with_suffix('.npz')
    if sha(path)!=row['array_sha256']:raise ValueError('Source trajectory hash differs')
    with np.load(path,allow_pickle=False) as z:
        links=z['links'];positions=z['positions']
        aux_end=z['aux_endpoints'];aux_geometry=z['aux_geometry'];aux_support=z['aux_support'];aux_valid=z['aux_valid'];aux_labels=z['aux_labels']
    env=World(**row['spec'],deployment=row['deployment'],observer=None,max_steps=1)
    ns,n=row['search_drones'],row['total_drones']
    assert positions.shape[1:]==(n,2)
    onset=np.zeros_like(links,dtype=bool)
    onset[1:-1]=links[:-2]&~links[1:-1]&~links[2:]
    events=[];end=len(links)*.5-8.5
    for i in range(ns,n):
        for j in range(n):
            if i==j:continue
            for t in np.flatnonzero(onset[:,i,j]):
                if 24<=t*.5<=end:events.append((int(t),i,j))
    ticks=np.array(sorted({t for t,_,_ in events}|{t-6 for t,_,_ in events}|{((t-6)//2)*2 for t,_,_ in events}),int)
    step=ticks//6;alpha=(ticks%6)/6
    xy=positions[step]*(1-alpha[:,None,None])+positions[step+1]*alpha[:,None,None]
    xyz=np.concatenate((xy,np.full((*xy.shape[:-1],1),50.)),axis=-1)
    tx=np.repeat(xyz,n,axis=1).reshape(-1,3)
    rx=np.tile(xyz,(1,n,1)).reshape(-1,3)
    loss,los,_=env.city.radio.propagation(tx,rx,env.scenario)
    loss=loss.reshape(len(ticks),n,n);los=los.reshape(len(ticks),n,n)
    jloss=env.city.radio.propagation(env.jammer[None],xyz.reshape(-1,3),env.scenario)[0].reshape(len(ticks),n)
    index={int(t):i for i,t in enumerate(ticks)}
    reconstructed={};without_jammer={};without_shadow={};geometry_only={}
    def available(l,interference,shadow):
        out=np.empty((n,n));sinr=np.empty((n,n))
        p=hop_probabilities(np.ascontiguousarray(l),np.ascontiguousarray(interference),np.ascontiguousarray(shadow),out,sinr)
        return (p>=.9)&(p.T>=.9)
    for k,t in enumerate(ticks):
        active=env.jam_start<=t*.5<env.jam_end
        interference=10**((env.jam_power-jloss[k])/10)*active
        reconstructed[int(t)]=available(loss[k],interference,env.shadow[t])
        if not np.array_equal(reconstructed[int(t)],links[t]):
            raise AssertionError('RF reconstruction differs from logged labels at '+row['region']+':'+str(t))
        if t in {e[0] for e in events}:
            without_jammer[int(t)]=available(loss[k],np.zeros(n),env.shadow[t])
            without_shadow[int(t)]=available(loss[k],interference,np.zeros((n,n)))
            geometry_only[int(t)]=available(loss[k],np.zeros(n),np.zeros((n,n)))
    counts=dict(events=len(events),recovered_without_jammer=0,recovered_without_shadow=0,
                below_threshold_even_without_jammer_and_shadow=0,recovered_only_if_both_removed=0,
                geometry_loss_increase_ge3db_in_previous3s=0,los_to_nlos_in_previous3s=0,
                within1s_after_hidden_jammer_switch_on=0)
    quality=dict(usable_latest_pre3s_geometry_points=0,los_forecast_correct=0,actual_nlos_points=0,
                 nlos_correctly_forecast=0,first_next_event_matches_this_onset=0)
    errors=[];aux_index={int(t):k for k,t in enumerate(aux_end)}
    horizons=np.array([.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8])
    for t,i,j in events:
        a=without_jammer[t][i,j];b=without_shadow[t][i,j];c=geometry_only[t][i,j]
        counts['recovered_without_jammer']+=int(a)
        counts['recovered_without_shadow']+=int(b)
        counts['below_threshold_even_without_jammer_and_shadow']+=int(not c)
        counts['recovered_only_if_both_removed']+=int(c and not a and not b)
        counts['geometry_loss_increase_ge3db_in_previous3s']+=int(loss[index[t],i,j]-loss[index[t-6],i,j]>=3)
        counts['los_to_nlos_in_previous3s']+=int(los[index[t-6],i,j] and not los[index[t],i,j])
        counts['within1s_after_hidden_jammer_switch_on']+=int(0<=t*.5-env.jam_start<=1 and env.jam_start>0)
        before=((t-6)//2)*2
        ai=aux_index.get(before);r=i-ns
        if ai is not None and aux_support[ai,r,j] and aux_geometry[ai,r,j,-1]>.5:
            h=int(np.flatnonzero(np.isclose(horizons,(t-before)*.5))[0])
            g=aux_geometry[ai,r,j];forecast_los=g[13+h]>.5;actual_los=bool(los[index[t],i,j])
            quality['usable_latest_pre3s_geometry_points']+=1
            quality['los_forecast_correct']+=int(forecast_los==actual_los)
            quality['actual_nlos_points']+=int(not actual_los)
            quality['nlos_correctly_forecast']+=int(not actual_los and not forecast_los)
            quality['first_next_event_matches_this_onset']+=int(aux_valid[ai,r,j] and aux_labels[ai,r,j]==t-before)
            errors.append(float(abs(g[h]-(loss[index[t],i,j]-loss[index[before],i,j]))))
    return dict(region=row['region'],spec=row['spec'],array_sha256=row['array_sha256'],
                rf_reconstruction_exact=True,checked_frames=len(ticks),counts=counts,
                geometry_quality_counts=quality,geometry_delta_absolute_errors_db=errors)


def main():
    metadata=sorted(FOLDER.glob('*.json'));rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        pending=[pool.submit(case,str(p)) for p in metadata]
        for future in as_completed(pending):
            rows.append(future.result())
            print(json.dumps(dict(completed=len(rows),total=len(metadata),region=rows[-1]['region'])),flush=True)
    counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
    assert counts['events']==3033
    result=dict(created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
                scope='Offline diagnostic only, same saved 8 development trajectories from v6 step144; not training or final test',
                counts=counts,fractions={k:v/counts['events'] for k,v in counts.items() if k!='events'},
                cases=sorted(rows,key=lambda r:r['region']),
                interpretation='Counterfactual flags overlap. Removing RF factors holds the recorded positions fixed; not a rerun of policy behavior. Geometry-only weakness is not a mutually exclusive root-cause class.',
                model_inputs_modified=False,source_settings_modified=False)
    quality={k:sum(r['geometry_quality_counts'][k] for r in rows) for k in rows[0]['geometry_quality_counts']}
    errors=np.array([e for r in rows for e in r.pop('geometry_delta_absolute_errors_db')])
    quality.update(los_forecast_agreement=quality['los_forecast_correct']/quality['usable_latest_pre3s_geometry_points'],
                   nlos_event_forecast_recall=quality['nlos_correctly_forecast']/quality['actual_nlos_points'],
                   geometry_delta_absolute_error_median_db=float(np.median(errors)),
                   geometry_delta_absolute_error_p90_db=float(np.percentile(errors,90)))
    result['existing_geometry_forecast_at_latest_integer_3s_or_3p5s_lead']=quality
    atomic_json(OUT,result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
