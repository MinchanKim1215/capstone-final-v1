"""Read-only predecessor preservation, exact packet views and prediction audit."""
from collections import Counter
from pathlib import Path
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,PARENT,folds
from .evaluate import verify_freeze,case_prediction,records
from .data import arrays,INPUT_KEYS
from .wire import extend_trace
from .episodes import cohort
from .inference import predict


def packet_check(base,delta):
    revised=extend_trace(base,delta);shape=base.shape[:2]
    old=base[...,30:66].copy().view('<i2').reshape(*shape,9,2)
    path=revised[...,30:66].copy().view('<i2').reshape(*shape,9,2).astype(float)/4
    tail=revised[...,114:126].copy().view('<i2').reshape(*shape,3,2).astype(float)/4
    stamp=base[...,4:8].copy().view('<u4').reshape(shape)
    issued=base[...,8:12].copy().view('<u4').reshape(shape)
    elapsed=(stamp-issued)/1000
    np.testing.assert_equal(elapsed,np.broadcast_to(elapsed[:,:1],elapsed.shape))
    if not np.isin(elapsed,[0,1,2]).all():raise ValueError('Unsupported packet issuing clock')
    reconstructed=path.copy()
    for phase in range(3):
        select=elapsed[:,0]==phase;endpoint=path[select,:,3-phase]
        knots=np.concatenate([endpoint[:,:,None],tail[select]],axis=2)
        for h in range(9):
            t=phase+h
            if t<=3:continue
            left=min((t-3)//3,2);fraction=(t-3-left*3)/3
            reconstructed[select,:,h]=knots[:,:,left]*(1-fraction)+knots[:,:,left+1]*fraction
    np.testing.assert_equal(np.rint(reconstructed*4).astype('<i2'),old)
    for a,b in [(revised[...,:30],base[...,:30]),(revised[...,66:114],base[...,66:114])]:np.testing.assert_equal(a,b)
    return int(np.prod(shape))


def main():
    frozen=verify_freeze()
    from forecast_graph.evaluate import verify_freeze as parent_verify
    parent_verify();pa=read(PARENT/'audit.json')
    if not pa['passed']:raise ValueError('Parent audit failed')
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as OLD
    original=legacy_hashes()
    if original!=read(OLD/'legacy_before.json'):raise ValueError('Original code or weights changed')
    parent_delivery=PARENT/'delivery_manifest.json';preserved=0
    if parent_delivery.exists():
        for relative,item in read(parent_delivery)['files'].items():
            if sha(PARENT.parent/relative)!=item['sha256']:raise ValueError('Parent delivery changed: '+relative)
            preserved+=1
    development=read(ARTIFACTS/'data_manifest.json')['records'];fresh=read(ARTIFACTS/'confirmation_data_manifest.json')['records']
    previous_data={i['key']:i for i in read(PARENT/'data_manifest.json')['records']}
    if {i['key'] for i in development}!=set(previous_data):raise ValueError('Development cases changed')
    for item in development:
        old=previous_data[item['key']]
        if item['base_path']!=old['path'] or item['base_sha256']!=old['sha256'] or item['source_record']!=old['source_record'] or item['queries']!=old['queries']:
            raise ValueError('Original queries or target file changed')
    if Counter(i['role'] for i in development)!=dict(train=245,selection=45,calibration=45) or len(fresh)!=42:raise ValueError('Incomplete cohort')
    for role in ('selection','calibration','confirmation'):
        if {i['region'] for i in development+fresh if i['role']==role}!=set(folds()[role]):raise ValueError('Spatial role changed')
    old_confirm={i['source_record']['spec']['seed'] for i in read(PARENT/'confirmation_data_manifest.json')['records']}
    if any(i['source_record']['spec']['seed'] in old_confirm for i in fresh):raise ValueError('Previous confirmation seed reused')
    access=read(ARTIFACTS/'confirmation_access.json')
    if access['freeze_sha256']!=sha(ARTIFACTS/'freeze.json') or access['opened_at']<frozen['created_at']:raise ValueError('Test accessed before freeze')
    messages=0;queries=0;packets=0;training_queries=0
    for item in development+fresh:
        r=item['source_record'];key=r['key'];new=r['role']=='confirmation';trace=(ARTIFACTS/'fresh_traces' if new else PARENT/'traces')/(key+'.npz')
        if sha(r['array_path'])!=r['array_sha256']:raise ValueError('Source trajectory changed')
        for path,expected in [(item['base_path'],item['base_sha256']),(item['mission_path'],item['mission_sha256']),
                              (ARTIFACTS/'deltas'/(key+'.npz'),read(ARTIFACTS/'deltas'/(key+'.json'))['sha256'])]:
            if sha(path)!=expected:raise ValueError('Data binding changed: '+str(path))
        tm=read(trace.with_suffix('.json'))
        if sha(trace)!=tm['sha256']:raise ValueError('RF trace changed')
        with np.load(trace,allow_pickle=False) as a:base=a['tx'];rx=a['rx'];n=int(a['n']);ns=int(a['ns'])
        with np.load(ARTIFACTS/'deltas'/(key+'.npz'),allow_pickle=False) as a:messages+=packet_check(base,a['delta'])
        dm=read(ARTIFACTS/'deltas'/(key+'.json'))
        if not new:
            for flag in ('original_packets_reconstruct_exactly','RF_and_RX_exact','trajectories_and_packets_exact'):
                if not dm[flag]:raise ValueError('Counterfactual preview changed the flight')
            for name,value in dm['contract']['source_pins'].items():
                if sha(ROOT/name)!=value:raise ValueError('Replay implementation changed')
            if dm['contract']['base_trace_sha256']!=sha(trace):raise ValueError('Replay base changed')
        with np.load(item['base_path'],allow_pickle=False) as a:
            q=a['query'];queries+=len(q)
            if len(np.unique(q,axis=0))!=len(q):raise ValueError('Duplicate query')
            if r['role']=='train':training_queries+=len(q)
            elif q[:,1].any():raise ValueError('Observer cohort changed')
            for k in INPUT_KEYS:
                if not np.isfinite(a[k]).all():raise ValueError('Nonfinite old inputs')
        with np.load(item['mission_path'],allow_pickle=False) as a:
            for k in a.files:
                if not np.isfinite(a[k]).all():raise ValueError('Nonfinite revised inputs')
        if item['cohort_path'] and sha(item['cohort_path'])!=item['cohort_sha256']:raise ValueError('Target cohort changed')
        if new:
            if sha(item['reference_path'])!=item['reference_sha256']:raise ValueError('Parent reference changed')
            with np.load(r['array_path'],allow_pickle=False) as a:observed=a['rf_values'][::2,...,33]>0
            np.testing.assert_equal(np.unpackbits(rx,axis=-1)[...,:n][:,ns:],observed)
            sp=ARTIFACTS/'service_cache'/(key+'.npz')
            if sha(sp)!=read(sp.with_suffix('.json'))['sha256']:raise ValueError('Packet labels changed')
            from forecast_service.replay import service_health
            with np.load(sp,allow_pickle=False) as a:
                exact=service_health(a['packet_born'],a['packet_status'],a['packet_delay'],len(a['service_healthy']));packets+=len(a['packet_born'])
                for k,v in exact.items():np.testing.assert_equal(v,a[k])
    if training_queries!=94080:raise ValueError('Training budget changed')
    #Independently infer frozen models on three query rows in every condition
    #and fold. These do not read or write prediction caches.
    inspected=[];max_error=0.;prediction_files=0
    all_items=development+fresh
    for scope in ('selection','calibration','confirmation'):
        for kind in ('clear','continuous','intermittent'):
            item=next(i for i in all_items if i['role']==scope and i['source_record']['spec']['kind']==kind)
            pos=np.unique(np.linspace(0,item['queries']-1,3,dtype=int));checked={}
            for profile in ('original','mission'):
                model=str(ARTIFACTS/'models'/('v12_'+profile)/'model.json');pred=predict(model,arrays(item,profile,pos),device='cuda')
                for task in ('link','service'):
                    candidate=next(i['candidate'] for i in frozen['candidates'] if i['candidate']['task']==task and i['candidate']['tag']=='v12_'+profile)
                    cached,_=case_prediction(candidate,item,scope)
                    wanted=cached[pos,0] if task=='link' else cached[pos,0,0]
                    actual=pred[task][:,:wanted.shape[1]] if task=='link' else pred[task]
                    delta=float(np.max(np.abs(actual-wanted)));max_error=max(max_error,delta);checked[profile+'_'+task]=delta
            inspected.append(dict(scope=scope,kind=kind,key=item['key'],query_rows=pos.tolist(),maximum_errors=checked))
    for outcome in read(ARTIFACTS/'completed_comparison.json')['outcomes']:
        for scope in ('selection','calibration','confirmation'):
            path=ARTIFACTS/'evaluation'/outcome['candidate']['task']/outcome['candidate']['tag']/scope/'summary.json'
            summary=read(path)
            for case in summary['cases']:
                if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Prediction file changed')
                prediction_files+=1
    if max_error>1e-4:raise ValueError('Independent inference differs')
    for name in ('stream_preflight.json','runtime_checks.json','collector_preflight.json'):
        if not read(ARTIFACTS/name)['passed']:raise ValueError('Required preflight failed: '+name)
    value=dict(passed=True,created_at=now(),original_dependencies_preserved=len(original),parent_delivery_files_preserved=preserved,
        parent_audit_sha256=sha(PARENT/'audit.json'),source_flights=len(all_items),training_queries=training_queries,total_queries=queries,
        beacon_messages_roundtrip_checked=messages,payload_bytes=128,extra_airtime_s=0.,fresh_settled_packets_verified=packets,
        prediction_files_verified=prediction_files,independent_prediction_cases=inspected,maximum_independent_probability_error=max_error,
        same_queries_and_targets_as_V11=True,confirmation_seed_reuse=False,original_final_test_opened=False,
        automatic_PPO_replacement=False,hardware_validated=False,production_ready=False,freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'audit.json',value);print(value,flush=True)


if __name__=='__main__':main()
