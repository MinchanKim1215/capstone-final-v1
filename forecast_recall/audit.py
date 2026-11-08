"""Preservation, complete pre-event bags, original targets and independent risk proof."""
from collections import Counter
from pathlib import Path
import numpy as np
from . import ROOT,ARTIFACTS,PARENT
from .contracts import read,write,sha,now,folds,BASELINE
from .evaluate import verify_freeze,case_prediction
from .data import query_plan
from .features import steps
from .embedding import extract
from .inference import predict
from .episodes import cohort
from forecast_graph.contracts import V10
from forecast_graph.episodes import physical
from forecast_graph.features import LocalHistory,ReceivedTrace,build
from forecast_navigation.wire import extend_trace
from forecast_navigation.audit import packet_check
from forecast_rebuild.planning import public_context


def independent(item):
    r=item['source_record'];ep=physical(r);n=ep.values.shape[2];ns=r['n_search'];fresh=item['role']=='confirmation'
    trace=(ARTIFACTS/'fresh_traces' if fresh else ROOT.parent/'forecast_graph/artifacts/traces')/(item['key']+'.npz')
    delta=(ARTIFACTS if fresh else PARENT)/'deltas'/(item['key']+'.npz')
    with np.load(item['path'],allow_pickle=False) as a:
        positions=np.unique(np.linspace(0,len(a['ends'])-1,3,dtype=int));ends=a['ends'][positions]
        expected={k:a[k][positions] for k in ('embedding','logits')};inputs={k:a[k] for k in ('embedding','logits','history','ends')}
    with np.load(trace,allow_pickle=False) as a:tx=a['tx'];rx=a['rx']
    with np.load(delta,allow_pickle=False) as a:tx=extend_trace(tx,a['delta'])
    city,_=public_context(r);city.radio.enable_cache();local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns)
    f=build(local,ReceivedTrace(tx,rx,n,ns),np.c_[ends,np.zeros(len(ends),int)],city);f.update(n=np.full(len(ends),n),ns=np.full(len(ends),ns),observer=np.full(len(ends),ns))
    actual=extract(f,device='cpu');errors={k:float(np.abs(actual[k]-expected[k]).max()) for k in actual}
    history=steps(ep.values[:,0],ep.masks[:,0],ep.intents[:,0],ep.energy[:,0],n,ns)
    errors['history']=float(np.abs(history.astype(float)-inputs['history'].astype(float)).max())
    model_errors={}
    if item['role']!='train':
        small=dict(inputs,embedding=inputs['embedding'][positions],logits=inputs['logits'][positions],ends=inputs['ends'][positions])
        for variant in ('bce','event','memory_event'):
            candidate=next(i['candidate'] for i in read(ARTIFACTS/'freeze.json')['candidates'] if i['candidate']['tag']=='v13_'+variant)
            computed=predict(candidate['model'],small,device='cpu');cached,_=case_prediction(candidate,item,item['role'])
            model_errors[variant]=float(np.abs(computed-cached[positions,0,0]).max())
    if errors['history']!=0 or any(v>1e-4 for v in errors.values()) or any(v>1e-4 for v in model_errors.values()):raise ValueError('Independent inference/input mismatch')
    return dict(key=item['key'],role=item['role'],kind=r['spec']['kind'],queries=positions.tolist(),frozen_context_errors=errors,adapter_errors=model_errors)


def main():
    frozen=verify_freeze();parent=read(PARENT/'delivery_manifest.json')
    for relative,value in parent['files'].items():
        if sha(PARENT.parent/relative)!=value['sha256']:raise ValueError('Previous delivery changed: '+relative)
    for p,value in parent.get('supplementary_outputs',{}).items():
        if sha(p)!=value['sha256']:raise ValueError('Previous PDF or supporting output changed')
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as OLD
    original=legacy_hashes()
    if original!=read(OLD/'legacy_before.json'):raise ValueError('Original sources/weights changed')
    data=read(ARTIFACTS/'data_manifest.json')['records'];fresh=read(ARTIFACTS/'confirmation_data_manifest.json')['records']
    if Counter(i['role'] for i in data)!=dict(train=245,selection=45,calibration=45) or len(fresh)!=42:raise ValueError('Study cohorts incomplete')
    previous={i['key']:i for i in read(PARENT/'data_manifest.json')['records']}
    if set(previous)!={i['key'] for i in data}:raise ValueError('Development sources changed')
    for i in data:
        if i['source_record']!=previous[i['key']]['source_record']:raise ValueError('Development source changed')
    for role in ('selection','calibration','confirmation'):
        if {i['region'] for i in data+fresh if i['role']==role}!=set(folds()[role]):raise ValueError('Spatial cohort changed')
    old_seeds={i['source_record']['spec']['seed'] for i in read(PARENT/'confirmation_data_manifest.json')['records']}
    from forecast_graph import ARTIFACTS as GRAPH
    old_seeds |= {i['source_record']['spec']['seed'] for i in read(GRAPH/'confirmation_data_manifest.json')['records']}
    if any(i['source_record']['spec']['seed'] in old_seeds for i in fresh):raise ValueError('Previous confirmation seed reused')
    access=read(ARTIFACTS/'confirmation_access.json')
    if access['opened_at']<frozen['created_at'] or access['freeze_sha256']!=sha(ARTIFACTS/'freeze.json'):raise ValueError('Test opened before freeze')
    query_count=0;natural=0;bags=0;packet_count=0;beacons=0;parent_cohorts=0
    for index,item in enumerate(data+fresh):
        r=item['source_record'];fresh_case=r['role']=='confirmation';meta=read(Path(item['path']).with_suffix('.json'))
        if sha(item['path'])!=item['sha256'] or sha(r['array_path'])!=r['array_sha256']:raise ValueError('Data/source checksum mismatch')
        for name,value in meta['source_pins'].items():
            if sha(ROOT/name)!=value:raise ValueError('Input generation source changed')
        for name,value in meta.get('local_cache_sha256',{}).items():
            if sha(Path(r['cache_path'])/(name+'.npy'))!=value:raise ValueError('Original local RF cache changed')
        service=(ARTIFACTS if fresh_case else V10)/'service_cache'/(item['key']+'.npz')
        if sha(service)!=meta['service_sha256']:raise ValueError('Ground truth binding changed')
        with np.load(item['path'],allow_pickle=False) as a:
            query_count+=len(a['ends']);natural+=int(a['uniform'].sum()) if r['role']=='train' else 0;bags+=len(a['bags'])
            if len(np.unique(a['query']))!=len(a['query']):raise ValueError('Duplicate time query')
            for key in ('embedding','logits','history'):
                if not np.isfinite(a[key]).all():raise ValueError('Nonfinite model input')
            if r['role']=='train':
                ep=physical(r)
                with np.load(service,allow_pickle=False) as s:health=s['service_healthy']
                q,u,b,event,links,gt,support=query_plan(ep,health,item['key'])
                for key,value in [('query',q),('uniform',u),('bags',b),('event_times',event),('links',links),('y',gt['targets'][q,0,0]),('up',gt['currently_up'][q,0,0]),('support',support[q])]:np.testing.assert_equal(a[key],value)
            elif not fresh_case:
                from forecast_navigation.episodes import cohort as parent_cohort
                old=parent_cohort(previous[item['key']],'service');new=cohort(item)
                for name in ('ends','links','targets','valid','currently_up','legacy_support'):np.testing.assert_equal(getattr(new,name),getattr(old,name))
                parent_cohorts+=1
        if fresh_case:
            if sha(item['reference_path'])!=item['reference_sha256']:raise ValueError('Fresh predecessor risk changed')
            raw_meta=read(r['metadata_path'])
            if raw_meta['request']['checkpoint_sha256']!=sha(BASELINE):raise ValueError('Data collection policy changed')
            trace=ARTIFACTS/'fresh_traces'/(item['key']+'.npz');delta=ARTIFACTS/'deltas'/(item['key']+'.npz')
            for p in (trace,delta):
                if sha(p)!=read(p.with_suffix('.json'))['sha256']:raise ValueError('Fresh beacon trace changed')
            with np.load(trace,allow_pickle=False) as a:tx=a['tx'];rx=a['rx'];ns=int(a['ns']);n=int(a['n'])
            with np.load(delta,allow_pickle=False) as a:beacons+=packet_check(tx,a['delta'])
            with np.load(r['array_path'],allow_pickle=False) as a:
                values=a['rf_values'];masks=a['rf_masks'];ends=a['aux_endpoints'];ends=ends[ends>=40]
                np.testing.assert_equal(np.unpackbits(rx,axis=-1)[...,:n][:,ns:],values[::2,...,33]>0)
            from forecast_service.replay import service_health
            with np.load(service,allow_pickle=False) as a:
                packet_count+=len(a['packet_born']);exact=service_health(a['packet_born'],a['packet_status'],a['packet_delay'],len(a['service_healthy']))
                for k,v in exact.items():np.testing.assert_equal(a[k],v)
                healthy=a['service_healthy']
            from forecast_rebuild.targets import labels
            service_links=np.zeros((len(healthy),2,2),bool);service_links[:,1,0]=healthy;gt=labels(service_links,ends,1);q=np.flatnonzero(gt['valid'][:,0,0])
            support=masks[ends[q]-1,0,:,41]&masks[ends[q]-1,0,:,46]&(values[ends[q]-1,0,:,46]>.5);support[:,ns]=False
            with np.load(item['path'],allow_pickle=False) as a:
                for key,value in [('ends',ends[q]),('query',q),('links',service_links),('y',gt['targets'][q,0,0]),('up',gt['currently_up'][q,0,0]),('support',support.any(1))]:np.testing.assert_equal(a[key],value)
        write(ARTIFACTS/'audit_status.json',dict(complete=index+1,total=len(data+fresh),updated_at=now()))
    if natural!=125440 or bags!=2393:raise ValueError('Sampling budget or complete events changed')
    prediction_files=0
    for item in frozen['candidates']:
        for scope in ('selection','calibration','confirmation'):
            s=read(ARTIFACTS/'evaluation'/item['candidate']['tag']/scope/'summary.json')
            for c in s['cases']:
                if sha(c['predictions_path'])!=c['prediction_sha256']:raise ValueError('Prediction archive changed')
                prediction_files+=1
    independent_checks=[]
    for role in ('train','selection','calibration','confirmation'):
        for kind in ('clear','continuous','intermittent'):
            item=next(i for i in data+fresh if i['role']==role and i['source_record']['spec']['kind']==kind);independent_checks.append(independent(item))
    for name in ('preflight.json','stream_preflight.json','collector_preflight.json','runtime_checks.json'):
        if not read(ARTIFACTS/name)['passed']:raise ValueError('Required check failed: '+name)
    value=dict(passed=True,created_at=now(),previous_delivery_files_preserved=len(parent['files']),previous_supporting_outputs_preserved=len(parent.get('supplementary_outputs',{})),
        original_dependencies_preserved=len(original),source_flights=len(data+fresh),training_queries=sum(i['queries'] for i in data if i['role']=='train'),
        natural_training_queries=natural,complete_training_event_bags=bags,total_queries=query_count,parent_cohort_identity_checks=parent_cohorts,
        fresh_beacon_roundtrips=beacons,fresh_settled_packets=packet_count,prediction_files_verified=prediction_files,independent_checks=independent_checks,
        same_GT_qualification_lead_hysteresis_and_event_denominators=True,new_confirmation_seed_reuse=False,
        original_final_test_opened=False,automatic_PPO_replacement=False,hardware_validated=False,production_ready=False,freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'audit.json',value);print(value,flush=True)


if __name__=='__main__':main()
