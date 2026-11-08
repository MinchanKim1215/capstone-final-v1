"""Preserved deliveries, exact cohorts, fresh GT and independent20graph replay."""
from pathlib import Path
from collections import Counter
import numpy as np
from . import ROOT,ARTIFACTS,PARENT,NAV
from .contracts import read,write,sha,now,folds,BASELINE
from .evaluate import verify_freeze,case_prediction
from .features import OFFSETS,build
from .inference import predict
from forecast_recall.episodes import cohort
from forecast_graph.episodes import physical
from forecast_graph.features import LocalHistory,ReceivedTrace
from forecast_navigation.wire import extend_trace
from forecast_navigation.audit import packet_check
from forecast_rebuild.planning import public_context

def independent(item):
    r=item['source_record'];ep=physical(r);n=ep.values.shape[2];ns=r['n_search'];city,_=public_context(r);city.radio.enable_cache()
    with np.load(item['path']) as a:
        positions=np.unique(np.linspace(0,len(a['ends'])-1,3,dtype=int));ends=a['ends'][positions]
        cached={k:a[k] for k in ('node','edge','index','physics','global_features','n','ns','observer')}
    with np.load(item['trace_path']) as a:tx=a['tx'];rx=a['rx']
    with np.load(item['delta_path']) as a:tx=extend_trace(tx,a['delta'])
    actual=build(LocalHistory(ep.values,ep.masks,ep.intents,n,ns),ReceivedTrace(tx,rx,n,ns),ends,city);errors={}
    for key in ('node','edge'):
        wanted=cached[key][cached['index'][positions]];computed=actual[key][actual['index']]
        errors[key]=float(np.abs(wanted.astype(float)-computed.astype(float)).max())
    for key in ('physics','global_features'):errors[key]=float(np.abs(cached[key][positions].astype(float)-actual[key].astype(float)).max())
    if any(errors.values()):raise ValueError('Independent graph inputs differ: '+str(errors))
    actual.update(n=np.full(len(positions),n),ns=np.full(len(positions),ns),observer=np.full(len(positions),ns));model_errors={}
    if item['role']!='train':
        for variant in ('short_bce','short_event','long_event'):
            candidate=next(x['candidate'] for x in read(ARTIFACTS/'freeze.json')['candidates'] if x['candidate']['tag']=='v14_'+variant)
            computed=predict(candidate['model'],actual,device='cpu');p,_=case_prediction(candidate,item,item['role']);delta=float(np.abs(computed-p[positions,0,0]).max());model_errors[variant]=delta
            if delta>1e-4:raise ValueError('Independent CPU risk differs: '+str(delta))
    return dict(key=item['key'],role=item['role'],kind=r['spec']['kind'],queries=positions.tolist(),input_errors=errors,model_errors=model_errors)

def main():
    f=verify_freeze();preserved={}
    for folder in (PARENT,NAV):
        delivery=read(folder/'delivery_manifest.json')
        for relative,value in delivery['files'].items():
            if sha(folder.parent/relative)!=value['sha256']:raise ValueError('Sealed predecessor changed: '+relative)
        for path,value in delivery.get('supplementary_outputs',{}).items():
            if sha(path)!=value['sha256']:raise ValueError('Previous supporting file changed')
        preserved[folder.parent.name]=dict(files=len(delivery['files']),supporting=len(delivery.get('supplementary_outputs',{})))
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as OLD
    original=legacy_hashes()
    if original!=read(OLD/'legacy_before.json'):raise ValueError('Original source or weights changed')
    data=read(ARTIFACTS/'data_manifest.json')['records'];fresh=read(ARTIFACTS/'confirmation_data_manifest.json')['records'];old={i['key']:i for i in read(PARENT/'data_manifest.json')['records']}
    if Counter(i['role'] for i in data)!=dict(train=245,selection=45,calibration=45) or len(fresh)!=42 or set(old)!={i['key'] for i in data}:raise ValueError('Incomplete or changed cohorts')
    for role in ('selection','calibration','confirmation'):
        if {i['region'] for i in data+fresh if i['role']==role}!=set(folds()[role]):raise ValueError('Spatial split changed')
    used=set()
    for folder in (PARENT,NAV,ROOT.parent/'forecast_graph/artifacts'):
        used|={i['source_record']['spec']['seed'] for i in read(folder/'confirmation_data_manifest.json')['records']}
    if any(i['source_record']['spec']['seed'] in used for i in fresh):raise ValueError('Confirmation seed reused')
    access=read(ARTIFACTS/'confirmation_access.json')
    if access['opened_at']<f['created_at'] or access['freeze_sha256']!=sha(ARTIFACTS/'freeze.json'):raise ValueError('Confirmation opened before freeze')
    train_queries=0;uniform=0;bags=0;queries=0;beacons=0;packets=0;parent_cohorts=0
    for count,item in enumerate(data+fresh):
        r=item['source_record'];new=item['role']=='confirmation'
        for path,value in [(item['path'],item['sha256']),(item['parent_path'],item['parent_sha256']),(r['array_path'],r['array_sha256']),(item['trace_path'],item['trace_sha256']),(item['delta_path'],item['delta_sha256'])]:
            if sha(path)!=value:raise ValueError('Data/source changed: '+str(path))
        for name,value in item['source_pins'].items():
            if sha(ROOT/name)!=value:raise ValueError('Feature source changed')
        with np.load(item['path']) as a,np.load(item['parent_path']) as p:
            for key in ('ends','query','y','up','support','links','uniform','bags','logits'):np.testing.assert_equal(a[key],p[key])
            np.testing.assert_equal(a['times'][a['index']],a['ends'][:,None]+OFFSETS)
            if (a['times'][a['index']]>a['ends'][:,None]).any():raise ValueError('Future snapshot input')
            for key in ('node','edge','physics','global_features'):
                if not np.isfinite(a[key]).all():raise ValueError('Nonfinite cached graph')
            queries+=len(a['ends'])
            if item['role']=='train':train_queries+=len(a['ends']);uniform+=int(a['uniform'].sum());bags+=len(a['bags'])
        if not new:
            if item['source_record']!=old[item['key']]['source_record'] or item['parent_sha256']!=old[item['key']]['sha256']:raise ValueError('Parent source/cohort changed')
            parent_cohorts+=1
        else:
            for path,value in [(item['legacy_path'],item['legacy_sha256']),(item['reference_path'],item['reference_sha256']),(item['navigation_reference_path'],item['navigation_reference_sha256'])]:
                if sha(path)!=value:raise ValueError('Fresh baseline changed')
            meta=read(r['metadata_path'])
            if meta['request']['checkpoint_sha256']!=sha(BASELINE):raise ValueError('PPO collection policy changed')
            with np.load(item['trace_path']) as a:tx=a['tx'];rx=a['rx'];n=int(a['n']);ns=int(a['ns'])
            with np.load(item['delta_path']) as a:beacons+=packet_check(tx,a['delta'])
            with np.load(r['array_path']) as a:
                values=a['rf_values'];masks=a['rf_masks'];ends=a['aux_endpoints'];ends=ends[ends>=40]
                np.testing.assert_equal(np.unpackbits(rx,axis=-1)[...,:n][:,ns:],values[::2,...,33]>0)
            from forecast_service.replay import service_health
            with np.load(ARTIFACTS/'service_cache'/(item['key']+'.npz')) as a:
                packets+=len(a['packet_born']);exact=service_health(a['packet_born'],a['packet_status'],a['packet_delay'],len(a['service_healthy']))
                for key,value in exact.items():np.testing.assert_equal(a[key],value)
                healthy=a['service_healthy']
            from forecast_rebuild.targets import labels
            links=np.zeros((len(healthy),2,2),bool);links[:,1,0]=healthy;gt=labels(links,ends,1);q=np.flatnonzero(gt['valid'][:,0,0]);support=masks[ends[q]-1,0,:,41]&masks[ends[q]-1,0,:,46]&(values[ends[q]-1,0,:,46]>.5);support[:,ns]=False
            with np.load(item['path']) as a:
                for key,value in [('ends',ends[q]),('query',q),('links',links),('y',gt['targets'][q,0,0]),('up',gt['currently_up'][q,0,0]),('support',support.any(1))]:np.testing.assert_equal(a[key],value)
        write(ARTIFACTS/'audit_status.json',dict(complete=count+1,total=len(data+fresh),updated_at=now()))
    if (train_queries,uniform,bags)!=(148949,125440,2393):raise ValueError('Training sample budget changed')
    prediction_files=0
    for entry in f['candidates']:
        for scope in ('selection','calibration','confirmation'):
            summary=read(ARTIFACTS/'evaluation'/entry['candidate']['tag']/scope/'summary.json')
            for case in summary['cases']:
                if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Archived probability changed')
                prediction_files+=1
    checks=[]
    for role in ('train','selection','calibration','confirmation'):
        for kind in ('clear','intermittent','continuous'):checks.append(independent(next(i for i in data+fresh if i['role']==role and i['source_record']['spec']['kind']==kind)))
    for name in ('preflight.json','collector_preflight.json','stream_preflight.json','runtime_checks.json'):
        a=read(ARTIFACTS/name)
        if not(a.get('passed') or a.get('state')=='passed'):raise ValueError('Check not passed: '+name)
    value=dict(passed=True,created_at=now(),preserved_deliveries=preserved,original_dependencies_preserved=len(original),source_flights=len(data+fresh),training_queries=train_queries,
        uniform_training_queries=uniform,complete_event_bags=bags,total_queries=queries,parent_cohorts_exact=parent_cohorts,fresh_beacon_roundtrips=beacons,fresh_settled_packets=packets,
        prediction_files_verified=prediction_files,independent_checks=checks,same_event_denominator_and_lead_window=True,confirmation_seeds_new=True,original_final_test_opened=False,
        automatic_PPO_replacement=False,hardware_validated=False,production_ready=False,freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'audit.json',value);print(value,flush=True)

if __name__=='__main__':main()
