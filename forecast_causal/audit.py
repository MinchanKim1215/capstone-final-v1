"""377 source flights, exact preserved outcomes, causal inputs and sealed ancestors."""
from pathlib import Path
from collections import Counter
import numpy as np
from . import ROOT,ARTIFACTS,PARENT,NAV,WORKSPACE
from .contracts import read,write,sha,now,folds,BASELINE
from .evaluate import verify_freeze,case_prediction
from .state import from_replay
from .features import summarize,tree_features
from .physics import Geometry
from .wire import SLOTS
from .inputs import load
from .inference import predict
from forecast_recall.episodes import cohort
from forecast_graph.episodes import physical
from forecast_graph.features import LocalHistory,ReceivedTrace
from forecast_rebuild.planning import public_context

def independent(item):
    r=item['source_record'];ep=physical(r);n=ep.values.shape[2];ns=r['n_search'];city,_=public_context(r);city.radio.enable_cache()
    with np.load(item['path']) as a:
        pos=np.unique(np.linspace(0,len(a['ends'])-1,3,dtype=int));ends=a['ends'][pos];cached={k:a[k] for k in ('base','network','physics','network_history')}
    v,m=from_replay(item['replay_path']);history=summarize(v,m,n,ns);np.testing.assert_equal(history.astype(np.float16),cached['network_history']);np.testing.assert_equal(tree_features(history,ends),cached['network'][pos])
    with np.load(item['replay_path']) as a:tx=a['tx'].copy();rx=a['rx']
    tx[...,SLOTS]=0;tx[...,127]=0;geometry=Geometry(city).build(LocalHistory(ep.values,ep.masks,ep.intents,n,ns),ReceivedTrace(tx,rx,n,ns),ends)
    np.testing.assert_equal(geometry,cached['physics'][pos]);errors={}
    if item['role']!='train':
        inputs=load(item)
        for key in ('base','network','physics','ends','n','ns','baseline','embedding','logits'):inputs[key]=inputs[key][pos]
        for entry in read(ARTIFACTS/'freeze.json')['candidates']:
            c=entry['candidate']
            if c['family']!='new':continue
            p,_=case_prediction(c,item,item['role']);actual=predict(c['model'],inputs,device='cpu');delta=float(np.abs(actual-p[pos,0,0]).max());errors[c['tag']]=delta
            if delta>1e-4:raise ValueError('Independent CPU prediction differs: '+str(delta))
    return dict(key=item['key'],role=item['role'],kind=r['spec']['kind'],queries=pos.tolist(),network_history_exact=True,network_tree_exact=True,physics_exact=True,model_errors=errors)

def main():
    f=verify_freeze();preserved={}
    for folder in (PARENT,NAV,WORKSPACE/'forecast_temporal/artifacts',WORKSPACE/'forecast_peer/artifacts'):
        d=read(folder/'delivery_manifest.json')
        for relative,v in d['files'].items():
            if sha(folder.parent/relative)!=v['sha256']:raise ValueError('Sealed predecessor changed: '+relative)
        for path,v in d.get('supplementary_outputs',{}).items():
            if sha(path)!=v['sha256']:raise ValueError('Previous supporting output changed')
        preserved[folder.parent.name]=dict(files=len(d['files']),supporting=len(d.get('supplementary_outputs',{})))
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as OLD
    original=legacy_hashes()
    if original!=read(OLD/'legacy_before.json'):raise ValueError('Original source or weights changed')
    data=read(ARTIFACTS/'data_manifest.json')['records'];fresh=read(ARTIFACTS/'confirmation_data_manifest.json')['records'];parents={i['key']:i for i in read(PARENT/'data_manifest.json')['records']}
    if Counter(i['role'] for i in data)!=dict(train=245,selection=45,calibration=45) or len(fresh)!=42 or set(parents)!={i['key'] for i in data}:raise ValueError('Cohort changed')
    for role in ('selection','calibration','confirmation'):
        if {i['region'] for i in data+fresh if i['role']==role}!=set(folds()[role]):raise ValueError('Spatial split changed')
    used={i['source_record']['spec']['seed'] for i in data}
    for folder in (PARENT,NAV,WORKSPACE/'forecast_graph/artifacts',WORKSPACE/'forecast_temporal/artifacts'):
        used|={i['source_record']['spec']['seed'] for i in read(folder/'confirmation_data_manifest.json')['records']}
    if any(i['source_record']['spec']['seed'] in used for i in fresh):raise ValueError('Confirmation seed reused')
    access=read(ARTIFACTS/'confirmation_access.json')
    if access['opened_at']<f['created_at'] or access['freeze_sha256']!=sha(ARTIFACTS/'freeze.json'):raise ValueError('Fresh confirmation opened before freeze')
    pathfit=read(ARTIFACTS/'path_uncertainty.json');train={i['key']:i for i in data if i['role']=='train'}
    if not pathfit['training_only'] or {r['key'] for r in pathfit['sources']}!=set(train):raise ValueError('Path uncertainty was not TRAIN only')
    for r in pathfit['sources']:
        if r['array_sha256']!=train[r['key']]['source_record']['array_sha256']:raise ValueError('Path-fit source changed')
    tq=uniform=bags=queries=beacons=packets=0
    for count,item in enumerate(data+fresh):
        r=item['source_record'];is_new=item['role']=='confirmation';replay=read(Path(item['replay_path']).with_suffix('.json'));parent=replay['parent_item']
        for path,value in ((item['path'],item['sha256']),(item['parent_path'],item['parent_sha256']),(item['replay_path'],item['replay_sha256']),(r['array_path'],r['array_sha256'])):
            if sha(path)!=value:raise ValueError('Bound source changed: '+str(path))
        for pins in (replay['source_pins'],item['contract']['source_pins']):
            for name,value in pins.items():
                if sha(ROOT/name)!=value:raise ValueError('Input compiler changed: '+name)
        if item['contract']['uncertainty_sha256']!=sha(ARTIFACTS/'path_uncertainty.json'):raise ValueError('Path uncertainty changed')
        with np.load(item['path']) as a,np.load(item['parent_path']) as p:
            for key in ('ends','query','y','up','support','links','uniform','bags'):np.testing.assert_equal(a[key],p[key])
            if a['base'].shape!=(item['queries'],434) or a['network'].shape!=(item['queries'],1116) or a['physics'].shape!=(item['queries'],54):raise ValueError('Input dimension changed')
            for key in ('physics','network_history'):
                if np.isinf(a[key]).any():raise ValueError('Infinite '+key)
            if not np.isfinite(a['baseline']).all():raise ValueError('Nonfinite baseline')
            queries+=len(a['ends'])
            if item['role']=='train':tq+=len(a['ends']);uniform+=int(a['uniform'].sum());bags+=len(a['bags'])
        if not is_new:
            if r!=parents[item['key']]['source_record'] or item['parent_sha256']!=parents[item['key']]['sha256']:raise ValueError('Development parent changed')
            for field,value in parents[item['key']].get('local_cache_sha256',{}).items():
                if sha(Path(r['cache_path'])/(field+'.npy'))!=value:raise ValueError('Original RF local cache changed')
            service=WORKSPACE/'forecast_service/artifacts/service_cache'/(item['key']+'.npz')
        else:
            if read(r['metadata_path'])['request']['checkpoint_sha256']!=sha(BASELINE):raise ValueError('PPO collection policy changed')
            service=ARTIFACTS/'service_cache'/(item['key']+'.npz')
        with np.load(item['replay_path']) as a,np.load(service) as p:
            for key in ('packet_born','packet_status','packet_delay'):np.testing.assert_equal(a[key],p[key])
            packets+=len(a['packet_born']);beacons+=int(a['tx'].shape[0]*a['tx'].shape[1]);n=int(a['n']);ns=int(a['ns'])
            if float(a['delay_decomposition_max_error'])>1e-8 or float(a['predeadline_decomposition_max_error'])>1e-8:raise ValueError('Packet decomposition error')
            if not(replay['RF_and_commands_exact'] and replay['original_128B_packets_restore_exactly'] and replay['packet_outcomes_exact']):raise ValueError('Replay preservation failed')
            if is_new:
                from forecast_service.replay import service_health
                from forecast_rebuild.targets import labels
                exact=service_health(p['packet_born'],p['packet_status'],p['packet_delay'],len(p['service_healthy']))
                for key,value in exact.items():np.testing.assert_equal(p[key],value)
                health=p['service_healthy'];links=np.zeros((len(health),2,2),bool);links[:,1,0]=health
                with np.load(r['array_path']) as raw:
                    ends=raw['aux_endpoints'];ends=ends[ends>=40];gt=labels(links,ends,1);q=np.flatnonzero(gt['valid'][:,0,0]);values=raw['rf_values'];masks=raw['rf_masks'];support=masks[ends[q]-1,0,:,41]&masks[ends[q]-1,0,:,46]&(values[ends[q]-1,0,:,46]>.5);support[:,ns]=False
                    np.testing.assert_equal(np.unpackbits(a['rx'],axis=-1)[...,:n][:,ns:],values[::2,...,33]>0)
                with np.load(item['path']) as z:
                    for key,value in (('ends',ends[q]),('query',q),('links',links),('y',gt['targets'][q,0,0]),('up',gt['currently_up'][q,0,0]),('support',support.any(1))):np.testing.assert_equal(z[key],value)
        write(ARTIFACTS/'audit_status.json',dict(complete=count+1,total=len(data+fresh),updated_at=now()))
    if (tq,uniform,bags)!=(148949,125440,2393):raise ValueError('Training sample budget changed')
    prediction_files=0
    for entry in f['candidates']:
        for scope in ('selection','calibration','confirmation'):
            s=read(ARTIFACTS/'evaluation'/entry['candidate']['tag']/scope/'summary.json')
            for case in s['cases']:
                if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Archived probability changed')
                prediction_files+=1
    checks=[]
    for role in ('train','selection','calibration','confirmation'):
        for kind in ('clear','intermittent','continuous'):checks.append(independent(next(i for i in data+fresh if i['role']==role and i['source_record']['spec']['kind']==kind)))
    for name in ('instrumentation_preflight.json','input_preflight.json','model_preflight.json','collector_preflight.json','stream_preflight.json','runtime_checks.json'):
        if not read(ARTIFACTS/name).get('passed'):raise ValueError('Required check failed: '+name)
    causes=read(ARTIFACTS/'selection_event_causes.json')
    if causes['event_count']!=200 or causes['forecast_outcomes']['risk_below_threshold']!=134:raise ValueError('Initial cause cohort changed')
    verify_freeze();value=dict(passed=True,created_at=now(),preserved_deliveries=preserved,original_dependencies_preserved=len(original),source_flights=len(data+fresh),training_queries=tq,uniform_training_queries=uniform,complete_event_bags=bags,total_queries=queries,
        parent_cohorts_exact=len(data),original_packet_outcomes_checked=packets,original128B_roundtrips=beacons,prediction_files_verified=prediction_files,independent_checks=checks,train_only_path_fit_flights=len(train),same_event_denominator_and_lead_window=True,confirmation_seeds_new=True,
        original_final_test_opened=False,automatic_PPO_replacement=False,hardware_validated=False,production_ready=False,freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'audit.json',value);print(value,flush=True)

if __name__=='__main__':main()
