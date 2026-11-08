"""Observed input replay, immutable ancestry and prospectively blind cohort proof."""
from pathlib import Path
from collections import Counter
import numpy as np
from . import ROOT,ARTIFACTS,PARENT,GRAPH
from .contracts import read,write,sha,now,folds
from .evaluate import verify_freeze,case_prediction
from .features import steps
from .data import inputs as load_inputs
from .inference import predict
from forecast_graph.episodes import physical

def main():
    f=verify_freeze();protocol=read(ARTIFACTS/'protocol.json');parent=read(PARENT/'delivery_manifest.json')
    for name,value in parent['files'].items():
        if sha(PARENT.parent/name)!=value['sha256']:raise ValueError('V13 sealed delivery changed')
    for path,value in parent.get('supplementary_outputs',{}).items():
        if sha(path)!=value['sha256']:raise ValueError('V13 supporting file changed')
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as OLD
    original=legacy_hashes()
    if original!=read(OLD/'legacy_before.json'):raise ValueError('Original code/weights changed')
    v14=read(GRAPH/'audit.json')
    if not v14['passed']:raise ValueError('Shared cohort original GT audit incomplete')
    test_summaries=[read(p)['created_at'] for p in (GRAPH/'evaluation').glob('*/confirmation/summary.json')]
    if not test_summaries or protocol['created_at']>=min(test_summaries):raise ValueError('Shared test was not prospectively registered')
    access=read(ARTIFACTS/'confirmation_access.json')
    if access['opened_at']<f['created_at'] or access['freeze_sha256']!=sha(ARTIFACTS/'freeze.json'):raise ValueError('Own confirmation opened before source/model freeze')
    data=read(ARTIFACTS/'data_manifest.json')['records'];fresh=read(ARTIFACTS/'confirmation_data_manifest.json')['records'];sources=read(GRAPH/'confirmation_data_manifest.json')['records'];previous={i['key']:i for i in read(PARENT/'data_manifest.json')['records']}
    if Counter(i['role'] for i in data)!=dict(train=245,selection=45,calibration=45) or len(fresh)!=42 or {i['key'] for i in fresh}!={i['key'] for i in sources}:raise ValueError('Incomplete or different cohorts')
    if any(i['source_record']!=previous[i['key']]['source_record'] for i in data):raise ValueError('Training or development source changed')
    for role in ('selection','calibration','confirmation'):
        if {i['region'] for i in data+fresh if i['role']==role}!=set(folds()[role]):raise ValueError('Spatial split changed')
    count=natural=bags=0
    for index,item in enumerate(data+fresh):
        r=item['source_record'];new=item['role']=='confirmation'
        for path,value in [(item['path'],item['sha256']),(item['parent_path'],item['parent_sha256']),(r['array_path'],r['array_sha256'])]:
            if sha(path)!=value:raise ValueError('Data/source hash mismatch')
        for name,value in item['source_pins'].items():
            if sha(ROOT/name)!=value:raise ValueError('Peer representation source changed')
        if not new:
            for name,value in previous[item['key']].get('local_cache_sha256',{}).items():
                if sha(Path(r['cache_path'])/(name+'.npy'))!=value:raise ValueError('Original raw observation cache changed')
        else:
            if sha(item['reference_path'])!=item['reference_sha256'] or sha(item['v14_data_path'])!=item['v14_data_sha256']:raise ValueError('Shared baseline/cohort changed')
        ep=physical(r);n=ep.values.shape[2];ns=r['n_search'];observed=steps(ep.values[1::2,0],ep.masks[1::2,0],ep.intents[1::2,0],ep.energy[1::2,0],n,ns)
        with np.load(item['path']) as a,np.load(item['parent_path']) as p:
            np.testing.assert_equal(a['peer_history'],observed)
            for key in ('ends','query','y','up','support','links','uniform','bags'):np.testing.assert_equal(a[key],p[key])
            count+=len(a['ends'])
            if item['role']=='train':natural+=int(a['uniform'].sum());bags+=len(a['bags'])
        write(ARTIFACTS/'audit_status.json',dict(complete=index+1,total=len(data+fresh),updated_at=now()))
    if (natural,bags)!=(125440,2393):raise ValueError('Sampling changed')
    prediction_files=0
    for entry in f['candidates']:
        for scope in ('selection','calibration','confirmation'):
            summary=read(ARTIFACTS/'evaluation'/entry['candidate']['tag']/scope/'summary.json')
            for case in summary['cases']:
                if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Archived risk changed')
                prediction_files+=1
    checks=[]
    for role in ('selection','calibration','confirmation'):
        for kind in ('clear','intermittent','continuous'):
            item=next(i for i in data+fresh if i['role']==role and i['source_record']['spec']['kind']==kind);a=load_inputs(item);ix=np.unique(np.linspace(0,len(a['ends'])-1,3,dtype=int));small=dict(a,embedding=a['embedding'][ix],logits=a['logits'][ix],ends=a['ends'][ix],n=a['n'][ix],ns=a['ns'][ix]);errors={}
            for variant in ('global_event','peer_bce','peer_event'):
                candidate=next(i['candidate'] for i in f['candidates'] if i['candidate']['tag']=='v15_'+variant);p,_=case_prediction(candidate,item,role);computed=predict(candidate['model'],small,device='cpu');errors[variant]=float(np.abs(computed-p[ix,0,0]).max())
            if any(v>1e-4 for v in errors.values()):raise ValueError('Independent CPU risk differs')
            checks.append(dict(role=role,kind=kind,key=item['key'],queries=ix.tolist(),errors=errors))
    for name in ('preflight.json','stream_preflight.json','runtime_checks.json'):
        if not read(ARTIFACTS/name)['passed']:raise ValueError('Required check incomplete')
    value=dict(passed=True,created_at=now(),source_flights=len(data+fresh),training_queries=sum(i['queries'] for i in data if i['role']=='train'),uniform_training_queries=natural,complete_event_bags=bags,total_queries=count,
        original_dependencies_preserved=len(original),previous_delivery_files_preserved=len(parent['files']),V14_GT_audit_sha256=sha(GRAPH/'audit.json'),same_blind42_cohort=True,
        registration_at=protocol['created_at'],first_V14confirmation_summary_at=min(test_summaries),registered_before_any_V14predictive_outcome=True,raw_generation_before_V15freeze=True,
        prediction_files_verified=prediction_files,independent_checks=checks,full_raw_peer_history_replay_exact=True,same_GT_qualification_lead_hysteresis_event_denominator=True,
        original_final_test_opened=False,production_ready=False,hardware_validated=False,automatic_PPO_replacement=False,freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'audit.json',value);print(value,flush=True)

if __name__=='__main__':main()
