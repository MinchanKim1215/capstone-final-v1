"""Artifact integrity, real reception, issued-motion timing and held-out checks."""
from collections import Counter
from pathlib import Path
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,now,verify,V10,role,folds
from .evaluate import verify_freeze
from .codec import MAX_N
from forecast_rebuild.contracts import maps,verify_prior
from forecast_rebuild.dataset import Episode


def main():
    verify_freeze();prior=read(V10/'delivery_manifest.json')
    for relative,item in prior['files'].items():
        if sha(V10.parent/relative)!=item['sha256']:raise ValueError('Prior V10 delivery changed: '+relative)
    from forecast_service.evaluate import verify_freeze as verify_v10
    verify_v10();original=verify_prior()
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as OLD
    preserved=legacy_hashes()
    if preserved!=read(OLD/'legacy_before.json'):raise ValueError('Original sources,176checkpoint or reference weights changed')
    data=read(ARTIFACTS/'data_manifest.json')['records'];fresh=read(ARTIFACTS/'confirmation_data_manifest.json')['records']
    counts=Counter(i['role'] for i in data)
    if counts!=dict(train=245,selection=45,calibration=45) or len(fresh)!=42:raise ValueError('Incomplete study cohorts')
    regions={k:{i['region'] for i in data+fresh if i['role']==k} for k in ('train','selection','calibration','confirmation')}
    registry=maps();groups={k:{registry[r]['joint_spatial_group'] for r in v} for k,v in regions.items()}
    if any(groups['train']&groups[k] for k in ('selection','calibration','confirmation')):raise ValueError('Training geography overlap')
    for k in ('selection','calibration','confirmation'):
        if regions[k]!=set(folds()[k]):raise ValueError('Development geography changed')
    all_records=data+fresh;queries=0;messages=0;committed_points=0;max_error=0.;reception_checks=0
    errors_by_h=[[] for _ in range(9)];boundaries=0;settled=0
    for item in all_records:
        r=item['source_record']
        if role(r['region'])!=r['role']:raise ValueError('Record has wrong spatial role')
        if sha(item['path'])!=item['sha256'] or sha(r['array_path'])!=r['array_sha256']:raise ValueError('Input archive changed')
        trace=ARTIFACTS/'traces'/(r['key']+'.npz');tm=read(trace.with_suffix('.json'))
        if sha(trace)!=tm['sha256'] or tm.get('source_sha256')!=r['array_sha256']:raise ValueError('Plan trace binding changed')
        with np.load(trace,allow_pickle=False) as a:tx=a['tx'];rx=a['rx'];n=int(a['n']);ns=int(a['ns'])
        if tx.shape[-1]!=128 or n>MAX_N or (tx[...,0]!=11).any() or (tx[...,2]!=n).any() or (tx[...,3]!=ns).any():raise ValueError('Beacon wire contract invalid')
        stamp=tx[...,4:8].copy().view('<u4').reshape(tx.shape[:2])/1000
        issued=tx[...,8:12].copy().view('<u4').reshape(tx.shape[:2])/1000
        if (issued>stamp).any():raise ValueError('Packet contains unissued future command')
        np.testing.assert_equal(stamp,np.broadcast_to(np.arange(len(tx))[:,None],stamp.shape))
        messages+=int(np.prod(tx.shape[:2]));received=np.unpackbits(rx,axis=-1)[...,:n].astype(bool)
        with np.load(r['array_path'],allow_pickle=False) as raw:positions=raw['positions']
        if r.get('cache_path'):
            ep=Episode(r);original_rx=np.asarray(ep.values[::2,...,33])>0
        else:
            with np.load(r['array_path'],allow_pickle=False) as raw:original_rx=raw['rf_values'][::2,...,33]>0
        np.testing.assert_equal(received[:,ns:],original_rx);reception_checks+=len(tx)*(n-ns)*n
        path=tx[...,30:66].copy().view('<i2').reshape(*tx.shape[:2],9,2)/4
        for horizon in range(9):
            absolute=stamp[:,0]+horizon;lo=np.floor(absolute/3).astype(int);good=lo+1<len(positions)
            f=(absolute[good]-lo[good]*3)/3
            actual=positions[lo[good]]*(1-f[:,None,None])+positions[lo[good]+1]*f[:,None,None]
            error=np.abs(path[good,:,horizon]-actual)
            committed=absolute[good,None]<=issued[good]+3+1e-9
            if committed.any():
                maximum=float(error[committed].max());max_error=max(max_error,maximum);committed_points+=int(committed.sum())
                if maximum>.1250001:raise ValueError('Advertised committed trajectory differs from issued controller motion')
            if r['role']=='selection':errors_by_h[horizon].append(np.linalg.norm(path[good,:,horizon]-actual,axis=-1).ravel())
        with np.load(item['path'],allow_pickle=False) as a:
            q=a['query'];queries+=len(q)
            if len(np.unique(q,axis=0))!=len(q):raise ValueError('Duplicate within-flight query')
            if r['role']!='train' and q[:,1].any():raise ValueError('Primary cohort changed observer')
            for key in ('node_local','node_shared','edge_local','edge_shared','physics','global'):
                if not np.isfinite(a[key]).all():raise ValueError('Nonfinite model input')
        service_root=ARTIFACTS if r['role']=='confirmation' else V10
        sp=service_root/'service_cache'/(r['key']+'.npz');sm=read(sp.with_suffix('.json'))
        if sha(sp)!=sm['sha256']:raise ValueError('Packet source changed')
        with np.load(sp,allow_pickle=False) as a:
            born=a['packet_born'];status=a['packet_status'];delay=a['packet_delay'];settled+=len(born)
            boundaries+=int(((status==1)&(delay>.25)&(delay<=.25+1e-10)).sum())
            from forecast_service.replay import service_health
            exact=service_health(born,status,delay,len(a['service_healthy']))
            for key,value in exact.items():np.testing.assert_equal(value,a[key])
    for name in ('preflight.json','received_cache_preflight.json','runtime_checks.json'):
        if not read(ARTIFACTS/name)['passed']:raise ValueError('Required verification failed: '+name)
    forecast_diagnostic=[dict(horizon_s=h,selection_plan_position_rmse_m=float(np.sqrt(np.mean(np.concatenate(parts)**2))) if parts else None)
        for h,parts in enumerate(errors_by_h)]
    from .audit_predictions import main as prediction_audit
    prediction_checks=prediction_audit()
    value=dict(passed=True,created_at=now(),parent_delivery_files_preserved=len(prior['files']),
        original_dependencies_preserved=len(preserved),prior_project_protocol_sources_verified=sum(r['source_files'] for r in original),source_flights=len(all_records),
        training_flights=245,training_queries=sum(r['queries'] for r in data if r['role']=='train'),total_queries=queries,
        beacon_messages=messages,payload_bytes=128,extra_beacon_airtime_s=0.,reception_bits_checked=reception_checks,
        committed_plan_points_checked=committed_points,maximum_committed_coordinate_quantization_error_m=max_error,
        tentative_plan_diagnostic=forecast_diagnostic,settled_packets_verified=settled,deadline_roundoff_boundary_packets=boundaries,
        input_causality_verified=True,spatial_training_overlap=False,original_final_test_opened=False,prediction_checks=prediction_checks,
        primary_link_observer='First relay; previous all-relay aggregate is not the comparator',
        freeze_sha256=sha(ARTIFACTS/'freeze.json'),production_ready=False,hardware_validated=False)
    write(ARTIFACTS/'audit.json',value);print(value)


if __name__=='__main__':main()

