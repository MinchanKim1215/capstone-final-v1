"""Source/label/candidate preservation and independent small-batch checks."""
from collections import Counter,defaultdict
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,now,verify,V9,maps,role
from .evaluate import verify_freeze
from .data import ServiceEpisode,truth
from .features import build
from .inference import tree_predict
from forecast_rebuild.dataset import Episode,Samples
from forecast_rebuild.inference import Predictor
from forecast_rebuild.planning import public_context


def main():
    verify();frozen=verify_freeze();registry=maps();parent=read(V9/'delivery_manifest.json')
    from .runtime_aligned import check_adapter
    check_adapter()
    if read(ARTIFACTS/'runtime_checks.json')['export_sha256']!=sha(ARTIFACTS/'research_candidates.json'):raise ValueError('Runtime export differs from its test')
    for path,expected in parent['files'].items():
        if sha(path)!=expected:raise ValueError('Parent delivery artifact changed')
    original=read(V9/'dataset.json')['rows'];fresh=read(ARTIFACTS/'confirmation_dataset.json')['rows'];rows=original+fresh
    physical={r['array_path']:r['array_sha256'] for r in rows};roles=defaultdict(set)
    for record in rows:
        if role(record['region'])!=record['role']:raise ValueError('Wrong geography role')
        roles[record['role']].add(registry[record['region']]['joint_spatial_group'])
    for a in roles:
        for b in roles:
            if a!=b and roles[a]&roles[b]:raise ValueError('Optimization and held-out spatial groups overlap')
    for i,(path,expected) in enumerate(physical.items()):
        if sha(path)!=expected:raise ValueError('Original physical trajectory changed')
        if i%100==0:write(ARTIFACTS/'audit_status.json',dict(stage='physical_source_hashes',complete=i+1,total=len(physical),updated_at=now()))
    lookup={r['key']:r for r in rows};sources={r['key']:r for r in rows if r['source_kind'] in ('fresh_v9','fresh_v10')};packet_count=0;hand_checks=[];roundoff_packets=0;roundoff_sources=0
    for key,record in sources.items():
        meta=read(ARTIFACTS/'service_cache'/(key+'.json'));path=meta['path']
        if sha(path)!=meta['sha256'] or meta['request']['source_sha256']!=record['array_sha256']:raise ValueError('Service packet source changed')
        with np.load(path,allow_pickle=False) as a:
            born=a['packet_born'];st=a['packet_status'];delay=a['packet_delay'];health=a['service_healthy'];pdr=a['timely_pdr_1s']
            source_meta=read(record['metadata_path']);timely=(st==1)&(delay<=.25+1e-10)
            strict=(st==1)&(delay<=.25)
            assert len(born)==source_meta['generated_packets'] and np.isclose(strict.mean(),source_meta['timely_pdr'],atol=1e-12,rtol=1e-12)
            boundary=timely&~strict;roundoff_packets+=int(boundary.sum());roundoff_sources+=int(boundary.any())
            if boundary.any():assert np.all((delay[boundary]>.25)&(delay[boundary]<=.25+1e-10))
            for q in np.linspace(5,len(health)-1,10).astype(int):
                cutoff=q*.5-.25;take=(born>cutoff-1)&(born<=cutoff)
                expected=float(timely[take].mean());assert np.isclose(pdr[q],expected,atol=1e-12,rtol=1e-12)
                assert bool(health[q])==bool(expected>=.9)
            hand_checks.append(key);packet_count+=len(born)
        if record['source_kind']=='fresh_v9' or meta.get('physical_link_frames_exact',False):
            assert meta['physical_link_frames_exact'] and meta['positions_exact'] and meta['local_telemetry_endpoint_exact'] and meta['final_packet_outcomes_exact']
        else:assert meta['post_flight_read_only_logging']
    manifest=read(ARTIFACTS/'samples/manifest.json');group_queries=defaultdict(set);source_counts=Counter()
    for split in ('train','selection','calibration'):
        index=np.load(ARTIFACTS/'samples'/(split+'_index.npy'),mmap_mode='r')
        y=np.load(ARTIFACTS/'samples'/(split+'_y.npy'),mmap_mode='r')
        for e in np.unique(index[:,0]):
            record=manifest['records'][int(e)]
            if record['role']!=split:raise ValueError('Non-training row entered optimization')
            positions=np.flatnonzero(index[:,0]==e);query=index[positions,1:];key=record.get('clean_record_key',record['key'])
            if split=='train':
                pairs={tuple(q) for q in query}
                if group_queries[key]&pairs:raise ValueError('Clean/augmented queries duplicate a physical training sample')
                group_queries[key] |= pairs;source_counts[key]+=len(query)
            _,labels=truth(record);chosen=np.linspace(0,len(query)-1,min(16,len(query))).astype(int)
            np.testing.assert_equal(y[positions[chosen]],labels['targets'][query[chosen,0],0,0])
    for part in ('train','selection'):
        old=Samples(V9/'samples/merged_trees',part)
        if part=='train':
            for e in np.unique(old.index[:,0]):assert old.records[int(e)]['role']=='train'
    done=read(ARTIFACTS/'completed_comparison.json');checks=[]
    for item in frozen['models']:
        model=item['pin']['path'];task=item['task'];meta=read(model);predictor=Predictor(model,'cuda') if task=='link' else None
        proxy=Predictor(meta['link_model'],'cuda') if meta['family']=='service_proxy' else None;maximum=0.;count=0
        outcome=next(o for o in done['outcomes'] if o['task']==task and o['tag']==item['tag'])
        for summary_path in (item['selection_summary'],item['calibration_summary'],outcome['raw_summary']):
            summary=read(summary_path);records={r['key']:r for r in read(summary['dataset_path'])['rows']}
            for kind in ('clear','continuous','intermittent'):
                case=next(c for c in summary['cases'] if c['kind']==kind);record=records[case['key']];ep=Episode(record)
                if task=='link':
                    q=ep.eligible(legacy=True);q=q[np.linspace(0,len(q)-1,min(12,len(q))).astype(int)];expected=predictor.predict(ep,q)[0];t,r,j=q.T
                    with np.load(case['predictions_path'],allow_pickle=False) as a:actual=a['prediction'][t,r,j]
                else:
                    view=ServiceEpisode(record);t=np.flatnonzero(view.valid[:,0,0]&view.legacy_support[:,0,0]);t=t[np.linspace(0,len(t)-1,min(12,len(t))).astype(int)]
                    if proxy:
                        expected=np.full((len(t),6),np.nan,np.float32)
                        for k,time in enumerate(t):
                            peers=[j for j in range(ep.values.shape[2]) if j!=record['n_search'] and ep.legacy_support[time,0,j]]
                            raw=proxy.predict(ep,np.array([[time,0,j] for j in peers]))[0]
                            for head in (2,5):expected[k,head]=raw[:,head].max()
                    else:
                        city,_=public_context(record);expected=tree_predict(model,build(ep,np.c_[t,np.zeros(len(t),int)],record['n_search'],city))
                    with np.load(case['predictions_path'],allow_pickle=False) as a:actual=a['prediction'][t,0,0]
                if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Cached predictions changed')
                np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5,equal_nan=True)
                good=np.isfinite(expected)&np.isfinite(actual);maximum=max(maximum,float(np.max(np.abs(actual[good]-expected[good]))));count+=len(expected)
        checks.append(dict(task=task,tag=item['tag'],queries=count,max_abs_probability_error=maximum))
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as PRIOR
    preserved=legacy_hashes()
    if preserved!=read(PRIOR/'legacy_before.json'):raise ValueError('Legacy sources or weights changed')
    verify_freeze();write(ARTIFACTS/'audit.json',dict(passed=True,created_at=now(),parent_delivery_files_preserved=len(parent['files']),
        original_dependencies_preserved=len(preserved),physical_archives_verified=len(physical),packet_source_archives_verified=len(sources),settled_packets_verified=packet_count,
        deadline_hand_check_sources=len(hand_checks),training_physical_service_sources=len(source_counts),training_service_query_rows=sum(source_counts.values()),
        deadline_tolerance_s=1e-10,legacy_strict_deadline_counter_separately_verified=True,roundoff_boundary_packets=roundoff_packets,roundoff_boundary_sources=roundoff_sources,
        clean_augmented_queries_disjoint=True,train_roles_verified=True,spatial_overlap=False,prediction_checks=checks,
        original_final_test_opened=False,production_ready=False,freeze_sha256=sha(ARTIFACTS/'freeze.json')))
    print(ARTIFACTS/'audit.json')


if __name__=='__main__':main()
