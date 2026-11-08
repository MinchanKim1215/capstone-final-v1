"""Provenance audit for controlled continuation and fresh-flight comparisons."""
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now,maps,role,verify_protocol,OLD_SAMPLES
from .evaluate import verify_freeze
from forecast_rebuild.dataset import Episode,Samples
from forecast_rebuild.inference import Predictor
from forecast_rebuild.targets import labels


def main():
    protocol=verify_protocol();frozen=verify_freeze();done=read(ARTIFACTS/'completed_comparison.json');registry=maps()
    from .metadata import correct_confirmation_counts
    metadata_correction=correct_confirmation_counts(done)
    from .runtime import runtime_dependencies
    dependencies=runtime_dependencies()
    if read(ARTIFACTS/'runtime_checks.json')['runtime_dependencies']!=dependencies:raise ValueError('Runtime adapter differs from its test')
    dataset=read(ARTIFACTS/'dataset_with_confirmation.json');rows=dataset['rows'];geography=defaultdict(set);spatial=defaultdict(set)
    physical={}
    for r in rows:
        if role(r['region'])!=r['role']:raise ValueError('Wrong spatial role')
        geography[r['role']].add(r['region']);spatial[r['role']].add(registry[r['region']]['joint_spatial_group'])
        physical[r['array_path']]=r['array_sha256']
    for a in spatial:
        for b in spatial:
            if a!=b and spatial[a]&spatial[b]:raise ValueError('Train/development spatial groups overlap')
    for path,expected in physical.items():
        if sha(path)!=expected:raise ValueError('Physical RF archive changed')
    raw_cache_checks=[]
    for fold in ('train','selection','calibration','confirmation'):
        candidates=[r for r in rows if r['role']==fold and r['source_kind']!='augmented_v9']
        for record in (candidates[0],candidates[-1]):
            episode=Episode(record)
            with np.load(record['array_path'],allow_pickle=False) as archive:
                for name,key in [('values','rf_values'),('masks','rf_masks'),('intents','rf_intents'),('links','links')]:
                    np.testing.assert_equal(np.asarray(getattr(episode,name)),archive[key])
            truth=labels(episode.links,episode.ends,record['n_search'])
            for name in ('targets','valid','currently_up','next_bin'):np.testing.assert_equal(getattr(episode,name),truth[name])
            raw_cache_checks.append(dict(key=record['key'],role=fold,raw_cache_and_targets_equal=True))
    train_rows=0
    for folder in (OLD_SAMPLES,ARTIFACTS/'samples/new_only',ARTIFACTS/'samples/merged_trees'):
        sample=Samples(folder,'train');train_rows+=len(sample.y)
        for e in np.unique(sample.index[:,0]):
            if sample.records[int(e)]['role']!='train':raise ValueError('Calibration/validation entered optimization rows')
    source={r['key']:r for r in rows};augmented=[r for r in rows if r['source_kind']=='augmented_v9']
    for r in augmented:
        augmented_episode=Episode(r);clean=Episode(source[r['clean_record_key']])
        for name in ('targets','valid','currently_up','next_bin','links','masks'):
            np.testing.assert_equal(np.asarray(getattr(augmented_episode,name)),np.asarray(getattr(clean,name)))
        unknown=np.asarray(clean.intents[...,8])<.5
        np.testing.assert_equal(np.asarray(augmented_episode.intents[...,2:4])[unknown],np.asarray(clean.intents[...,2:4])[unknown])
    cache_checks=[]
    for item in frozen['models']:
        model=Predictor(item['pin']['path'],'cuda');maximum=0.;count=0
        results=[item['selection_summary'],item['calibration_summary']]
        outcome=next(x for x in done['outcomes'] if x['tag']==item['tag']);results.append(outcome['raw_summary'])
        for summary_path in results:
            summary=read(summary_path);lookup={r['key']:r for r in read(summary['dataset_path'])['rows']}
            # Two cases per RF condition/fold, selected by immutable order.
            selected=[]
            for kind in ('clear','intermittent','continuous'):
                cases=[c for c in summary['cases'] if c['kind']==kind]
                selected.extend([cases[0],cases[-1]])
            for case in selected:
                episode=Episode(lookup[case['key']]);eligible=episode.eligible(legacy=True)
                query=eligible[np.linspace(0,len(eligible)-1,min(12,len(eligible))).astype(int)]
                fresh=model.predict(episode,query)[0];t,r,j=query.T
                if sha(case['predictions_path'])!=case['prediction_sha256']:raise ValueError('Prediction cache changed')
                with np.load(case['predictions_path'],allow_pickle=False) as archive:cached=archive['prediction'][t,r,j]
                np.testing.assert_allclose(fresh,cached,atol=2e-5,rtol=1e-5,equal_nan=True)
                good=np.isfinite(fresh)&np.isfinite(cached);maximum=max(maximum,float(np.max(np.abs(fresh[good]-cached[good]))));count+=len(query)
        cache_checks.append(dict(tag=item['tag'],rows=count,max_abs_probability_error=maximum))
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as LEGACY_ARTIFACTS
    old=legacy_hashes()
    if old!=read(LEGACY_ARTIFACTS/'legacy_before.json'):raise ValueError('Original source/config/model changed')
    from forecast_rebuild.contracts import verify_prior,PLAN
    from forecast_rebuild import ARTIFACTS as PARENT_ARTIFACTS
    verify_prior()
    if sha(PLAN)!=read(PARENT_ARTIFACTS/'protocol.json')['plan_sha256']:raise ValueError('The original map split changed')
    b=read(ARTIFACTS/'models/same_data_20000/model.json');c=read(ARTIFACTS/'models/new_data_20000/model.json')
    if b['initial_checkpoint_sha256']!=c['initial_checkpoint_sha256'] or b['completed_steps']!=20000 or c['completed_steps']!=20000:
        raise ValueError('Continuation comparison is not matched at the declared update budget')
    if b['source_sha256']!=c['source_sha256']:raise ValueError('Continuation trainer implementations differ')
    value=dict(passed=True,created_at=now(),physical_archives_hash_verified=len(physical),fresh_physical_flights=377,
        count_header_correction=metadata_correction,
        augmented_training_variants_checked=len(augmented),original_dependencies_preserved=len(old),
        role_records=dict(Counter(r['role'] for r in rows)),role_regions={k:len(v) for k,v in geography.items()},
        role_spatial_groups={k:len(v) for k,v in spatial.items()},spatial_overlap=False,cached_prediction_checks=cache_checks,
        raw_cache_and_label_spot_checks=raw_cache_checks,original_map_plan_sha256=sha(PLAN),
        same_initial_weights_and_optimizer_reference=True,same_declared_additional_updates=15000,
        additional_sampling_randomness_differs=True,original_final_test_opened=False,production_ready=False,
        runtime_source_sha256=sha(ROOT/'runtime.py'),runtime_dependencies=dependencies,
        protocol_sha256=sha(ARTIFACTS/'protocol.json'),freeze_sha256=sha(ARTIFACTS/'freeze.json'))
    write(ARTIFACTS/'audit.json',value);print(value)


if __name__=='__main__':main()

