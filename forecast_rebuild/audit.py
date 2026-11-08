"""Final preservation, provenance and spatial-isolation audit."""
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now,verify_prior,maps,role
from .calibration import verify_freeze,audit_cohort
from .dataset import Episode
from .targets import labels
from .inference import Predictor


def main():
    frozen=verify_freeze();pins=verify_prior();registry=maps();completed=read(ARTIFACTS/'completed_comparison.json')
    from adaptive_joint_expanded.contracts import legacy_hashes
    from adaptive_joint_expanded import ARTIFACTS as prior_artifacts
    expected=read(prior_artifacts/'legacy_before.json');actual=legacy_hashes()
    if expected!=actual:raise ValueError('An original source/config/model changed')
    summary=read(completed['selected_confirmation']);dataset=read(summary['dataset_path']);rows=dataset['rows']
    geography=defaultdict(set);spatial=defaultdict(set)
    for record in rows:
        if record['role']!=role(record['region']):raise ValueError('RF source assigned to the wrong geography fold')
        geography[record['role']].add(record['region']);spatial[record['role']].add(registry[record['region']]['joint_spatial_group'])
        if sha(record['array_path'])!=record['array_sha256']:raise ValueError('A recorded RF source changed')
        if record.get('geometry_path') and sha(record['geometry_path'])!=record['geometry_sha256']:raise ValueError('Historical geometry changed')
    for a in spatial:
        for b in spatial:
            if a!=b and spatial[a]&spatial[b]:raise ValueError('Development/train spatial groups overlap')
    for key in ('selected_selection','selected_calibration','selected_confirmation','baseline_confirmation'):
        audit_cohort(read(completed[key]))
    # Recompute sparse predictions instead of trusting cached file names.
    prediction_checks=[]
    for model_label,model in [('selected',completed['selected_model']),('baseline','baseline')]:
        predictor=Predictor(model,'cuda');maximum=0.;count=0
        keys=[model_label+'_calibration',model_label+'_confirmation']
        if model_label=='selected':keys.append('selected_selection')
        for key in keys:
            result=read(completed[key]);lookup={r['key']:r for r in read(result['dataset_path'])['rows']}
            for case in result['cases']:
                episode=Episode(lookup[case['key']]);eligible=np.argwhere(episode.valid)
                chosen=eligible[np.linspace(0,len(eligible)-1,min(12,len(eligible))).astype(int)]
                fresh=predictor.predict(episode,chosen)[0];t,r,j=chosen.T
                with np.load(case['predictions_path'],allow_pickle=False) as z:stored=z['prediction'][t,r,j]
                np.testing.assert_allclose(fresh,stored,atol=2e-5,rtol=1e-5,equal_nan=True)
                finite=np.isfinite(fresh)&np.isfinite(stored)
                maximum=max(maximum,float(np.max(np.abs(fresh[finite]-stored[finite]))));count+=len(chosen)
        prediction_checks.append(dict(model=model_label,rows=count,max_abs_probability_error=maximum))
    checked=[]
    for fold in ('train','selection','calibration','confirmation'):
        cases=[r for r in rows if r['role']==fold]
        # First and last per immutable record order; includes old/new sources.
        for record in (cases[0],cases[-1]):
            episode=Episode(record)
            with np.load(record['array_path'],allow_pickle=False) as raw:
                for name,source in [('values','rf_values'),('masks','rf_masks'),('intents','rf_intents'),('links','links')]:
                    np.testing.assert_equal(np.asarray(getattr(episode,name)),raw[source])
            truth=labels(episode.links,episode.ends,record['n_search'])
            for name in ('targets','valid','currently_up','next_bin'):np.testing.assert_equal(getattr(episode,name),truth[name])
            checked.append(dict(key=record['key'],fold=fold,raw_cache_and_labels_identical=True))
    result=dict(created_at=now(),passed=True,raw_archives_hash_verified=len(rows),original_dependencies_hash_verified=len(actual),
        prior_packages=pins,fold_flights=dict(Counter(r['role'] for r in rows)),
        fold_regions={k:len(v) for k,v in geography.items()},fold_spatial_groups={k:len(v) for k,v in spatial.items()},
        training_development_spatial_overlap=False,raw_cache_spot_checks=checked,
        cached_prediction_recomputation=prediction_checks,
        runtime_precision_adapter_sha256=sha(ROOT/'runtime_stable.py'),
        runtime_test_sha256=sha(ARTIFACTS/'selected_runtime_checks.json'),
        frozen_model_and_alarm_verified=True,freeze_sha256=sha(ARTIFACTS/'model_alarm_freeze.json'),
        original_176_checkpoint_preserved=True,original_final_test_opened=False,production_ready=False)
    write(ARTIFACTS/'audit.json',result);print(result)


if __name__=='__main__':main()
