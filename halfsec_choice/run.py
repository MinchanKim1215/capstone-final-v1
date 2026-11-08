"""Fit the guarded selector, then compare it on previously unused maps."""
import argparse
from collections import defaultdict
from functools import lru_cache
from time import perf_counter
from unittest.mock import patch
import os

import lightgbm as lgb
import numpy as np

from fast_halfsec import runner
from fast_halfsec.evaluation import accepted, audit, key, paired, pooled
from forecast_rebuild.contracts import digest, now, read, sha, write

from . import ARTIFACTS, FORECAST
from .choice import INPUTS, Selector
from .experiment import PROTOCOL, archive_paths, build_data, cached, exclusive_run, gather, prepare, save, status
from .records import cost, predictors

MODEL = ARTIFACTS / 'models/selector.txt'
PROFILE = ARTIFACTS / 'models/selector_profile.json'
FREEZE = ARTIFACTS / 'freeze.json'
REFERENCE = 'physics_05s'
LEARNED = 'choice_05s'
THRESHOLDS = (.1, .25, .5, .75, 1., 1.5, 2., 3., 4., 6.)
EXPLORATORY_THRESHOLD = .25
PARAMETERS = dict(objective='regression', metric='l2', learning_rate=.05, num_leaves=7,
                  max_depth=3, min_data_in_leaf=10, lambda_l2=2., force_col_wise=True,
                  deterministic=True, seed=20261108, num_threads=4, verbosity=-1)
ROUNDS = 110


def load_split(manifest, role):
    rows = [row for row in manifest['records'] if row['role'] == role]
    data = defaultdict(list)
    for index, row in enumerate(rows):
        if sha(row['archive_path']) != row['archive_sha256']:
            raise ValueError(f'Modified branch archive: {row["source"]}')
        with np.load(row['archive_path'], allow_pickle=False) as archive:
            for name in ('features', 'reference', 'alternative', 'live'):
                data[name].append(archive[name])
            data['anchor'].append(index * 1_000_000 + archive['frames'].astype(np.int64))
            data['group'].append(np.repeat(row['group'], len(archive['frames'])))
    arrays = {name: np.concatenate(values) for name, values in data.items()}
    arrays['gain'] = cost(arrays['reference']) - cost(arrays['alternative'])
    if not len(arrays['gain']) or not np.isfinite(arrays['features']).all():
        raise ValueError(f'No usable branch pairs in {role}')
    return arrays


def top_per_anchor(anchor, scores):
    order = np.lexsort((-scores, anchor))
    ordered = anchor[order]
    return order[np.r_[True, ordered[1:] != ordered[:-1]]]


def operating_stats(data, indices):
    reference, alternative = data['reference'][indices], data['alternative'][indices]
    gain = data['gain'][indices]
    return dict(changes=len(indices), maps=len(set(data['group'][indices])),
                actually_connected=int(data['live'][indices].sum()),
                beneficial=int((gain > 1e-6).sum()), worse=int((gain < -1e-6).sum()),
                starts_saved=int((reference[:, 0] - alternative[:, 0]).sum()),
                cut_seconds_saved=float((reference[:, 1] - alternative[:, 1]).sum()),
                gain_sum=float(gain.sum()), gain_mean=float(gain.mean()) if len(gain) else 0.)


def calibrate(model, data):
    scores = model.predict(data['features'], num_threads=1)
    top = top_per_anchor(data['anchor'], scores)
    candidates = [dict(threshold=threshold, **operating_stats(data, top[scores[top] >= threshold]))
                  for threshold in THRESHOLDS]
    qualified = [row for row in candidates if row['changes'] >= 6 and row['maps'] >= 3
                 and row['starts_saved'] >= 0 and row['cut_seconds_saved'] >= 0.
                 and row['gain_mean'] >= .1
                 and (row['starts_saved'] > 0 or row['cut_seconds_saved'] > 0.)]
    selected = max(qualified, key=lambda row: (row['starts_saved'], row['cut_seconds_saved'],
                                               row['gain_sum'], row['threshold']), default=None)
    threshold = selected['threshold'] if selected else EXPLORATORY_THRESHOLD
    oracle = top_per_anchor(data['anchor'], data['gain'])
    return dict(threshold=threshold, qualified=selected is not None,
                selected=next(row for row in candidates if row['threshold'] == threshold),
                candidates=candidates, anchors=len(top), pairs=len(scores),
                mae=float(np.abs(data['gain'] - scores).mean()),
                zero_prediction_mae=float(np.abs(data['gain']).mean()),
                oracle=operating_stats(data, oracle[data['gain'][oracle] > 1e-6]))


def fit(manifest):
    pins = dict(protocol=sha(PROTOCOL), data=sha(ARTIFACTS / 'branch_manifest.json'),
                forecast=sha(FORECAST), source=sha(__file__))
    if PROFILE.exists():
        profile = read(PROFILE)
        if profile['pins'] != pins or profile['model_sha256'] != sha(MODEL):
            raise ValueError('Checkpoint does not match frozen study inputs')
        return profile
    train = load_split(manifest, 'train')
    validation = load_split(manifest, 'calibration')
    _, index, count = np.unique(train['anchor'], return_inverse=True, return_counts=True)
    weights = 1. / count[index]
    training = lgb.Dataset(train['features'], label=train['gain'],
                           weight=weights / weights.mean(), feature_name=list(INPUTS))
    status('fitting_selector', train_pairs=len(train['gain']), calibration_pairs=len(validation['gain']),
           boosted_rounds=ROUNDS)
    start = perf_counter()
    model = lgb.train(PARAMETERS, training, num_boost_round=ROUNDS)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    pending = MODEL.with_suffix('.pending.txt')
    pending.write_text(model.model_to_string(), encoding='utf-8')
    os.replace(pending, MODEL)
    profile = dict(pins=pins, model_sha256=sha(MODEL), trained_at=now(), inputs=list(INPUTS),
                   parameters=PARAMETERS, rounds=model.current_iteration(), fit_wall_s=perf_counter() - start,
                   calibration=calibrate(model, validation))
    write(PROFILE, profile)
    status('selector_trained', model=str(MODEL), calibration=profile['calibration']['selected'],
           calibration_qualified=profile['calibration']['qualified'])
    print(f"trained {profile['rounds']} trees, {len(train['gain'])} training pairs, "
          f"{len(validation['gain'])} calibration pairs, {profile['calibration']['selected']}", flush=True)
    return profile


def freeze():
    pins = dict(protocol_sha256=sha(PROTOCOL), profile_sha256=sha(PROFILE),
                model_sha256=sha(MODEL), driver_sha256=sha(__file__))
    if FREEZE.exists():
        if read(FREEZE)['pins'] != pins:
            raise ValueError('Learned model or verification protocol changed')
    else:
        write(FREEZE, dict(pins=pins, frozen_at=now(), decision='No tuning on confirmation flights.'))
    return sha(FREEZE)


@lru_cache(maxsize=1)
def load_model():
    profile = read(PROFILE)
    if sha(MODEL) != profile['model_sha256'] or profile['inputs'] != list(INPUTS):
        raise ValueError('Invalid learned selector')
    return lgb.Booster(model_str=MODEL.read_text(encoding='utf-8')), profile


def mission_jobs(jobs):
    results = []
    for request in jobs:
        spec, mode = request['spec'], request['mode']
        if sha(FREEZE) != request['freeze'] or spec not in read(PROTOCOL)['fresh_confirmation']:
            raise ValueError('Confirmation flight was not in the frozen test')
        paths = archive_paths(f'confirmation/{mode}', digest(spec)[:20])
        old = cached(paths, request)
        if old is not None:
            results.append(old)
            continue
        if mode == LEARNED:
            model, profile = load_model()
            selector = Selector(model, profile['calibration']['threshold'], predictors())
            # Swap only the option ordering in this single-process flight.
            with patch.object(runner, 'options_for', selector):
                row, arrays = runner.full_mission(spec, REFERENCE)
            row.update(mode=mode, learned_candidates=selector.scored, eligible_ticks=selector.eligible)
            row['learned_changes'] = sum('expected_branch_gain' in event for event in row['events'])
        else:
            row, arrays = runner.full_mission(spec, REFERENCE)
        results.append(save(paths, request, row, arrays))
    return results


def verify_identical(reference, learned):
    ref = {key(row): row for row in reference}
    identical = 0
    for row in learned:
        if row['learned_changes']:
            continue
        previous = ref[key(row)]
        with np.load(row['archive_path'], allow_pickle=False) as a, np.load(previous['archive_path'], allow_pickle=False) as b:
            for name in ('positions', 'links', 'network', 'applied_offsets'):
                np.testing.assert_array_equal(a[name], b[name])
        identical += 1
    return identical


def confirm(plan, manifest, profile, workers):
    frozen = freeze()
    groups = defaultdict(list)
    for spec in plan['fresh_confirmation']:
        for mode in (REFERENCE, LEARNED):
            groups[spec['region']].append(dict(spec=spec, mode=mode, freeze=frozen))
    rows = gather(list(groups.values()), mission_jobs, 'fresh_map_mission_comparison', workers)
    status('auditing_missions', flights=len(rows))
    reference = [row for row in rows if row['mode'] == REFERENCE]
    learned = [row for row in rows if row['mode'] == LEARNED]
    summary = pooled(learned)
    reference_summary = pooled(reference)
    difference = paired(reference, learned)
    checked = audit(rows)
    checked['unchanged_choice_flights_exactly_equal'] = verify_identical(reference, learned)
    decision = (profile['calibration']['qualified'] and accepted(difference, summary, plan['acceptance'])
                and summary['cut_seconds_per_h'] <= reference_summary['cut_seconds_per_h'])
    by_case = []
    controls = {key(row): row for row in reference}
    for row in learned:
        base = controls[key(row)]
        by_case.append(dict(spec=row['spec'], before=base['onsets'], after=row['onsets'],
                            seconds_before=base['cut_seconds'], seconds_after=row['cut_seconds'],
                            completed_before=base['mission_completed'], completed_after=row['mission_completed'],
                            learned_changes=row['learned_changes'], eligible_ticks=row['eligible_ticks'],
                            learned_candidates=row['learned_candidates'], before_timely=base['timely_pdr'],
                            after_timely=row['timely_pdr'], duration_before=base['duration_s'],
                            duration_after=row['duration_s']))
    result = dict(completed_at=now(), freeze_sha256=frozen, branch_data=manifest['counts'],
                  calibration=profile['calibration'], baseline=reference_summary, learned=summary,
                  paired=difference, audit=checked, acceptance_passed=decision,
                  learned_changes=sum(row['learned_changes'] for row in learned),
                  changed_flights=sum(row['learned_changes'] > 0 for row in learned),
                  eligible_ticks=sum(row['eligible_ticks'] for row in learned),
                  learned_candidates=sum(row['learned_candidates'] for row in learned), by_case=by_case,
                  archives=[{name: row[name] for name in ('spec', 'mode', 'archive_path', 'archive_sha256')}
                            for row in rows])
    write(ARTIFACTS / 'results.json', result)
    status('completed', state='completed', acceptance_passed=decision,
           flights=len(rows), learned_changes=result['learned_changes'],
           before_onsets=reference_summary['starts'], after_onsets=summary['starts'])
    print(f"completed {len(rows)} flights: starts {reference_summary['starts']} -> {summary['starts']}; "
          f"changes={result['learned_changes']}; adopted={decision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    with exclusive_run():
        try:
            status('verifying_training_inputs')
            plan = prepare()
            manifest = build_data(plan, args.workers)
            profile = fit(manifest)
            confirm(plan, manifest, profile, args.workers)
        except Exception as error:
            status('failed', state='failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
