"""모형/임계값을 바꾸지 않는 사후 오류 분해입니다. 주 평가를 대체하지 않습니다."""
import json

import numpy as np

from swarm_sim.forecast_evaluation import binary_metrics
from swarm_sim.regions import ROOT, read_json, write_json
from upgrade.data import collection, checked_observations
from upgrade.evaluate import RESULTS, grouped


def run():
    selection = read_json(RESULTS/'selection.json')
    selected = selection['selected_variant']+'_s11'
    names = ['A0_three_regions_s11', selected]
    manifest, protocol = collection()
    records = [r for r in manifest['records'] if r['split'] == 'test']
    rows, flags, regions, completed, full_stationary, storage = [], [], [], [], [], {n: [] for n in names}
    for record in records:
        x, _ = checked_observations(record)
        with np.load(RESULTS/'dense'/selected/(record['id']+'.npz'), allow_pickle=False) as f:
            index, y = f['index'], f['y']
        moving = np.linalg.norm(x[..., :2], axis=-1) > .25
        cumulative = np.concatenate((np.zeros((1, 8, 8), np.int32), np.cumsum(moving, axis=0)), axis=0)
        k, i, j = index.T
        active = (cumulative[k+1,i,j]-cumulative[k-39,i,j]) > 0
        any_move = moving.any(axis=(1,2))
        prefix = np.r_[0, np.cumsum(any_move)]
        stationary30 = (k >= 59) & (prefix[k+1]-prefix[np.maximum(0,k-59)] == 0)
        rows.append(y); flags.append(active); full_stationary.append(stationary30)
        regions.append(np.full(len(y), record['region']))
        completed.append(np.full(len(y), record['metadata']['mission']['mission_completed']))
        for name in names:
            with np.load(RESULTS/'dense'/name/(record['id']+'.npz'), allow_pickle=False) as f:
                np.testing.assert_array_equal(index, f['index'])
                storage[name].append(f['probability'])
    y, moving, complete, stationary, region = [np.concatenate(v) for v in (rows, flags, completed, full_stationary, regions)]
    result = {'posthoc_diagnostic_only': True, 'primary_test_cohort_not_changed': True, 'models_and_thresholds_not_changed': True,
              'rules': {'moving_history': 'observer speed >0.25m/s at least once in past 20s',
                        'all_swarm_stationary_30s': 'no drone speed >0.25m/s throughout preceding 30s',
                        'mission_completed': 'full original path completed before 7200s deadline'}, 'results': {}}
    for name in names:
        p = np.concatenate(storage[name])
        threshold = read_json(RESULTS/'runs'/name/'evaluation.json')['threshold']
        slices = {'all': np.ones(len(y),bool), 'moving_observer_history': moving, 'stationary_observer_history': ~moving,
                  'completed_mission_records': complete, 'deadline_mission_records': ~complete,
                  'all_swarm_stationary_30s': stationary, 'other_than_all_swarm_stationary_30s': ~stationary}
        metrics = {key: grouped(y[take], p[take], region[take], threshold) for key,take in slices.items()}
        fp = (p.astype(float) >= threshold) & (y == 0)
        metrics['false_alarm_distribution'] = {'all_fp': int(fp.sum()),
            'fp_on_stationary_observer_history_fraction': float(np.mean(~moving[fp])),
            'fp_on_deadline_missions_fraction': float(np.mean(~complete[fp])),
            'fp_when_all_swarm_stationary_30s_fraction': float(np.mean(stationary[fp]))}
        result['results'][name] = metrics
    write_json(RESULTS/'posthoc_diagnostics.json', result)
    print(json.dumps({'stage': 'posthoc_diagnostics', 'primary_unchanged': True,
                      'selected_false_alarm_distribution': result['results'][selected]['false_alarm_distribution']},ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    run()
