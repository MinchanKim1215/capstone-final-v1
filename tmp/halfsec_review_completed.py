"""Review complete-flight sensitivity and matched forecast uncertainty."""
from collections import defaultdict

import numpy as np

from fast_halfsec import ARTIFACTS
from fast_halfsec.context import registry
from fast_halfsec.evaluation import key, paired, pooled
from fast_halfsec.learning import Predictors, evaluate, flight
from fast_halfsec.protocol import PLAN, SEED, verify
from forecast_rebuild.contracts import now, read, sha, write


def main():
    complete = read(ARTIFACTS / 'completed.json')
    source = ARTIFACTS / 'final_results.json'
    if sha(source) != complete['result_sha256']:
        raise ValueError('Result digest differs from the completed campaign')
    result = read(source)
    plan = read(PLAN)
    verify(plan)
    rows = [read(item['path']) for item in result['confirmation_manifest']]
    groups = {mode: [row for row in rows if row['mode'] == mode] for mode in plan['modes']}
    common = set.intersection(*[{key(row) for row in group if row['mission_completed']}
                                for group in groups.values()])
    totals = {mode: pooled([row for row in group if key(row) in common])
              for mode, group in groups.items()} if common else {}
    profile = Predictors(read(ARTIFACTS/'freeze.json')['forecast']['path'])
    cases = [flight(row) for row in groups['physics_05s']]
    items = ('physics', 'early_current', 'early_history')
    stats = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    for case in cases:
        group = registry()[case['spec']['region']]['study_group']
        for item in items:
            scores = case['physics'] if item == 'physics' else profile.scores(item, case['x'])
            threshold = profile.profile['calibration'][item]['threshold']
            row = evaluate([case],[scores],threshold,3.,7.)
            stats[group][item] += [row['hits'],row['events'],row['alerts'],row['hours']]
    rng = np.random.default_rng(SEED+300)
    sampled = rng.integers(0, len(stats), (5000, len(stats)))
    samples = {name: np.asarray([row[name] for row in stats.values()])[sampled].sum(axis=1)
               for name in items}
    intervals = {}
    for label, old, new in (('learned_vs_physics','physics','early_history'),
                            ('history_increment','early_current','early_history')):
        a,b = samples[old],samples[new]
        take = a[:,1]>0
        recall = 100.*(b[take,0]-a[take,0])/a[take,1]
        false = (b[:,2]-b[:,0])/b[:,3]-(a[:,2]-a[:,0])/a[:,3]
        intervals[label] = dict(recall_delta_pp_ci95=np.quantile(recall,[.025,.975]).tolist(),
                                false_or_late_rate_delta_ci95=np.quantile(false,[.025,.975]).tolist())
    comparison = {}
    if common:
        selected = {mode:[row for row in group if key(row) in common] for mode,group in groups.items()}
        comparison['clock_only'] = paired(selected['physics_3s'],selected['physics_05s'])
        comparison['ppo_increment'] = paired(selected['learned_05s'],selected['ppo_05s'])
    review = dict(created_at=now(),source_sha256=sha(source),cohort_flights=len(plan['confirmation']),
                  all_modes_completed_conditions=len(common),
                  completion_failures={mode:[row['spec'] for row in group if not row['mission_completed']]
                                       for mode,group in groups.items()},
                  common_completed_control=totals,common_completed_comparison=comparison,
                  spatial_group_forecast_ci95=intervals,
                  scope='Sensitivity after frozen confirmation. All-condition results remain the primary evaluation.')
    write(ARTIFACTS/'result_review.json',review)
    print(dict(common_complete=len(common),forecast_intervals=intervals,control_comparisons=comparison),flush=True)


if __name__=='__main__':
    main()
