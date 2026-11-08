"""Independent whole-flight checks and paired comparisons."""
from collections import defaultdict

import numpy as np

from adaptive_joint_expanded.contracts import DEPLOYMENT_BY_KEY
from final_v1_fast.runner import intervals
from forecast_rebuild.contracts import sha
from .context import registry
from .learning import HEADS, Predictors, evaluate, flight
from .protocol import SEED


def key(row):
    spec = row['spec']
    return spec['region'],spec['direction'],spec['seed'],spec['kind']


def pooled(rows):
    hours = sum(row['duration_s'] for row in rows)/3600.
    generated = sum(row['generated'] for row in rows)
    starts = sum(row['onsets'] for row in rows)
    return dict(flights=len(rows), hours=hours, starts=starts,
                starts_per_h=starts/hours, cut_seconds=sum(row['cut_seconds'] for row in rows),
                cut_seconds_per_h=sum(row['cut_seconds'] for row in rows)/hours,
                timely_pdr=sum(row['timely'] for row in rows)/max(1,generated),
                network_fraction=sum(row['network_fraction']*row['duration_s'] for row in rows)/(hours*3600.),
                mean_mission_s=hours*3600./len(rows),
                relay_m=float(np.mean([row['relay_travel_m'] for row in rows])),
                completed=sum(row['mission_completed'] for row in rows),
                violations=sum(not row['safety_passed'] for row in rows),
                prevention_requests=sum(row['prevention_requests'] for row in rows),
                median_planner_p95_ms=float(np.median([row['planner_p95_ms'] for row in rows])),
                planner_over_200ms=sum(row.get('planner_over_200ms',0) for row in rows),
                planner_evaluations=sum(row.get('plan_ticks',0) for row in rows))


def paired(old, new):
    a,b = {key(row):row for row in old},{key(row):row for row in new}
    if set(a) != set(b) or len(a)!=len(old) or len(b)!=len(new):
        raise ValueError('Unpaired whole-flight comparison')
    groups = defaultdict(list)
    for condition in sorted(a):
        groups[registry()[condition[0]]['study_group']].append(condition)
    sums = np.asarray([[sum(a[k]['onsets'] for k in keys),sum(a[k]['duration_s'] for k in keys)/3600.,
                        sum(b[k]['onsets'] for k in keys),sum(b[k]['duration_s'] for k in keys)/3600.]
                       for keys in groups.values()])
    rng = np.random.default_rng(SEED+930)
    draw = sums[rng.integers(0,len(sums),(5000,len(sums)))].sum(axis=1)
    ci = np.quantile(draw[:,2]/draw[:,3]-draw[:,0]/draw[:,1],[.025,.975]).tolist()
    stats_a,stats_b = pooled(old),pooled(new)
    return dict(groups=len(groups), starts_delta=stats_b['starts']-stats_a['starts'],
                starts_per_h_delta=stats_b['starts_per_h']-stats_a['starts_per_h'], ci95=ci,
                duration_fraction=stats_b['hours']/stats_a['hours']-1.,
                timely_pdr_pp=100.*(stats_b['timely_pdr']-stats_a['timely_pdr']),
                completion_regressions=sum(a[k]['mission_completed'] and not b[k]['mission_completed'] for k in a),
                fewer_starts=sum(b[k]['onsets']<a[k]['onsets'] for k in a),
                more_starts=sum(b[k]['onsets']>a[k]['onsets'] for k in a))


def accepted(difference, summary, rules):
    return bool(summary['violations']<=rules['safety_violations'] and
                difference['completion_regressions']<=rules['completion_regressions'] and
                difference['timely_pdr_pp']>=-rules['max_timely_loss_pp'] and
                difference['duration_fraction']<=rules['max_duration_increase_fraction'] and
                difference['starts_delta']<=0 and difference['ci95'][1]<0.)


def audit(rows):
    frames = actions = 0
    shortest_gap = {}
    for row in rows:
        if sha(row['archive_path']) != row['archive_sha256']:
            raise ValueError('Changed confirmation archive')
        with np.load(row['archive_path'],allow_pickle=False) as z:
            links = z['links']
            reach = links.copy()
            for node in range(reach.shape[-1]):
                reach |= reach[:,:,node,None] & reach[:,None,node,:]
            ns = DEPLOYMENT_BY_KEY[row['deployment']].ns
            net = reach[:,0,1:ns].all(axis=-1)
            np.testing.assert_array_equal(net,z['network'])
            cuts = intervals(net,.5)
            if cuts != row['cuts'] or sum(a>0. for a,b in cuts) != row['onsets']:
                raise ValueError('Persistent-onset recomputation mismatch')
            frames += len(net)
        if not row['safety_passed'] or not row['packet_conserved']:
            raise ValueError('Safety or packet conservation failed')
        if row['mode']=='final_v1_3s':
            continue
        tick = 3. if row['mode']=='physics_3s' else .5
        times = np.asarray([event['time_s'] for event in row['events']])
        np.testing.assert_allclose(times/tick,np.rint(times/tick),atol=1e-9)
        for event in row['events']:
            if abs(event['ready_s']-event['time_s']-.2)>1e-8:
                raise ValueError('Planner readiness did not wait 0.2 seconds')
            if event.get('published_s',event['ready_s']) < event['ready_s']-1e-8:
                raise ValueError('Command published before ready')
        for applied_time, relay, offset, sent_time in row['applied_changes']:
            if applied_time-sent_time < .1-1e-8:
                raise ValueError('Command applied before transport delay')
            actions += 1
        if len(times)>1:
            mode = row['mode']
            shortest_gap[mode] = min(shortest_gap.get(mode,1e9),float(np.diff(times).min()))
    return dict(flights=len(rows), rf_frames=frames, exact_network_recomputed=True,
                safety_and_packet_conservation=True, offset_applications=actions,
                shortest_request_gap_s=shortest_gap, command_clocks_checked=True)


def summarize(rows, plan, forecast):
    grouped = {mode:[row for row in rows if row['mode']==mode] for mode in plan['modes']}
    aggregate = {mode:pooled(group) for mode,group in grouped.items()}
    comparisons = dict()
    for name, old, new in (
        ('clock_only','physics_3s','physics_05s'),
        ('forecast_in_same_engine','physics_05s','learned_05s'),
        ('ppo_increment','learned_05s','ppo_05s'),
        ('learned_vs_same_3s','physics_3s','learned_05s'),
        ('ppo_vs_same_3s','physics_3s','ppo_05s'),
        ('learned_vs_final_v1','final_v1_3s','learned_05s'),
        ('ppo_vs_final_v1','final_v1_3s','ppo_05s'),
    ):
        difference = paired(grouped[old],grouped[new])
        difference.update(old=old,new=new,passed=accepted(difference,aggregate[new],plan['acceptance']))
        comparisons[name] = difference
    model = Predictors(forecast['path'])
    same = [flight(row) for row in grouped['physics_05s']]
    predictions = {}
    for name, thresholds in model.profile['calibration'].items():
        scores = [case['physics'] if name=='physics' else model.scores(name,case['x']) for case in same]
        predictions[name] = evaluate(same,scores,thresholds['threshold'],*thresholds['window'],ranking=True)
        predictions[name]['threshold'] = thresholds['threshold']
        predictions[name]['window_s'] = thresholds['window']
    return dict(control=aggregate,paired=comparisons,forecast=predictions,audit=audit(rows),
                by_interference={kind:{mode:pooled([row for row in group if row['spec']['kind']==kind])
                                       for mode,group in grouped.items()}
                                 for kind in ('clear','intermittent','continuous')})
