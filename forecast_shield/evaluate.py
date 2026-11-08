"""One-to-one event matching and locked operating points for all variants."""
from collections import Counter, defaultdict

import numpy as np

from .protocol import CAP, VARIANTS, maps, read, sha


def load(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('Evaluation metadata changed')
    row = read(item['path'])
    if sha(row['archive_path']) != row['archive_sha256']:
        raise ValueError('Evaluation scores changed')
    with np.load(row['archive_path'], allow_pickle=False) as data:
        scores = dict(zip(data['columns'].tolist(), data['scores'].T))
    return row, scores


def series(scores, variant):
    key = {'v18':'v18', 'planner_3s':'planner', 'shield_3s':'shield', 'shield_1s':'shield'}[variant]
    keep = scores['time_s'] % 3 == 0 if variant != 'shield_1s' else np.ones_like(scores['time_s'], bool)
    return scores['time_s'][keep], scores[key][keep]


def flight(row, scores, variant, threshold, detail=False):
    times, score = series(scores, variant)
    enabled = score >= threshold
    starts = times[enabled & ~np.r_[False,enabled[:-1]]]
    onsets = np.asarray([e['onset_s'] for e in row['events']])
    used = np.zeros(len(onsets), bool)
    lead = []
    for t in starts:
        first = np.searchsorted(onsets,t+3.,side='left')
        stop = np.searchsorted(onsets,t+7.,side='left')
        for i in range(first,stop):
            if not used[i]:
                used[i] = True
                lead.append(float(onsets[i]-t))
                break
    n = int(used.sum())
    result = dict(hours=row['duration_s']/3600., events=len(onsets), alerts=len(starts), hits=n, false_or_late=len(starts)-n)
    if detail:
        reasons = Counter()
        by_cause = defaultdict(Counter)
        for i, event in enumerate(row['events']):
            onset = event['onset_s']
            window = (times > onset-7.) & (times <= onset-3.)
            if used[i]:
                reason = 'hit'
            elif not window.any():
                reason = 'no_tick'
            elif not np.isfinite(score[window]).any():
                reason = 'no_input'
            elif (score[window]>=threshold).any():
                reason = 'already_active_or_used'
            else:
                reason = 'below_threshold'
            reasons[reason] += 1
            label = 'building_supported' if ['building'] in event['repairs'] else 'other_or_mixed'
            by_cause[label][reason] += 1
        result.update(reasons=reasons, by_cause=by_cause, lead_s=lead,
                      usable=int(np.isfinite(score).sum()), ticks=len(times))
    return result


def metrics(stats):
    total = {k:sum(r[k] for r in stats) for k in ('hours','events','alerts','hits','false_or_late')}
    return dict(total, flights=len(stats), recall=total['hits']/max(1,total['events']),
                precision=total['hits']/max(1,total['alerts']),
                false_per_hour=total['false_or_late']/max(1e-9,total['hours']))


def summarize(cohort, variant, threshold, detail=False):
    cases = [flight(row, score, variant, threshold, detail) for row, score in cohort]
    result = metrics(cases)
    if detail:
        by_cause = defaultdict(Counter)
        for case in cases:
            for key,counts in case['by_cause'].items():
                by_cause[key].update(counts)
        leads = [v for case in cases for v in case['lead_s']]
        result.update(reasons=sum((x['reasons'] for x in cases),Counter()), by_cause=by_cause,
                      median_lead_s=float(np.median(leads)) if leads else None,
                      usable=sum(r['usable'] for r in cases), ticks=sum(r['ticks'] for r in cases))
    return result


def calibrate(cohort, variant, cap):
    candidates = np.linspace(.05,1.,20) if variant.startswith('shield') else [.5]
    # A silent operating point keeps a reproducible result if none meet the budget.
    rows = [(float(value), summarize(cohort,variant,value)) for value in [*candidates,1.01]]
    feasible = [(value,stats) for value,stats in rows if stats['false_per_hour']<=cap+1e-9]
    return max(feasible,key=lambda pair:(pair[1]['recall'],pair[1]['precision'],-pair[1]['false_per_hour'],pair[0]))


def select(calibration, selection):
    reference = summarize(calibration,'v18',.5)
    budgets = dict(priority=CAP, matched=min(CAP,reference['false_per_hour']))
    profiles = {}
    for name, cap in budgets.items():
        options = {}
        for variant in VARIANTS:
            threshold, cal = calibrate(calibration,variant,cap)
            options[variant] = dict(threshold=threshold, calibration=cal,
                                    selection=summarize(selection,variant,threshold))
        eligible = [k for k,v in options.items() if v['selection']['false_per_hour']<=cap+1e-9]
        if eligible:
            chosen = max(eligible,key=lambda k:(options[k]['selection']['recall'],options[k]['selection']['precision'],
                        -options[k]['selection']['false_per_hour'],-VARIANTS.index(k)))
            profile = dict(variant=chosen, threshold=options[chosen]['threshold'])
        else:
            profile = dict(variant='v18',threshold=.5)
        profiles[name] = dict(cap=cap, **profile, options=options)
    return dict(profiles=profiles, calibration_v18=reference, selection_v18=summarize(selection,'v18',.5))


def paired(cohort, variant, threshold):
    registry = maps()
    groups = defaultdict(lambda:np.zeros(8))
    for row,scores in cohort:
        a = flight(row,scores,'v18',.5)
        b = flight(row,scores,variant,threshold)
        groups[registry[row['spec']['region']]['joint_spatial_group']] += [a['events'],a['hours'],a['hits'],b['hits'],
                                                          a['alerts'],b['alerts'],a['false_or_late'],b['false_or_late']]
    matrix = np.stack(list(groups.values()))
    size = len(matrix)
    draws = np.random.default_rng(2026110420).multinomial(size,np.full(size,1./size),size=6000)
    events,hours,a,b,alarms_a,alarms_b,fa,fb = (draws @ matrix).T
    return dict(groups=size, recall_delta_ci95=np.quantile((b-a)/events,[.025,.975]).tolist(),
                precision_delta_ci95=np.quantile(b/alarms_b.clip(1)-a/alarms_a.clip(1),[.025,.975]).tolist(),
                false_per_hour_delta_ci95=np.quantile((fb-fa)/hours,[.025,.975]).tolist())
