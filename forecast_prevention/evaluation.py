"""Summarize coverage and measure prevention on paired actual trajectories."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import ARTIFACTS
from .protocol import maps, read, sha, write, now


def survey_summary(manifest, plan):
    by_condition = defaultdict(Counter)
    by_map = defaultdict(Counter)
    pre_cut = defaultdict(list)
    for record in manifest['archived'] + manifest['historical']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('Event evidence changed')
        payload = read(record['path'])
        event_data = payload.get('cause_analysis', payload)
        if 'patterns_path' in record:
            if sha(record['patterns_path']) != record['patterns_sha256']:
                raise ValueError('Precursor evidence changed')
            observations = read(record['patterns_path'])['rows']
        else:
            observations = payload['observed_precursors']
        summary = Counter(flights=1, network_runs=len(event_data['network_events']),
                          service_runs=len(event_data['service_events']),
                          individual_link_runs=event_data['individual_undirected_link_runs'],
                          factorial_frames=event_data['verified_factorial_frames'])
        for event in event_data['network_events']:
            summary['network_onsets'] += not event['left_censored']
            minimal = event['first_second_minimal_repairs']
            summary['network_any_single_repair'] += any(len(part) == 1 for part in minimal)
            summary['network_combination_required'] += bool(minimal) and all(len(part) > 1 for part in minimal)
            summary['network_no_fixed_repair'] += not minimal
            for part in minimal:
                summary['network_repair_' + '+'.join(part)] += 1
        for event in event_data['service_events']:
            summary['service_network_down_at_onset'] += event['network_down_at_onset']
            if event['flags'] is not None:
                summary['service_packet_attributed'] += 1
                for key, value in event['flags'].items():
                    summary['service_' + key] += value
        for row in observations:
            if row['comparison'] is not None:
                pre_cut[str(row['lead_s'])].append(row)
        by_condition[record['kind']].update(summary)
        by_map[record['region']].update(summary)
    if set(by_map) != set(plan['eligible_regions']):
        raise ValueError('Full eligible-map diagnosis is incomplete')
    totals = sum(by_condition.values(), Counter())
    checks = {}
    for lead, observations in pre_cut.items():
        checks[lead] = dict(matched_windows=len(observations), measurements={})
        for field in observations[0]['pre_cut']:
            pairs = [(r['pre_cut'][field], r['comparison'][field]) for r in observations
                     if r['pre_cut'][field] is not None and r['comparison'][field] is not None]
            if not pairs:
                continue
            a, b = np.asarray(pairs).T
            checks[lead]['measurements'][field] = dict(n=len(pairs), before_cut_median=float(np.median(a)),
                 healthy_control_median=float(np.median(b)), paired_difference_mean=float((a-b).mean()))
    result = dict(created_at=now(), maps=len(by_map), archived_flights=len(manifest['archived']),
                  new_historical_flights=len(manifest['historical']), totals=totals,
                  condition=by_condition, by_map=by_map, observed_precursors=checks,
                  attribution='Minimal factor removals that restore both first half-second frames. Evidence may overlap.',
                  survey_manifest_sha256=sha(ARTIFACTS / 'survey_manifest.json'))
    write(ARTIFACTS / 'cause_summary.json', result)
    return result


def metrics(rows):
    total_time = sum(row['duration_s'] for row in rows)
    hours = total_time / 3600.
    persistent = sum(row['persistent_onsets'] for row in rows)
    cut_time = sum(row['persistent_cut_seconds'] for row in rows)
    packets = sum(row['generated_packets'] for row in rows)
    events = [event for row in rows for event in row.get('intervention', [])]
    valid = [event for event in events if not event['motion_guard_blocked']]
    return dict(flights=len(rows), flight_hours=hours, persistent_onsets=persistent,
                onsets_per_hour=persistent/hours, persistent_cut_seconds=cut_time,
                cut_seconds_per_hour=cut_time/hours, initial_persistent_cuts=sum(row['initial_persistent_cut'] for row in rows),
                worst_persistent_cut_s=max((b-a for row in rows for a,b in row['persistent_cut_intervals']), default=0.),
                mission_completed=sum(row['mission_completed'] for row in rows),
                duration_s=total_time, relay_travel_m=sum(row['relay_travel_m'] for row in rows),
                service_onsets=sum(row['service_onsets'] for row in rows),
                timely_pdr=sum(row['timely_pdr']*row['generated_packets'] for row in rows)/packets,
                coverage_mean=float(np.mean([row['observable_coverage'] for row in rows])),
                safety_violations=sum(row['actual_safety_violations'] for row in rows),
                overrides=len(events), motion_guard_blocked=len(events)-len(valid),
                prevention_before_actual_cut=sum(e['reason']=='prevention' and e['actual_pre_action_connected'] for e in valid),
                all_overrides_actual_connected=sum(e['actual_pre_action_connected'] for e in valid),
                intervention_experts=dict(Counter(e['expert'] for e in valid)),
                reason_counts=dict(sum((Counter(row.get('decision_counts', {})) for row in rows), Counter())))


def aggregate(rows):
    arms = defaultdict(list)
    for row in rows:
        arms[row['arm']].append(row)
    return {arm: metrics(cases) for arm, cases in arms.items()}


def case_key(row):
    spec = row['spec']
    return spec['region'], spec['kind'], spec['direction'], spec['seed']


def match(baseline, candidate):
    a = {case_key(row): row for row in baseline}
    b = {case_key(row): row for row in candidate}
    if len(a) != len(baseline) or len(b) != len(candidate) or a.keys() != b.keys():
        raise ValueError('Closed-loop arms do not share the same cases')
    pairs = [(a[key], b[key]) for key in sorted(a)]
    if any(x['deployment'] != y['deployment'] for x, y in pairs):
        raise ValueError('Fleet changed across paired arms')
    return pairs


def paired_ci(pairs):
    registry = maps()
    spatial = defaultdict(lambda: np.zeros(6))
    for a, b in pairs:
        group = registry[a['spec']['region']]['joint_spatial_group']
        spatial[group] += [a['persistent_onsets'], b['persistent_onsets'], a['duration_s']/3600,
                           b['duration_s']/3600, a['persistent_cut_seconds'], b['persistent_cut_seconds']]
    matrix = np.stack(list(spatial.values()))
    n = len(matrix)
    draws = np.random.default_rng(2026110417).multinomial(n, np.full(n, 1/n), size=10000)
    old, new, h0, h1, seconds0, seconds1 = (draws @ matrix).T
    return dict(spatial_groups=n, onsets_per_hour_ci95=np.quantile(new/h1-old/h0,[.025,.975]).tolist(),
                cut_seconds_per_hour_ci95=np.quantile(seconds1/h1-seconds0/h0,[.025,.975]).tolist())


def comparison(baseline, results, plan):
    base = metrics(baseline)
    out = {}
    guards = plan['decision_guards']
    for arm in sorted({row['arm'] for row in results}):
        rows = [row for row in results if row['arm'] == arm]
        pairs = match(baseline, rows)
        new = metrics(rows)
        loss = sum(a['mission_completed'] and not b['mission_completed'] for a,b in pairs)
        duration_ratio = new['duration_s']/base['duration_s']
        travel_ratio = new['relay_travel_m']/base['relay_travel_m']
        pdr_delta = new['timely_pdr']-base['timely_pdr']
        feasible = (not loss and new['safety_violations']==0 and
                    duration_ratio<=guards['max_total_duration_ratio'] and
                    travel_ratio<=guards['max_relay_travel_ratio'] and
                    pdr_delta>=guards['minimum_timely_pdr_delta'])
        out[arm] = dict(feasible=feasible, mission_completion_regressions=loss,
                        duration_ratio=duration_ratio, relay_travel_ratio=travel_ratio, timely_pdr_delta=pdr_delta,
                        onsets_per_hour_delta=new['onsets_per_hour']-base['onsets_per_hour'],
                        cut_seconds_per_hour_delta=new['cut_seconds_per_hour']-base['cut_seconds_per_hour'],
                        paired_uncertainty=paired_ci(pairs),
                        fewer_onsets=sum(b['persistent_onsets']<a['persistent_onsets'] for a,b in pairs),
                        more_onsets=sum(b['persistent_onsets']>a['persistent_onsets'] for a,b in pairs))
    return out


def select_prevention(baseline, results, plan):
    cases = aggregate(results)
    checks = comparison(baseline, results, plan)
    return min(('geometry', 'switch'), key=lambda arm: (not checks[arm]['feasible'],
               cases[arm]['onsets_per_hour'], cases[arm]['cut_seconds_per_hour'], cases[arm]['relay_travel_m'], arm))


def shared_prefix(baseline, selected):
    examples = []
    for a, b in match(baseline, selected):
        with np.load(a['archive_path'], allow_pickle=False) as x, np.load(b['archive_path'], allow_pickle=False) as y:
            ac, bc = x['commands'], y['commands']
            count = min(len(ac), len(bc))
            changes = np.flatnonzero(np.any(ac[:count] != bc[:count], axis=(1,2)))
            if not len(changes):
                continue
            step = int(changes[0])
            end = step*6
            if step == 0 or end+24>min(len(x['network']), len(y['network'])):
                continue
            np.testing.assert_array_equal(x['nodes'][:step+1], y['nodes'][:step+1])
            np.testing.assert_array_equal(x['links'][:end], y['links'][:end])
            np.testing.assert_array_equal(x['service'][:end], y['service'][:end])
            event = next((event for event in b['intervention'] if event['time_s']==step*3 and
                          event['reason']=='prevention' and not event['motion_guard_blocked']), None)
            if event is None or not x['network'][end-1]:
                continue
            for onset, stop in a['persistent_cut_intervals']:
                lead = onset-step*3.
                if 0 <= lead < 7 and y['network'][end:end+24].all():
                    examples.append(dict(spec=a['spec'], decision_s=step*3., baseline_onset_s=onset,
                                         baseline_cut_s=stop-onset, lead_s=lead, same_prefix=True,
                                         connected_after_alternative_for_s=12., intervention=event))
                    break
    return examples
