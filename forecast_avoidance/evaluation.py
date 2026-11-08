"""Separate passive detection, direct branches and full-mission prevention."""
from collections import Counter

import numpy as np

from . import ARTIFACTS
from .protocol import V17, read, sha, write


def cause_label(repairs):
    single = sorted(part[0] for part in repairs if len(part) == 1)
    return '+'.join(single) if single else ('combined_only' if repairs else 'unresolved')


def recheck_survey():
    manifest = read(V17 / 'survey_manifest.json')
    evidence = Counter()
    by_condition = {kind: Counter() for kind in ('clear', 'intermittent', 'continuous')}
    by_region = {}
    single_isolation = 0
    for key in ('archived', 'historical'):
        for record in manifest[key]:
            if sha(record['path']) != record['sha256']:
                raise ValueError('Cause file changed: ' + record['path'])
            data = read(record['path'])
            if key == 'historical':
                data = data['cause_analysis']
            by_map = by_region.setdefault(record['region'], Counter())
            for cut in data['network_events']:
                label = cause_label(cut['first_second_minimal_repairs'])
                evidence[label] += 1
                by_condition[record['kind']][label] += 1
                by_map[label] += 1
                groups = cut['search_components_with_relays']
                single_isolation += len(groups) == 2 and min(len(g) for g in groups) == 1
    if len(by_region) != 312 or evidence.total() != 3936:
        raise ValueError('Broad attribution coverage changed')
    result = dict(maps=len(by_region), flights=len(manifest['archived'])+len(manifest['historical']),
                  total=evidence.total(), exclusive_evidence=evidence, by_condition=by_condition,
                  by_region=by_region, one_aircraft_isolated=single_isolation,
                  survey_manifest_sha256=sha(V17 / 'survey_manifest.json'))
    write(ARTIFACTS / 'survey_recheck.json', result)
    return result


def advisory(row):
    events = [event for event in row['cut_evidence'] if event['onset_s'] > 0.]
    matched = {}
    alerts = []
    active = False
    for tick in row['advisory']:
        predicted = tick['predicted']
        risk = tick['estimated_connected'] and any(3 <= h < 7 for h in predicted)
        if risk and not active:
            t = tick['time_s']
            event = next((k for k, event in enumerate(events) if k not in matched
                          and 3 <= event['onset_s']-t < 7), None)
            if event is not None:
                matched[event] = t
            alerts.append(dict(time_s=t, event=event, expert=tick['expert']))
        active = risk
    outcomes = Counter()
    groups = {}
    branches = []
    for k, event in enumerate(events):
        onset = event['onset_s']
        window = [tick for tick in row['advisory'] if 3 <= onset-tick['time_s'] < 7]
        risk = [tick for tick in window if tick['estimated_connected'] and any(3 <= h < 7 for h in tick['predicted'])]
        choices = [tick for tick in risk if tick['reason'] == 'prevention']
        if k in matched:
            reason = 'timely_alert'
        elif risk:
            reason = 'alert_already_active_or_used'
        elif not window:
            reason = 'no_decision_in_lead_window'
        elif all(tick['reason']=='insufficient_observation' for tick in window):
            reason = 'observations_missing'
        else:
            reason = 'forecast_missed_window'
        outcomes[reason] += 1
        label = cause_label(event['repairs'])
        groups.setdefault(label, Counter())[reason] += 1
        if choices and choices[0]['time_s']+30 <= row['duration_s']:
            branches.append(dict(time_s=choices[0]['time_s'], baseline_onset_s=onset, cause=label))
    return dict(onsets=len(events), alerts=len(alerts), hits=len(matched), reasons=outcomes,
                cause_groups=groups, branch_candidates=branches, expert_alerts=Counter(a['expert'] for a in alerts),
                usable_decisions=sum(t['expert'] is not None for t in row['advisory']),
                total_decisions=len(row['advisory']),
                trend_decisions=sum(t['expert']=='radio_trend' for t in row['advisory']))


def advisory_summary(rows):
    records = [advisory(row) for row in rows]
    events = sum(row['onsets'] for row in records)
    alerts = sum(row['alerts'] for row in records)
    hits = sum(row['hits'] for row in records)
    hours = sum(row['duration_s'] for row in rows)/3600
    groups = {}
    for row in records:
        for label, counts in row['cause_groups'].items():
            groups.setdefault(label, Counter()).update(counts)
    return dict(onsets=events, alerts=alerts, hits=hits, recall=hits/max(1, events),
                precision=hits/max(1, alerts), false_or_late_per_fleet_hour=(alerts-hits)/hours,
                reasons=sum((r['reasons'] for r in records), Counter()), by_cause=groups,
                trend_decisions=sum(r['trend_decisions'] for r in records),
                usable_decisions=sum(r['usable_decisions'] for r in records),
                total_decisions=sum(r['total_decisions'] for r in records),
                branch_candidates=sum(len(r['branch_candidates']) for r in records))


def branch_targets(rows):
    # A single preregistered eligible example per map and RF condition, at most 12.
    output = []
    for row in sorted(rows, key=lambda a: (a['spec']['kind'], a['spec']['region'])):
        options = advisory(row)['branch_candidates']
        if options:
            with np.load(row['archive_path'], allow_pickle=False) as old:
                healthy = old['network']
            options = [r for r in options if healthy[int(r['time_s']*2)-1]
                       and healthy[int(r['time_s']*2):int(r['baseline_onset_s']*2)].all()]
        if options:
            output.append(dict(options[0], baseline_path=row['archive_path'],
                               baseline_sha256=row['archive_sha256'], spec=row['spec']))
    balanced = []
    for kind in ('clear', 'intermittent', 'continuous'):
        balanced.extend([r for r in output if r['spec']['kind']==kind][:4])
    return balanced


def branch_result(target, row, data):
    end = int(target['time_s']*2)
    stop = end+60
    baseline = target['baseline_path']
    if sha(baseline) != target['baseline_sha256']:
        raise ValueError('Direct branch baseline changed')
    with np.load(baseline, allow_pickle=False) as old:
        for key in ('links', 'network', 'service'):
            np.testing.assert_array_equal(old[key][:end], data[key][:end])
        np.testing.assert_array_equal(old['commands'][:end//6], data['commands'][:end//6])
        np.testing.assert_array_equal(old['nodes'][:end//6+1], data['nodes'][:end//6+1])
        event, = row['intervention']
        base_cuts = old['network'][end:stop]
        alternate = data['network'][end:stop]
        if len(alternate)!=60 or len(base_cuts)!=60 or not old['network'][end-1]:
            raise ValueError('Branch lacks complete healthy shared prefix and 30s window')
        from forecast_prevention.diagnostics import outage_runs
        old_runs = outage_runs(base_cuts)
        new_runs = outage_runs(alternate)
        return dict(target, intervention=event, same_prefix=True, one_override=True,
                    baseline_runs_30s=[[s*.5, e*.5] for s, e in old_runs],
                    alternative_runs_30s=[[s*.5, e*.5] for s, e in new_runs],
                    alternative_connected_first_12s=bool(alternate[:24].all()),
                    baseline_down_seconds=float((~base_cuts).sum()*.5),
                    alternative_down_seconds=float((~alternate).sum()*.5))
