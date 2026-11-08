"""Fit both Transformer updates and confirm the Final V1 integration."""
import argparse
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from fast_halfsec.context import initialize, registry
from final_v1 import runner
from final_v1.evaluation import key, pooled
from final_v1.verify_delivery import audit_flight
from forecast_rebuild.contracts import now, read, sha, write

from . import ARTIFACTS
from . import alerts
from .bridge import MODES, run
from .collection import PLAN, cached, capture_job, collect, exclusive_run, gather, paths_for, prepare, save, status
from .model import load_heads
from .release import FREEZE, INTEGRATION, RESULT, fingerprints, load_frozen
from .training import FIT_FILE, fit, link_data, link_scores

FORECAST_REFERENCE = 'v7_common_trace'


def freeze_model(plan, development, payload):
    if FREEZE.exists():
        _, frozen = load_frozen()
        return frozen
    status('calibrating_fleet_alerts')
    timing, model = load_heads(payload)
    rows = [row for row in development['records'] if row['role'] == 'calibration']
    cases = list(alerts.cases(rows, timing, model))
    thresholds = alerts.calibrate(cases, plan['alert_cap_per_fleet_hour'])
    record = dict(frozen_at=now(), fingerprints=fingerprints(), thresholds=thresholds,
                  controls=list(MODES), forecast_reference=FORECAST_REFERENCE,
                  gate='Use the calibrated blend for the V21 advance score gate. Keep the native V18, physical and imminent-risk checks.',
                  acceptance=plan['promotion'])
    write(FREEZE, record)
    print("Fleet thresholds fixed before independent geography:", thresholds, flush=True)
    return record


@lru_cache(maxsize=1)
def worker_model(expected):
    return load_frozen(expected)


def mission_job(request):
    spec, mode, expected = request['spec'], request['mode'], request['freeze']
    if (spec not in read(PLAN)['cohorts']['confirmation'] or sha(FREEZE) != expected
            or registry()[spec['region']]['study_role'] != 'confirmation'):
        raise ValueError('Flight is outside frozen confirmation')
    if mode == FORECAST_REFERENCE:
        return capture_job(dict(role='confirmation', spec=spec, protocol=sha(PLAN), freeze=expected))
    paths = paths_for(f'missions/{mode}', spec)
    previous = cached(paths, request)
    if previous is not None:
        return previous
    if mode == 'v21':
        row, arrays = run(spec, mode)
    else:
        payload, profile = worker_model(expected)
        fleet = profile['thresholds']['fleet']
        row, arrays = run(spec, mode, payload, fleet['threshold'], network_weight=fleet['network_weight'])
    return save(paths, request, row, arrays)


def mission_jobs(jobs):
    return [mission_job(job) for job in jobs]


def compare_missions(reference, alternative):
    ref, alt = ({key(row): row for row in rows} for rows in (reference, alternative))
    if len(ref) != len(reference) or len(alt) != len(alternative) or ref.keys() != alt.keys():
        raise ValueError('Control arms are not paired on identical mission conditions')
    groups = defaultdict(lambda: np.zeros(4))
    per_case = []
    for case in sorted(ref):
        before, after = ref[case], alt[case]
        groups[registry()[case[0]]['study_group']] += [
            before['persistent_onsets'], before['duration_s'] / 3600.,
            after['persistent_onsets'], after['duration_s'] / 3600.,
        ]
        per_case.append(dict(spec=before['spec'], starts_before=before['persistent_onsets'],
                             starts_after=after['persistent_onsets'],
                             cut_s_before=before['persistent_cut_seconds'], cut_s_after=after['persistent_cut_seconds'],
                             complete_before=before['mission_completed'], complete_after=after['mission_completed'],
                             fleet_requests=sum(event.get('forecast_source') == 'fleet_transformer'
                                                for event in after['intervention'])))
    totals = np.stack(list(groups.values()))
    rng = np.random.default_rng(2026110813)
    draw = totals[rng.integers(len(totals), size=(5000, len(totals)))].sum(axis=1)
    limits = np.quantile(draw[:, 2] / draw[:, 3] - draw[:, 0] / draw[:, 1], [.025, .975])
    a, b = pooled(reference), pooled(alternative)
    return dict(groups=len(groups), ci95=limits.tolist(), starts_delta=b['starts'] - a['starts'],
                starts_per_h_delta=b['starts_per_h'] - a['starts_per_h'],
                cut_s_per_h_delta=b['cut_s_per_h'] - a['cut_s_per_h'],
                pdr_pp=100. * (b['timely_pdr'] - a['timely_pdr']),
                duration_fraction=b['hours'] / a['hours'] - 1.,
                completion_regressions=sum(row['complete_before'] and not row['complete_after'] for row in per_case),
                fleet_requests=sum(row['fleet_requests'] for row in per_case), cases=per_case)


def promotion(plan, forecast, missions, difference):
    rules = plan['promotion']
    checks = dict(no_safety_violations=missions['safety'] == 0,
                  completed_every_mission=missions['completed'] == missions['flights'],
                  pdr_ok=difference['pdr_pp'] >= -rules['maximum_pdr_loss_pp'],
                  duration_ok=difference['duration_fraction'] <= rules['maximum_mission_duration_fraction'],
                  cut_time_ok=difference['cut_s_per_h_delta'] <= 0.,
                  fewer_new_cuts_with_confidence=difference['ci95'][1] < 0.,
                  higher_early_recall=forecast['recall_delta_pp'] > 0.,
                  alert_budget_ok=forecast['modes']['fleet']['false_per_h'] <= plan['alert_cap_per_fleet_hour'])
    return dict(passed=all(checks.values()), checks=checks)


def assess(plan, development, payload, workers):
    frozen = freeze_model(plan, development, payload)
    expected = sha(FREEZE)
    if RESULT.exists():
        old = read(RESULT)
        if old['freeze_sha256'] != expected:
            raise ValueError('Stored confirmation used other weights')
        return old
    cohorts = plan['cohorts']['confirmation']
    common = [[dict(spec=spec, mode=FORECAST_REFERENCE, freeze=expected)] for spec in cohorts]
    traces = gather(common, mission_jobs, 'confirmation_forecast_traces', workers)
    timing, fleet = load_heads(payload)
    cases = list(alerts.cases(traces, timing, fleet))
    forecast = alerts.compare(cases, frozen['thresholds'])
    initialize()
    link_x, link_y = link_data(traces, torch.device('cpu'))
    link_forecast = dict(before=link_scores(runner.parent.MODEL.forecaster.timing, link_x, link_y),
                         after=link_scores(timing, link_x, link_y))
    write(ARTIFACTS / 'forecast_confirmation.json', dict(freeze_sha256=expected, links=link_forecast,
                                                       fleet=forecast))
    print("Confirmed forecasts:", forecast, flush=True)

    jobs = [[dict(spec=spec, mode=mode, freeze=expected) for mode in MODES] for spec in cohorts]
    mission_rows = gather(jobs, mission_jobs, 'confirmation_closed_loop', workers)
    status('auditing_confirmation', flights=len(traces) + len(mission_rows))
    rows_by_mode = {name: [row for row in mission_rows if row['upgrade_arm'] == name] for name in MODES}
    mission = {name: dict(pooled(rows), cut_seconds=sum(row['persistent_cut_seconds'] for row in rows))
               for name, rows in rows_by_mode.items()}
    comparisons = {mode: compare_missions(rows_by_mode['v21'], rows_by_mode[mode]) for mode in MODES[1:]}
    ledgers = {}
    frames = 0
    for row in traces + mission_rows:
        archive = Path(row['archive_path'])
        ledgers[str(archive)] = audit_flight(archive.with_suffix('.json'))
        frames += round(row['duration_s'] / .5)
    decision = promotion(plan, forecast, mission['v21_transformer'], comparisons['v21_transformer'])
    live = [event for row in rows_by_mode['v21_transformer'] for event in row['intervention']
            if event.get('forecast_source') == 'fleet_transformer']
    link = read(FIT_FILE)
    result = dict(completed_at=now(), freeze_sha256=expected, development=development['counts'],
                  training={name: {key: link[name][key] for key in ('selected_epoch', 'train_rows', 'device')}
                            for name in ('timing', 'network')},
                  link_forecast=link_forecast, forecast=forecast, calibration=frozen['thresholds'],
                  missions=mission, comparisons=comparisons, promotion=decision,
                  fleet_triggers=dict(total=len(live), connected=sum(e['actual_pre_action_connected'] for e in live),
                                      executed=sum(e['executed_as_requested'] for e in live),
                                      enabled_below_native_threshold=sum(e['enabled_by_fleet_score'] for e in live)),
                  audit=dict(flights=len(ledgers), rf_frames=frames, safety_and_packet_conservation=True,
                             flight_metadata_sha256=ledgers))
    write(RESULT, result)
    return result


def integrate(result):
    record = dict(mode='v21_transformer', default=result['promotion']['passed'],
                  installed_at=now(), freeze_sha256=result['freeze_sha256'], results_sha256=sha(RESULT))
    if INTEGRATION.exists():
        old = read(INTEGRATION)
        if {key: value for key, value in old.items() if key != 'installed_at'} != {
                key: value for key, value in record.items() if key != 'installed_at'}:
            raise ValueError('Installed extension differs from confirmation')
    else:
        write(INTEGRATION, record)
    status('completed', state='completed', default_controller=record['mode'] if record['default'] else 'v21',
           evaluated_flights=result['audit']['flights'])
    print("Final V1 upgrade verified:", result['promotion'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    with exclusive_run():
        try:
            status('verifying_inputs')
            plan = prepare()
            development = collect(plan, args.workers)
            result = assess(plan, development, fit(development), args.workers)
            integrate(result)
        except Exception as error:
            status('failed', state='failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
