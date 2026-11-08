"""Refit forecast components and verify the full trained stack."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import multiprocessing as mp

import numpy as np
import torch
from torch import nn

from fast_halfsec.context import registry
from final_v1 import control, runner
from final_v1.evaluation import pooled
from final_v1.transformer_upgrade import alerts, bridge, training as supervised
from final_v1.transformer_upgrade.campaign import compare_missions
from final_v1.transformer_upgrade.capture import COMMAND_STEP, collect_flight
from final_v1.transformer_upgrade.collection import cached, save
from final_v1.transformer_upgrade.model import FleetForecaster, fleet_inputs
from final_v1.verify_delivery import audit_flight
from forecast_learning import training as forest_training
from forecast_learning.features import COLUMNS
from forecast_rebuild.contracts import digest, now, read, sha, write

from . import ARTIFACTS
from .run import checkpoint_path, incumbent, worker_model
from .schedule import PLAN, status

SELECTION = ARTIFACTS / 'core_selection.json'
SUPERVISION = ARTIFACTS / 'supervision_manifest.json'
FLEET_FILE = ARTIFACTS / 'models/fleet.pt'
FIT = ARTIFACTS / 'supervised_models.json'
FREEZE = ARTIFACTS / 'freeze.json'
RESULT = ARTIFACTS / 'results.json'


def pointer(step):
    path = checkpoint_path(step)
    return dict(step=step, path=str(path), sha256=sha(path))


def core(record):
    return worker_model(record['path'], record['sha256'], 1.)


@lru_cache(maxsize=1)
def full_stack(expected):
    if sha(FREEZE) != expected:
        raise ValueError('The full model was not frozen with these weights')
    record = read(FREEZE)
    if record['protocol'] != sha(PLAN) or record['source'] != sha(__file__) or record['fit'] != sha(FIT):
        raise ValueError('Full model dependencies changed')
    fitted = read(FIT)
    if sha(FLEET_FILE) != fitted['fleet_sha256']:
        raise ValueError('Fleet model changed after training')
    network = FleetForecaster()
    network.load_state_dict(torch.load(FLEET_FILE, map_location='cpu', weights_only=True)['network'])
    early_meta, short_meta = (fitted['forests'][name] for name in ('early', 'short'))
    for meta in (early_meta, short_meta):
        if sha(meta['path']) != meta['sha256']:
            raise ValueError('A fitted V21 model changed')
    config = dict(bridge.load_profile()[1], dimension=len(COLUMNS), rounds=early_meta['rounds'])
    profile = ('learned', config, forest_training.restore_model(early_meta),
               forest_training.restore_model(short_meta))
    return network.eval(), profile, record


def control_flight(spec, checkpoint, expected=None):
    model = core(checkpoint)
    _, network, threshold, _ = incumbent()
    profile = None
    if expected:
        network, profile, record = full_stack(expected)
        threshold = record['operating']['candidate']
    payload = dict(timing=model.forecaster.timing.state_dict(), network=network.state_dict())
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner.parent, 'MODEL', model))
        if profile is not None:
            stack.enter_context(patch.object(bridge, 'load_profile', lambda: profile))
            stack.enter_context(patch.object(control, 'load_profile', lambda: profile))
        return bridge.run(spec, 'v21_transformer', payload, threshold['threshold'],
                          network_weight=threshold['network_weight'])


def trace_flight(spec, checkpoint, comparison=None):
    other = core(comparison) if comparison else None
    model = core(checkpoint)
    stock = model.policy
    alternative = []

    def trace_policy(b, mode):
        values, _ = other.forecaster.encode(b['values'], b['masks'], b['intents'], b['geometry'])
        alternative.append(values.detach().numpy())
        return stock(b, mode)

    with ExitStack() as stack:
        stack.enter_context(patch.object(runner.parent, 'MODEL', model))
        if other is not None:
            stack.enter_context(patch.object(model, 'policy', trace_policy))
        row, arrays = collect_flight(spec)
    if other is not None:
        arrays['other_embedding'] = np.asarray(alternative, np.float32)
        if arrays['other_embedding'].shape != arrays['embedding'].shape:
            raise ValueError('The two forecasts did not use the same link observations')
    return row, arrays


def job(request):
    phase, spec = request['phase'], request['spec']
    cohort = 'confirmation' if phase.startswith('confirmation') else phase
    plan = read(PLAN)
    expected_role = {'selection': 'calibration', 'supervise': 'train'}.get(cohort, cohort)
    if spec not in plan[cohort] or registry()[spec['region']]['study_role'] != expected_role:
        raise ValueError('The job is outside its frozen geography')
    if cohort == 'confirmation' and sha(FREEZE) != request['freeze']:
        raise ValueError('Confirmation was opened before model freeze')
    prefix = ARTIFACTS / phase / digest(request)[:20]
    paths = prefix.with_suffix('.json'), prefix.with_suffix('.npz')
    previous = cached(paths, request)
    if previous is not None:
        return previous
    checkpoint = request['checkpoint']
    if phase in ('selection', 'confirmation_control'):
        model = request.get('mode')
        row, arrays = control_flight(spec, checkpoint, request['freeze'] if model == 'candidate' else None)
    else:
        row, arrays = trace_flight(spec, checkpoint, request.get('comparison'))
    row.update(group=registry()[spec['region']]['study_group'], stage=phase)
    return save(paths, request, row, arrays)


def gather(requests, stage, workers):
    rows = []
    begun = perf_counter()
    status(stage, complete=0, total=len(requests))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        pending = [pool.submit(job, request) for request in requests]
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            elapsed = perf_counter() - begun
            status(stage, complete=len(rows), total=len(requests), wall_s=elapsed,
                   eta_s=elapsed * (len(requests) - len(rows)) / len(rows), last_map=row['spec']['region'])
            print(f"{stage} {len(rows)}/{len(requests)} {row['spec']['region']} "
                  f"step={row['contract']['checkpoint']['step']} cuts={row['persistent_onsets']}", flush=True)
    return sorted(rows, key=lambda row: (row['spec']['region'], row['spec']['kind'],
                                         row['contract']['checkpoint']['step']))


def choose_core(plan):
    steps = plan['checkpoints_for_selection']
    pointers = {step: pointer(step) for step in steps}
    if SELECTION.exists():
        record = read(SELECTION)
        if record['protocol'] != sha(PLAN) or record['chosen'] != pointers[record['chosen']['step']]:
            raise ValueError('Selected joint model changed')
        return record['chosen']
    jobs = [dict(phase='selection', spec=spec, checkpoint=pointers[step])
            for step in steps for spec in plan['selection']]
    rows = gather(jobs, 'selecting_joint_checkpoint', plan['config']['workers'])
    results = {step: pooled([row for row in rows if row['contract']['checkpoint']['step'] == step])
               for step in steps}
    reference, limits = results[0], plan['goals']

    def rank(step):
        result = results[step]
        eligible = (result['completed'] == result['flights'] and result['safety'] == 0
                    and result['hours'] / reference['hours'] <= 1. + limits['max_duration_increase']
                    and 100. * (reference['timely_pdr'] - result['timely_pdr']) <= limits['max_pdr_loss_pp'])
        return (eligible, -result['starts_per_h'], -result['cut_s_per_h'], result['timely_pdr'], -step)

    selected = max(steps[1:], key=rank)
    write(SELECTION, dict(protocol=sha(PLAN), chosen=pointers[selected], qualified=rank(selected)[0],
                          development_results=results))
    return pointers[selected]


def supervision(plan, chosen):
    if SUPERVISION.exists():
        record = read(SUPERVISION)
        if record['protocol'] != sha(PLAN) or record['chosen'] != chosen:
            raise ValueError('Supervision was collected for another joint model')
        return record
    jobs = [dict(phase='supervise', spec=spec, checkpoint=chosen) for spec in plan['supervise']]
    jobs.extend(dict(phase='calibration', spec=spec, checkpoint=chosen, comparison=pointer(0))
                for spec in plan['calibration'])
    rows = gather(jobs, 'collecting_full_model_supervision', plan['config']['workers'])
    record = dict(protocol=sha(PLAN), chosen=chosen,
                  train=[row for row in rows if row['stage'] == 'supervise'],
                  calibration=[row for row in rows if row['stage'] == 'calibration'])
    write(SUPERVISION, record)
    return record


def forest_data(rows):
    x, y_early, y_short = [], [], []
    for row in rows:
        with supervised.row_arrays(row) as archive:
            route = archive['route']
            times = np.arange(len(route)) * COMMAND_STEP
            support = np.isfinite(route[:, 0]) & (times + 7. <= row['duration_s']) & (times > 0.)
            times = times[support]
            connected = archive['network'][np.rint(times * 2).astype(int) - 1]
            events = np.asarray([start for start, _ in row['persistent_cut_intervals'] if start > 0.])
            bins = [np.searchsorted(events, times + dt) for dt in (.5, 3., 7.)]
            x.append(route[support, :len(COLUMNS)])
            y_early.append(connected & (bins[2] > bins[1]))
            y_short.append(connected & (bins[1] > bins[0]))
    return tuple(np.concatenate(values) for values in (x, y_early, y_short))


def fit_heads(record, chosen):
    pins = dict(plan=sha(PLAN), chosen=chosen, supervision=sha(SUPERVISION), source=sha(__file__))
    if FIT.exists():
        if read(FIT)['pins'] != pins:
            raise ValueError('Supervised model provenance changed')
        return
    state = core(chosen).forecaster.timing.state_dict()
    torch.manual_seed(2026110819)
    status('training_fleet_model')
    with patch.object(supervised, 'NETWORK_FILE', FLEET_FILE), patch.object(supervised, 'status', status):
        network = supervised.fit_network(record['train'], record['calibration'], state,
                                         torch.device('cuda'), pins)
    status('training_v21_early_and_imminent')
    x, early, short = forest_data(record['train'])
    if not early.any() or not short.any():
        raise ValueError('The full model has no positive forest training events')
    with patch.object(forest_training, 'ARTIFACTS', ARTIFACTS):
        _, early_model = forest_training.train_model('dual_early', x, early)
        _, short_model = forest_training.train_model('dual_short', x, short)
    write(FIT, dict(pins=pins, fleet_sha256=sha(FLEET_FILE), fleet_rows=network['train_rows'],
                    fleet_epoch=network['selected_epoch'], forest_rows=len(x),
                    forests=dict(early=early_model, short=short_model)))


@torch.no_grad()
def forecast_cases(rows, chosen, profile, network, candidate_trace):
    timing = core(chosen).forecaster.timing
    old_timing, old_fleet, old_cut, _ = incumbent()
    _, old_cfg, old_early, _ = bridge.load_profile()
    _, new_cfg, new_early, _ = profile
    for row in rows:
        with supervised.row_arrays(row) as archive:
            route = archive['route']
            times = np.arange(len(route)) * COMMAND_STEP
            mask = np.isfinite(route[:, 0]) & (times + 7. <= row['duration_s'])
            base, other = archive['embedding'][mask], archive['other_embedding'][mask]
            candidate, reference = (base, other) if candidate_trace else (other, base)
            support = archive['support'][mask]
            x, labels = route[mask, :len(COLUMNS)], archive['network_y']
            old_values = fleet_inputs(reference, support, route[mask], old_timing)
            new_values = fleet_inputs(candidate, support, route[mask], timing)
            p = {name: np.full(len(route), np.nan) for name in ('v21', 'network', 'forest')}
            if mask.any():
                old_v21 = forest_training.make_score(old_early, old_cfg, x, x[:, alerts.SHIELD])
                p['v21'][mask] = (old_cut['network_weight'] * old_fleet.early_score(old_values).numpy()
                                  + (1. - old_cut['network_weight']) * old_v21)
                p['network'][mask] = network.early_score(new_values).numpy()
                p['forest'][mask] = forest_training.make_score(new_early, new_cfg, x, x[:, alerts.SHIELD])
        yield dict(times=times, scores=p, label=labels,
                   onsets=np.asarray([s for s, _ in row['persistent_cut_intervals'] if s > 0.]),
                   hours=row['duration_s'] / 3600., group=row['group'])


def mix(cases, weight):
    return [dict(row, scores=dict(row['scores'], fleet=weight * row['scores']['network']
                                  + (1. - weight) * row['scores']['forest'])) for row in cases]


def freeze(plan, record, chosen):
    if FREEZE.exists():
        _, _, value = full_stack(sha(FREEZE))
        return value
    status('calibrating_full_model')
    fitted = read(FIT)
    early_meta, short_meta = (fitted['forests'][name] for name in ('early', 'short'))
    profile = ('learned', dict(bridge.load_profile()[1], dimension=len(COLUMNS), rounds=early_meta['rounds']),
               forest_training.restore_model(early_meta), forest_training.restore_model(short_meta))
    fleet = FleetForecaster()
    fleet.load_state_dict(torch.load(FLEET_FILE, map_location='cpu', weights_only=True)['network'])
    cases = list(forecast_cases(record['calibration'], chosen, profile, fleet.eval(), True))
    cap = plan['goals']['false_per_fleet_hour']
    incumbent_cut = alerts.calibrate_one(cases, 'v21', cap)
    options = [dict(network_weight=weight, **alerts.calibrate_one(mix(cases, weight), 'fleet', cap))
               for weight in alerts.FLEET_WEIGHTS]
    selected = max(options, key=lambda row: (row['hits'], row['precision'], row['average_precision']))
    result = dict(protocol=sha(PLAN), fit=sha(FIT), source=sha(__file__), chosen=chosen,
                  operating=dict(incumbent=incumbent_cut, candidate=selected, options=options),
                  frozen_at=now())
    write(FREEZE, result)
    return result


def confirm(plan, chosen):
    frozen = sha(FREEZE)
    old = pointer(0)
    jobs = [dict(phase='confirmation_forecast', spec=spec, checkpoint=old,
                 comparison=chosen, freeze=frozen) for spec in plan['confirmation']]
    jobs.extend(dict(phase='confirmation_control', spec=spec, checkpoint=checkpoint,
                     freeze=frozen, mode=mode) for spec in plan['confirmation']
                for mode, checkpoint in (('incumbent', old), ('candidate', chosen)))
    rows = gather(jobs, 'confirming_full_model', plan['config']['workers'])
    status('auditing_full_model', flights=len(rows))
    ledger = [audit_flight(Path(row['archive_path']).with_suffix('.json')) for row in rows]
    network, profile, freeze_record = full_stack(frozen)
    operational = freeze_record['operating']
    cases = list(forecast_cases([row for row in rows if row['stage'] == 'confirmation_forecast'],
                                chosen, profile, network, False))
    blended = mix(cases, operational['candidate']['network_weight'])
    forecasts = dict(incumbent=alerts.evaluate(cases, 'v21', operational['incumbent']['threshold'], True),
                     candidate=alerts.evaluate(blended, 'fleet', operational['candidate']['threshold'], True))
    control_rows = [[row for row in rows if row['contract'].get('mode') == mode]
                    for mode in ('incumbent', 'candidate')]
    before, after = (dict(pooled(part), cut_seconds=sum(row['persistent_cut_seconds'] for row in part))
                     for part in control_rows)
    diff = compare_missions(*control_rows)
    tests = dict(complete=after['completed'] == after['flights'], safety=after['safety'] == 0,
                 fewer_starts=diff['ci95'][1] < 0., cut_seconds=diff['cut_s_per_h_delta'] <= 0.,
                 pdr=diff['pdr_pp'] >= -plan['goals']['max_pdr_loss_pp'],
                 duration=diff['duration_fraction'] <= plan['goals']['max_duration_increase'],
                 higher_recall=forecasts['candidate']['recall'] > forecasts['incumbent']['recall'],
                 alert_budget=forecasts['candidate']['false_per_h'] <= plan['goals']['false_per_fleet_hour'])
    return dict(completed_at=now(), freeze_sha256=frozen, checkpoint=chosen, forecasts=forecasts,
                missions=dict(incumbent=before, candidate=after), paired=diff,
                acceptance=dict(passed=all(tests.values()), checks=tests),
                audit=dict(flights=len(ledger), verified=True,
                           rf_frames=sum(round(row['duration_s'] / .5) for row in rows)))


def finish(plan, history):
    if RESULT.exists():
        result = read(RESULT)
        if result['freeze_sha256'] != sha(FREEZE):
            raise ValueError('Completed results refer to another full model')
    else:
        chosen = choose_core(plan)
        record = supervision(plan, chosen)
        fit_heads(record, chosen)
        freeze(plan, record, chosen)
        result = confirm(plan, chosen)
        result['training'] = dict(updates=len(history), flights=len(plan['train']), maps=plan['unique_train_maps'])
        write(RESULT, result)
    status('completed', state='completed', total_updates=plan['config']['updates'],
           training_flights=len(plan['train']), selected_update=result['checkpoint']['step'],
           accepted=result['acceptance']['passed'])
    print(f"Full model training finished. Acceptance={result['acceptance']}", flush=True)
