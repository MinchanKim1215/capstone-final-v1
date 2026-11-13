"""Run, resume, and independently evaluate the half-second joint campaign."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from functools import lru_cache
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import torch

from adaptive_joint_expanded.contracts import lock
from fast_halfsec.context import registry
from fast_halfsec.evaluation import audit, paired, pooled
from fast_halfsec.ppo import write_checkpoint
from final_v1.transformer_upgrade.alerts import calibrate_one, evaluate
from final_v1.transformer_upgrade.capture import label_network
from final_v1.transformer_upgrade.release import installed
from forecast_rebuild.contracts import digest, now, read, sha, write

from . import ARTIFACTS
from .mission import JointFlight
from .model import JointPolicy, initial_model
from .protocol import ARMS, PLAN, SEED, prepare, status
from .training import optimize, optimizer_for


LATEST = ARTIFACTS / 'latest.json'
FREEZE = ARTIFACTS / 'freeze.json'
RESULT = ARTIFACTS / 'results.json'


def retry_permission(operation, *args):
    for attempt in range(6):
        try:
            return operation(*args)
        except PermissionError:
            if attempt == 5:
                raise
            sleep(.2 * (attempt + 1))


def checkpoint(step):
    return ARTIFACTS / 'checkpoints' / f'joint_{step:04d}.pt'


def pointer(step):
    path = checkpoint(step)
    return dict(step=step, path=str(path), sha256=sha(path))


def load_state(record):
    if sha(record['path']) != record['sha256']:
        raise ValueError('The checkpoint contents changed')
    result = torch.load(record['path'], map_location='cpu', weights_only=True)
    if result['step'] != record['step'] or result['protocol'] != sha(PLAN):
        raise ValueError('The checkpoint does not belong to this campaign')
    return result


@lru_cache(maxsize=2)
def worker_model(step, path, sha256):
    torch.set_num_threads(1)
    result = load_state(dict(step=step, path=path, sha256=sha256))
    model = JointPolicy()
    model.load_state_dict(result['model'])
    return model.eval(), result['threshold']


@lru_cache(maxsize=1)
def original_payload():
    return installed()


def original_flight(spec):
    from final_v1.transformer_upgrade.bridge import run

    payload, frozen = original_payload()
    profile = frozen['thresholds']['fleet']
    row, arrays = run(spec, 'v21_transformer', payload, profile['threshold'],
                      network_weight=profile['network_weight'])
    result = dict(spec=spec, mode='final_v1_3s', deployment=row['deployment'],
                  duration_s=row['duration_s'], mission_completed=row['mission_completed'],
                  onsets=row['persistent_onsets'], cuts=row['persistent_cut_intervals'],
                  cut_seconds=row['persistent_cut_seconds'], network_fraction=row['search_connected_mean'],
                  generated=row['generated_packets'], timely=round(row['timely_pdr'] * row['generated_packets']),
                  timely_pdr=row['timely_pdr'], delivered=row['delivered_packets'],
                  relay_travel_m=row['relay_travel_m'], search_travel_m=row['search_travel_m'],
                  safety_passed=bool(row['audited']['passed']), packet_conserved=bool(row['audited']['packet_conserved']),
                  coverage=row['observable_coverage'], min_separation_m=row['audited']['minimum_swept_separation_m'],
                  planner_p95_ms=row['simulator_step_p95_ms'], wall_s=row['wall_s'],
                  prevention_requests=sum(event.get('actual_pre_action_connected', False)
                                          for event in row['intervention']), events=row['intervention'])
    return result, {name: arrays[name] for name in ('network', 'links', 'positions', 'commands', 'requested')}


def job(stage, spec, arm, model=None, other=None, explore=False):
    return dict(stage=stage, spec=spec, arm=arm, model=model, comparison=other,
                explore=explore, protocol=sha(PLAN))


def paths(request):
    folder = ARTIFACTS / 'flights' / request['stage']
    prefix = folder / digest(request)[:24]
    return prefix.with_suffix('.json'), prefix.with_suffix('.npz')


def cached(request):
    meta, archive = paths(request)
    if not meta.exists() or not archive.exists():
        return None
    result = read(meta)
    if result['contract'] != request or sha(archive) != result['archive_sha256']:
        raise ValueError(f'Rollout no longer matches the frozen request: {meta.name}')
    return result


def one(request):
    previous = cached(request)
    if previous is not None:
        return previous
    stage, spec, arm = request['stage'], request['spec'], request['arm']
    role = 'train' if stage.startswith('train_') else ('calibration' if stage in ('selection', 'calibration')
                                                       else 'confirmation')
    if registry()[spec['region']]['study_role'] != role or request['protocol'] != sha(PLAN):
        raise ValueError('Invalid flight role or frozen input')
    if role == 'confirmation' and not FREEZE.is_file():
        raise ValueError('Confirmation is sealed until weights and thresholds are fixed')
    if arm == 'native_3s':
        row, arrays = retry_permission(original_flight, spec)
    else:
        model, threshold = worker_model(**request['model'])
        comparison = worker_model(**request['comparison'])[0] if request['comparison'] else None
        cadence = 3. if arm == 'warm_3s' else .5
        # The simulator can encounter an unrelated OneDrive reader of a map cache.
        # A retry reconstructs the entire flight with the same immutable request.
        row, arrays = retry_permission(run_joint, spec, model, threshold, request['explore'], comparison, cadence)
    meta, archive = paths(request)
    archive.parent.mkdir(parents=True, exist_ok=True)
    pending = archive.with_suffix('.pending.npz')
    np.savez_compressed(pending, **arrays)
    retry_permission(os.replace, pending, archive)
    row.update(contract=request, arm=arm, archive_path=str(archive), archive_sha256=sha(archive))
    write(meta, row)
    print(f"{stage} {arm} {spec['region']} {spec['kind']} {row['wall_s']:.1f}s wall, "
          f"cuts={row['onsets']}, complete={row['mission_completed']}, "
          f"PPO decisions={row.get('joint_actor_decisions', 0)}", flush=True)
    return row


def run_joint(spec, model, threshold, explore, other, cadence):
    from swarm_sim.mission import MAX_MISSION
    return JointFlight(model, threshold, explore, other).run(spec, cadence, MAX_MISSION)


def run_map(requests):
    return [one(request) for request in requests]


def preflight(plan):
    record = ARTIFACTS / 'preflight.json'
    protocol = sha(PLAN)
    if record.exists():
        evidence = read(record)
        if evidence.get('protocol') == protocol:
            return
    status('preflight_halfsec_joint')
    torch.manual_seed(SEED)
    model, threshold = initial_model()
    checks = (
        dict(region='bundang', direction='east_west', seed=1199569006, kind='clear'),
        dict(region='gyeonggi_fast_20261027_03', direction='east_west', seed=1710826027, kind='clear'),
    )
    folder = ARTIFACTS / 'preflight'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, spec in enumerate(checks):
        if registry()[spec['region']]['study_role'] != 'train':
            raise ValueError('Preflight can only use training geography')
        row, arrays = JointFlight(model, threshold, True, None).run(spec, .5, 360.)
        if row['plan_ticks'] != 720 or not row['transformer_evaluations']:
            raise ValueError('The causal Transformer did not run on the half-second clock')
        path = folder / f'flight_{index}.npz'
        np.savez_compressed(path, **arrays)
        rows.append(dict(row, archive_path=str(path), archive_sha256=sha(path)))
    model.to('cuda')
    stats = optimize(model, optimizer_for(model), rows, torch.device('cuda'), 1, SEED)
    if not stats['decisions'] or not stats.get('ppo_gradient_to_timing'):
        raise ValueError('The combined simulation and GPU gradient did not pass')
    fields = ('spec', 'plan_ticks', 'transformer_evaluations', 'joint_actor_decisions', 'planner_p95_ms',
              'safety_passed', 'packet_conserved', 'archive_sha256')
    write(record, dict(protocol=protocol, flights=[{key: row[key] for key in fields} for row in rows],
                       gpu_update=stats))
    print(f'Half-second Transformer/PPO preflight passed: {stats}', flush=True)


def collect(pool, requests, stage, **progress):
    rows, by_map = [], defaultdict(list)
    for request in requests:
        row = cached(request)
        if row is None:
            by_map[request['spec']['region']].append(request)
        else:
            rows.append(row)
    begun = perf_counter()
    initial = len(rows)
    status(stage, complete=initial, total=len(requests), **progress)
    pending = [pool.submit(run_map, group) for group in by_map.values()]
    for task in as_completed(pending):
        rows.extend(task.result())
        elapsed = perf_counter() - begun
        status(stage, complete=len(rows), total=len(requests), wall_s=elapsed,
               stage_eta_s=elapsed * (len(requests) - len(rows)) / max(1, len(rows) - initial),
               **progress)
    if len(rows) != len(requests):
        raise ValueError('A whole-flight cohort is incomplete')
    return sorted(rows, key=lambda row: (row['spec']['region'], row['spec']['kind'], row['arm']))


def train(plan, pool):
    device = torch.device('cuda')
    torch.manual_seed(SEED)
    model, threshold = initial_model()
    model.to(device)
    optimizer = optimizer_for(model)
    if not checkpoint(0).exists():
        retry_permission(write_checkpoint, checkpoint(0),
                         dict(step=0, protocol=sha(PLAN), model=model.state_dict(), threshold=threshold))
    prior = read(LATEST) if LATEST.exists() else pointer(0)
    state = load_state(prior)
    model.load_state_dict(state['model'])
    threshold = state['threshold']
    if prior['step']:
        optimizer.load_state_dict(state['optimizer'])
    write(LATEST, prior)
    batch = plan['batch']
    for step in range(prior['step'] + 1, plan['updates'] + 1):
        stage = f'train_{step:03d}'
        specs = plan['train'][(step - 1) * batch:step * batch]
        binding = dict(previous=prior, specs=specs)
        current = checkpoint(step)
        if current.exists():
            result = load_state(pointer(step))
            if result['binding'] != binding:
                raise ValueError('An existing checkpoint has another training batch')
            model.load_state_dict(result['model'])
            optimizer.load_state_dict(result['optimizer'])
        else:
            fields = dict(update=step, total_updates=plan['updates'],
                          earlier_training_flights=(step - 1) * batch, training_flights_total=len(plan['train']))
            rows = collect(pool, [job(stage, spec, 'joint_05s', prior, explore=True) for spec in specs],
                           'collecting_halfsec_flights', **fields)
            status('gpu_joint_update', **fields, complete=len(rows), total=batch)
            stats = optimize(model, optimizer, rows, device, step, SEED)
            stats.update(flights=len(rows), completed=sum(row['mission_completed'] for row in rows),
                         new_cuts=sum(row['onsets'] for row in rows),
                         transformer_evaluations=sum(row['transformer_evaluations'] for row in rows),
                         planner_p95_median_ms=float(np.median([row['planner_p95_ms'] for row in rows])))
            result = dict(step=step, protocol=sha(PLAN), model=model.state_dict(), optimizer=optimizer.state_dict(),
                          binding=binding, threshold=threshold, stats=stats)
            retry_permission(write_checkpoint, current, result)
        prior = pointer(step)
        write(LATEST, prior)
        write(ARTIFACTS / 'updates' / f'{step:03d}.json', result['stats'])
        status('joint_checkpoint_saved', update=step, total_updates=plan['updates'],
               training_flights_completed=step * batch, training_flights_total=len(plan['train']),
               checkpoint=prior, **result['stats'])
        print(f"GPU joint update {step}/{plan['updates']}: {result['stats']}", flush=True)


def forecast_cases(rows):
    cases = []
    for row in rows:
        with np.load(row['archive_path'], allow_pickle=False) as arrays:
            n = len(arrays['network'])
            frame = arrays['plan_frames']
            valid = frame + 14 < n
            score = {mode: np.full(n, np.nan) for mode in ('warm', 'joint')}
            score['warm'][frame[valid]] = arrays['scores'][valid]
            score['joint'][frame[valid]] = arrays['comparison_scores'][valid]
            times = np.arange(n) * .5
            label = label_network(arrays['network'], row['cuts'], times)
        cases.append(dict(times=times, scores=score, label=label,
                          onsets=np.asarray([start for start, _ in row['cuts'] if start > 0.]),
                          hours=row['duration_s'] / 3600.))
    return cases


def freeze(plan, pool):
    if FREEZE.exists():
        record = read(FREEZE)
        if record['protocol'] != sha(PLAN) or record['selected'] != pointer(record['selected']['step']):
            raise ValueError('Frozen model or study changed')
        return record
    results = {}
    for step in plan['selection_steps']:
        requests = [job('selection', spec, 'joint_05s', pointer(step)) for spec in plan['selection']]
        rows = collect(pool, requests, 'selecting_checkpoint', checkpoint_step=step)
        results[step] = pooled(rows)
    base = results[0]
    budget = plan['objectives']
    eligible = [step for step in results if step and results[step]['completed'] >= base['completed']
                and results[step]['timely_pdr'] >= base['timely_pdr'] - budget['max_pdr_loss_pp'] / 100.
                and results[step]['mean_mission_s'] <= base['mean_mission_s'] *
                (1. + budget['max_mission_time_increase'])]
    candidates = eligible or [step for step in results if step]
    chosen = min(candidates, key=lambda step: (-results[step]['completed'], results[step]['starts_per_h'],
                                               results[step]['cut_seconds_per_h']))
    initial, selected = pointer(0), pointer(chosen)
    requests = [job('calibration', spec, 'warm_05s', initial, selected) for spec in plan['calibration']]
    rows = collect(pool, requests, 'calibrating_on_common_observations', checkpoint_step=chosen)
    cases = forecast_cases(rows)
    alerts = {name: calibrate_one(cases, name, budget['alert_budget_per_fleet_h']) for name in ('warm', 'joint')}
    result = dict(protocol=sha(PLAN), initial=initial, selected=selected, selection=results,
                  selection_qualified=bool(eligible), operational_threshold=load_state(initial)['threshold'],
                  calibration=alerts, frozen_at=now())
    write(FREEZE, result)
    return result


def confirm(plan, frozen, pool):
    initial, selected = frozen['initial'], frozen['selected']

    def request(stage, spec, arm):
        if arm == 'native_3s':
            return job(stage, spec, arm)
        if arm == 'joint_05s':
            return job(stage, spec, arm, selected)
        return job(stage, spec, arm, initial, selected if arm == 'warm_05s' else None)

    rows = collect(pool, [request('confirmation', spec, arm) for spec in plan['confirmation'] for arm in ARMS],
                   'confirming_same_conditions', checkpoint_step=selected['step'])
    stress = collect(pool, [request('stress', plan['navigation_stress'], arm) for arm in ARMS],
                     'checking_known_navigation_stall')
    status('auditing_flights', complete=0, total=len(rows) + len(stress))
    ledger = audit(rows + stress)
    groups = {arm: [row for row in rows if row['arm'] == arm] for arm in ARMS}
    missions = {arm: pooled(records) for arm, records in groups.items()}
    deltas = {name: paired(groups[ref], groups[alt]) for name, ref, alt in
              (('cadence', 'warm_3s', 'warm_05s'), ('joint_learning', 'warm_05s', 'joint_05s'),
               ('whole_system', 'native_3s', 'joint_05s'))}
    common = forecast_cases(groups['warm_05s'])
    operating = {name: evaluate(common, name, frozen['operational_threshold'], True) for name in ('warm', 'joint')}
    matched = {name: evaluate(common, name, frozen['calibration'][name]['threshold'], True)
               for name in ('warm', 'joint')}
    goals = plan['objectives']
    final = missions['joint_05s']
    full, within = deltas['whole_system'], deltas['joint_learning']
    checks = dict(safe=final['violations'] == 0, every_mission_completed=final['completed'] == final['flights'],
                  prior_stall_completed=next(row['mission_completed'] for row in stress if row['arm'] == 'joint_05s'),
                  cut_rate_improves=full['ci95'][1] < 0.,
                  cut_time_improves=final['cut_seconds_per_h'] <= missions['native_3s']['cut_seconds_per_h'],
                  gains_from_joint_learning=within['ci95'][1] < 0.,
                  pdr=full['timely_pdr_pp'] >= -goals['max_pdr_loss_pp'],
                  mission_time=full['duration_fraction'] <= goals['max_mission_time_increase'],
                  lead_recall=operating['joint']['recall'] >= operating['warm']['recall'],
                  alert_budget=operating['joint']['false_per_h'] <= goals['alert_budget_per_fleet_h'])
    return dict(completed_at=now(), frozen_sha256=sha(FREEZE), selected=selected,
                training=dict(updates=plan['updates'], flights=len(plan['train']), maps=plan['train_maps']),
                control=missions, paired=deltas, forecast_operating=operating, forecast_matched=matched,
                confirmation_maps=dict(new=plan['fresh_confirmation_maps'], prior=plan['prior_confirmation_maps']),
                stress={row['arm']: {key: row[key] for key in
                        ('mission_completed', 'coverage', 'duration_s', 'onsets', 'cut_seconds')} for row in stress},
                audit=ledger, acceptance=dict(passed=all(checks.values()), checks=checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error('--workers must be between 1 and 12')
    with lock(ARTIFACTS / 'run.lock'):
        keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
        keep_awake(0x80000001)
        try:
            status('validating_inputs')
            if not torch.cuda.is_available():
                raise RuntimeError('The full campaign requires the available CUDA GPU')
            plan = prepare()
            if RESULT.exists():
                result = read(RESULT)
                if result['frozen_sha256'] != sha(FREEZE):
                    raise ValueError('Evaluation does not match the frozen candidate')
            else:
                preflight(plan)
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
                    train(plan, pool)
                    result = confirm(plan, freeze(plan, pool), pool)
                write(RESULT, result)
            status('completed', state='completed', accepted=result['acceptance']['passed'],
                   training_flights_completed=len(plan['train']), total_updates=plan['updates'],
                   selected_update=result['selected']['step'])
            print(f"Finished half-second joint training and evaluation. {result['acceptance']}", flush=True)
        except Exception as error:
            status('failed', state='failed', error=repr(error))
            raise
        finally:
            keep_awake(0x80000000)


if __name__ == '__main__':
    main()
