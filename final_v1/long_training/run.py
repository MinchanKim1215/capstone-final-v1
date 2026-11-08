"""Run and resume a long background Final V1 training campaign."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from time import perf_counter
import ctypes
import multiprocessing as mp
import os

import numpy as np
import torch

from adaptive_joint_expanded.contracts import atomic_torch, lock, restore_rng, rng_state, seed_all
from adaptive_joint_expanded.learning import update
from adaptive_joint_expanded.model import optimizer
from fast_halfsec.context import initialize, registry
from final_v1 import runner
from final_v1.transformer_upgrade.bridge import run as verified_run
from final_v1.transformer_upgrade.collection import cached, save
from final_v1.transformer_upgrade.model import load_heads
from final_v1.transformer_upgrade.release import installed
from forecast_rebuild.contracts import now, read, sha, write

from . import ARTIFACTS
from .episode import TrainingController, episode
from .schedule import LATEST, PLAN, START, prepare, status


def checkpoint_path(step):
    return ARTIFACTS / 'checkpoints' / f'joint_{step:04d}.pt'


@lru_cache(maxsize=1)
def incumbent():
    payload, freeze = installed()
    timing, network = load_heads(payload)
    return timing, network, freeze['thresholds']['fleet'], payload


def initial_model(temperature):
    initialize()
    model = TrainingController(temperature)
    model.load_state_dict(runner.parent.MODEL.state_dict(), strict=True)
    model.forecaster.timing.load_state_dict(incumbent()[0].state_dict(), strict=True)
    return model


def verified_preflight(plan):
    check = ARTIFACTS / 'preflight.json'
    if check.exists():
        if read(check)['protocol_sha256'] != sha(PLAN):
            raise ValueError('Preflight belongs to another plan')
        return
    status('reproducing_incumbent')
    spec = plan['train'][0]
    model = initial_model(plan['config']['selector_temperature'])
    _, network, profile, payload = incumbent()
    r, a = verified_run(spec, 'v21_transformer', payload, profile['threshold'], max_steps=48,
                        network_weight=profile['network_weight'])
    t, b = episode(model, network, profile, spec, plan['config'], deterministic=True, max_steps=48)
    for name in ('commands', 'positions', 'links', 'network'):
        np.testing.assert_array_equal(a[name], b[name])
    if r['timely_pdr'] != t['timely_pdr'] or r['deployment'] != t['deployment']:
        raise ValueError('Long training does not reproduce the installed control')
    write(check, dict(protocol_sha256=sha(PLAN), checked_steps=48, spec=spec,
                      commands_positions_rf_and_packets_exact=True, audited=t['audited']))


def commit(model, opt, step, history):
    path = checkpoint_path(step)
    atomic_torch(path, dict(protocol_sha256=sha(PLAN), step=step, model=model.state_dict(),
                            optimizer=opt.state_dict(), rng=rng_state(), history=history))
    pointer = dict(path=str(path), sha256=sha(path), step=step, updated_at=now())
    write(LATEST, pointer)
    return pointer


def load_state(path, fingerprint):
    if sha(path) != fingerprint:
        raise ValueError('Checkpoint digest mismatch')
    item = torch.load(path, map_location='cpu', weights_only=False)
    if item['protocol_sha256'] != sha(PLAN):
        raise ValueError('Checkpoint belongs to another training plan')
    return item


@lru_cache(maxsize=1)
def worker_model(path, expected, temperature):
    initialize()
    model = TrainingController(temperature)
    model.load_state_dict(load_state(path, expected)['model'], strict=True)
    return model.eval()


def collect_one(request):
    spec, pointer, config = request['spec'], request['checkpoint'], request['config']
    if registry()[spec['region']]['study_role'] != 'train':
        raise ValueError('A training job tried to use a held-out map')
    prefix = ARTIFACTS / 'rollouts' / f"update_{request['update']:04d}" / f"{request['index']:02d}"
    paths = prefix.with_suffix('.json'), prefix.with_suffix('.npz')
    previous = cached(paths, request)
    if previous is not None:
        return previous
    model = worker_model(pointer['path'], pointer['sha256'], config['selector_temperature'])
    _, network, profile, _ = incumbent()
    row, arrays = episode(model, network, profile, spec, config)
    return save(paths, request, row, arrays)


def check_selector(model, episodes, device):
    maximum = 0.
    with torch.no_grad():
        for arrays, row in episodes:
            logits = model.select(torch.as_tensor(arrays['public_context'], device=device),
                                  torch.as_tensor(arrays['deployment_mask'], device=device))
            actual = torch.distributions.Categorical(logits=logits).log_prob(
                torch.as_tensor(arrays['selector_choice'], device=device))
            maximum = max(maximum, abs(float(actual) - float(arrays['selector_logp'])))
    if not np.isfinite(maximum) or maximum > 3e-4:
        raise ValueError('Formation selection log probability did not replay')
    return maximum


def train_core(plan):
    config = plan['config']
    if not torch.cuda.is_available():
        raise RuntimeError('This campaign requires the available CUDA GPU')
    device = torch.device(config['device'])
    torch.set_num_threads(1)
    verified_preflight(plan)
    seed_all(config['seed'])
    model = initial_model(config['selector_temperature']).to(device)
    opt = optimizer(model, config)
    if LATEST.exists():
        pointer = read(LATEST)
        checkpoint = load_state(pointer['path'], pointer['sha256'])
        if checkpoint['step'] != pointer['step']:
            raise ValueError('Committed checkpoint step differs from its pointer')
        model.load_state_dict(checkpoint['model'], strict=True)
        opt.load_state_dict(checkpoint['optimizer'])
        restore_rng(checkpoint['rng'])
        history = checkpoint['history']
    else:
        history = []
        pointer = commit(model, opt, 0, history)
    begun = perf_counter()
    width = config['batch_episodes']
    completed = pointer['step']
    if completed == config['updates']:
        return history
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=mp.get_context('spawn')) as pool:
        for step in range(completed + 1, config['updates'] + 1):
            batch = plan['train'][(step - 1) * width:step * width]
            jobs = [dict(spec=spec, checkpoint=pointer, config=config, update=step, index=index)
                    for index, spec in enumerate(batch)]
            status('collecting_on_policy', update=step, total_updates=config['updates'],
                   training_flights=(step - 1) * width, batch_done=0, batch_size=width,
                   total_flights=len(plan['train']), checkpoint=pointer['path'])
            rows = []
            for future in as_completed([pool.submit(collect_one, job) for job in jobs]):
                row = future.result()
                rows.append(row)
                status('collecting_on_policy', update=step, total_updates=config['updates'],
                       training_flights=(step - 1) * width + len(rows), batch_done=len(rows),
                       batch_size=width, total_flights=len(plan['train']), checkpoint=pointer['path'],
                       last_map=row['spec']['region'])
                print(f"update {step}/{config['updates']} flight {len(rows)}/{width} "
                      f"{row['spec']['region']} {row['deployment']} cuts={row['persistent_onsets']}", flush=True)
            episodes = []
            for row in sorted(rows, key=lambda item: item['contract']['index']):
                with np.load(row['archive_path'], allow_pickle=False) as archive:
                    episodes.append(({name: archive[name] for name in archive.files}, row))
            status('gpu_joint_update', update=step, total_updates=config['updates'],
                   training_flights=step * width, total_flights=len(plan['train']), device=str(device))
            selector_error = check_selector(model, episodes, device)
            stats = update(model, opt, episodes, config, step, device)
            stats.update(selector_logp_max_error=selector_error,
                         deployments=[row['deployment'] for row in rows],
                         completed=sum(row['mission_completed'] for row in rows),
                         onsets=sum(row['persistent_onsets'] for row in rows),
                         guard_overrides=sum(row['overrides'] for row in rows),
                         wall_s=perf_counter() - begun)
            history.append(stats)
            pointer = commit(model, opt, step, history)
            write(ARTIFACTS / 'history.json', history)
            status('joint_checkpoint_saved', update=step, total_updates=config['updates'],
                   training_flights=step * width, total_flights=len(plan['train']), checkpoint=pointer['path'])
            print(f"checkpoint {step} saved. encoder={stats['weight_change_l2']['forecaster.backbone']:.6g} "
                  f"timing={stats['weight_change_l2']['forecaster.timing']:.6g} "
                  f"actor={stats['weight_change_l2']['actor']:.6g} "
                  f"selector={stats['weight_change_l2']['selector']:.6g}", flush=True)
            del episodes
    return history


def main():
    with lock(ARTIFACTS / 'run.lock'):
        keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
        held = bool(keep_awake(0x80000001))
        try:
            if not START.exists():
                write(START, dict(started_at=now(), pid=os.getpid(), system_sleep_inhibited=held))
            status('verifying_inputs')
            plan = prepare()
            history = train_core(plan)
            from .finish import finish
            finish(plan, history)
        except Exception as error:
            status('failed', state='failed', error=repr(error))
            raise
        finally:
            keep_awake(0x80000000)


if __name__ == '__main__':
    main()
