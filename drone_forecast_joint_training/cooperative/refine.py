"""중단 이후의 독립 학습·공통 평가. 기존 최종 결과를 덮어쓰지 않는 실행기.

각 정책은 동일한 임무 환경에서 완주 상호작용으로 학습한다. 시험의 새
시나리오와 정책 선정 기준은 첫 학습 전에 동결하고, 시험 후 재선정하지 않는다.
"""

import argparse
import hashlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from cooperative.analyze import summarize
from cooperative.io import validated_result
from cooperative.locking import TrainingLock
from cooperative.policy import ActorCritic
from cooperative.run import episode
from cooperative.train import save, update
from swarm_sim.regions import ROOT, sha256_file

STUDY = ROOT / 'results/refinement'
MODELS = ROOT / 'model/refinement'
PLAN = ROOT / 'configs/refinement_plan.json'
VERSION = 'closed-loop-refinement-v2'
SOURCE_FILES = (
    'cooperative/train.py', 'cooperative/policy.py', 'cooperative/run.py',
    'cooperative/refine.py', 'cooperative/candidates.py', 'cooperative/world.py',
    'cooperative/prediction.py', 'cooperative/motion.py', 'cooperative/packets.py',
    'swarm_sim/radio.py',
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def fingerprint():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol():
    original = json.loads((ROOT / 'configs/cooperative_evaluation.json').read_text(encoding='utf-8'))
    specs = []
    for item in original['test']:
        token = f'{VERSION}:{item["region"]}:{item["direction"]}:{item["kind"]}:{item["seed"]}'
        specs.append(dict(item, seed=int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % (2**31 - 3000004)))
    result = {
        'format': VERSION, 'training_seeds': [27101, 27102, 27103],
        'training_iterations': 8, 'episodes_per_iteration': 8,
        'training_modes': ['risk', 'no_risk'],
        'forecast': 'frozen LightGBM; no predictor backpropagation',
        'no_risk_definition': 'remove learned risk and onset from actor AND critic; disable forecaster; same beacon cost and geometric motion extrapolation',
        'guard_regret': 0.35, 'validation': original['validation'], 'test': specs,
        'selection_constraints': original['selection_constraints'],
        'selection_order': original['selection_order'],
        'test_regions_previously_evaluated_in_v1': True,
        'test_conditions': 'new deterministic seeds on the existing eight held-out regions; not eight newly collected cities',
        'test_used_for_v2_selection': False,
        'baseline_results_must_be_fresh_or_explicitly_labeled_historical': True,
        'source_sha256': fingerprint(),
        'forecaster_sha256': sha256_file(ROOT / 'model/advance/selected/best.pt'),
    }
    if PLAN.exists():
        existing = json.loads(PLAN.read_text(encoding='utf-8'))
        if existing != result:
            raise RuntimeError('동결한 v2 실험 계획·코드와 다름. 새 연구 이름이 필요합니다.')
    else:
        atomic_json(PLAN, result)
    return result


def rollout_job(item):
    spec, controller, checkpoint, prefix, sources, sampling_seed = item
    torch.set_num_threads(1)
    if sampling_seed is not None:
        torch.manual_seed(int(sampling_seed))
    checkpoint_sha = sha256_file(checkpoint) if checkpoint else None
    previous = validated_result(prefix, spec, checkpoint_sha, rollout=controller.startswith('ppo_sample'))
    if previous is not None and previous.get('refinement_sources') == sources and previous.get('controller') == controller:
        return previous
    result = episode(spec, controller, checkpoint, prefix, record=controller.startswith('ppo_sample'))
    result['refinement_sources'] = sources
    atomic_json(Path(prefix).with_suffix('.json'), result)
    return result


def collect(jobs, workers, pool=None):
    results = []
    owned = pool is None
    if owned:
        pool = ProcessPoolExecutor(max_workers=workers)
    try:
        for future in as_completed([pool.submit(rollout_job, job) for job in jobs]):
            result = future.result()
            results.append(result)
            print(json.dumps({
                'stage': 'episode', 'completed': len(results), 'total': len(jobs),
                'region': result['region'], 'controller': result['controller'],
                'connection': result['search_connected_mean'], 'pdr': result['pdr'],
                'complete': result['mission_completed'], 'wall_s': round(result['wall_s'], 2),
            }), flush=True)
    finally:
        if owned:
            pool.shutdown(wait=True, cancel_futures=True)
    return results


def train(args):
    plan = protocol()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    name = f'{args.mode}_s{args.seed}'
    folder = MODELS / name
    folder.mkdir(parents=True, exist_ok=True)
    history = []
    net = ActorCritic()
    optim = torch.optim.Adam(net.parameters(), lr=7e-4, eps=1e-5)
    start = 0
    config = dict(use_risk=args.mode == 'risk', refinement_version=VERSION,
                  training_sources=plan['source_sha256'], batch_episodes=args.batch_episodes)
    lock = TrainingLock(folder / 'training.lock')
    pool = None
    try:
        last = folder / 'last.pt'
        if last.exists():
            if not args.resume:
                raise ValueError('이미 학습한 독립 실행. --resume으로 이어서 실행하세요.')
            ck = torch.load(last, map_location='cpu', weights_only=False)
            if ck['seed'] != args.seed or any(ck.get(k) != v for k, v in config.items()):
                raise ValueError('재개 설정·학습 코드·난수 또는 위험 입력 계약 불일치')
            net.load_state_dict(ck['model']); optim.load_state_dict(ck['optimizer'])
            history = ck['history']; start = ck['iteration']
            torch.set_rng_state(ck['torch_rng']); np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
        else:
            save(folder / 'initial.pt', net, optim, 0, [], args.seed, **config)
        maps = json.loads((ROOT / 'configs/usable_protocol.json').read_text(encoding='utf-8'))['maps']
        train_keys = sorted(e['key'] for e in maps if e['split'] == 'train')
        schedule = np.random.default_rng(args.seed).permutation(train_keys).tolist()
        started = time.perf_counter()
        pool = ProcessPoolExecutor(max_workers=args.workers)
        for iteration in range(start, args.iterations):
            ckpath = folder / f'iteration_{iteration:02d}.pt'
            if ckpath.exists():
                existing = torch.load(ckpath, map_location='cpu', weights_only=False)
                if any(not torch.equal(value, existing['model'][key]) for key, value in net.state_dict().items()):
                    raise RuntimeError('기존 수집 체크포인트와 현재 정책 불일치')
            else:
                save(ckpath, net, optim, iteration, history, args.seed, **config)
            jobs, paths = [], []
            for j in range(args.batch_episodes):
                k = iteration * args.batch_episodes + j
                cycle = k // len(schedule)
                spec = dict(region=schedule[k % len(schedule)], seed=args.seed + 10000 + k * 101,
                            kind=('continuous', 'intermittent', 'clear')[(k + cycle) % 3],
                            direction=('east_west', 'north_south')[(k // 3 + cycle) % 2])
                prefix = STUDY / 'training' / name / f'i{iteration:02d}_e{j:02d}'
                controller = 'ppo_sample' if config['use_risk'] else 'ppo_sample_no_risk'
                jobs.append((spec, controller, str(ckpath), str(prefix), plan['source_sha256'], args.seed * 100000 + k))
                paths.append(prefix.with_suffix('.npz'))
            results = collect(jobs, args.workers, pool=pool)
            for group in optim.param_groups:
                group['lr'] = 3e-4 if iteration == 0 else 7e-4
            stats = update(net, optim, paths, args.seed + iteration, epochs=6, target_kl=0.02)
            stats.update(iteration=iteration + 1, episodes=len(results),
                         connection_mean=float(np.mean([r['search_connected_mean'] for r in results])),
                         pdr_mean=float(np.mean([r['pdr'] for r in results])),
                         complete_count=sum(r['mission_completed'] for r in results),
                         elapsed_s=time.perf_counter() - started)
            history.append(stats)
            save(last, net, optim, iteration + 1, history, args.seed, **config)
            save(folder / f'iteration_{iteration+1:02d}.pt', net, optim, iteration + 1, history, args.seed, **config)
            atomic_json(STUDY / 'training' / name / 'history.json', history)
            print(json.dumps({'stage': 'update', 'run': name, **stats}), flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        lock.close()


def evaluate(args):
    plan = protocol()
    checkpoint = str(Path(args.checkpoint).resolve()) if args.checkpoint else None
    if args.controller.startswith('ppo') and checkpoint is None:
        raise ValueError('PPO 평가에는 체크포인트가 필요합니다.')
    specs = plan[args.split][:args.limit]
    jobs = []
    for spec in specs:
        name = f'{spec["region"]}_{spec["direction"]}_{spec["kind"]}_{spec["seed"]}'
        prefix = STUDY / args.split / args.tag / name
        jobs.append((spec, args.controller, checkpoint, str(prefix), plan['source_sha256'], None))
    results = collect(jobs, args.workers)
    results.sort(key=lambda r: (r['region'], r['direction'], r['kind'], r['seed']))
    atomic_json(STUDY / args.split / f'{args.tag}_complete.json', {
        'tag': args.tag, 'controller': args.controller, 'checkpoint': checkpoint,
        'checkpoint_sha256': sha256_file(checkpoint) if checkpoint else None,
        'source_sha256': plan['source_sha256'], 'results': results, 'summary': summarize(results),
    })


def ensemble(args):
    protocol()
    checkpoints = [torch.load(Path(p), map_location='cpu', weights_only=False) for p in args.checkpoints]
    if len(checkpoints) < 2 or len({c['seed'] for c in checkpoints}) != len(checkpoints):
        raise ValueError('서로 다른 학습 난수의 정책 두 개 이상 필요')
    use_risk = {c.get('use_risk', True) for c in checkpoints}
    if len(use_risk) != 1:
        raise ValueError('위험 입력 계약이 다른 정책을 혼합할 수 없음')
    result = dict(format='cooperative-logit-ensemble-v2', ensemble_models=[c['model'] for c in checkpoints],
                  member_seeds=[c['seed'] for c in checkpoints], use_risk=use_risk.pop(),
                  member_sha256=[sha256_file(Path(p)) for p in args.checkpoints],
                  guard_regret=0.35, forecast_frozen=True, future_truth_in_actor=False)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp'); torch.save(result, temporary); temporary.replace(path)
    print(json.dumps({'ensemble': str(path), 'members': len(checkpoints)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('protocol')
    t = sub.add_parser('train')
    t.add_argument('--mode', choices=['risk', 'no_risk'], required=True)
    t.add_argument('--seed', type=int, required=True)
    t.add_argument('--iterations', type=int, default=8)
    t.add_argument('--batch-episodes', type=int, default=8)
    t.add_argument('--workers', type=int, default=6)
    t.add_argument('--resume', action='store_true')
    e = sub.add_parser('evaluate')
    e.add_argument('--split', choices=['validation', 'test'], default='validation')
    e.add_argument('--controller', required=True)
    e.add_argument('--checkpoint')
    e.add_argument('--tag', required=True)
    e.add_argument('--workers', type=int, default=6)
    e.add_argument('--limit', type=int)
    a = sub.add_parser('ensemble')
    a.add_argument('--checkpoints', nargs='+', required=True)
    a.add_argument('--out', required=True)
    args = p.parse_args()
    if args.command == 'protocol':
        data = protocol(); print(json.dumps({'validation': len(data['validation']), 'test': len(data['test']), 'plan': str(PLAN)}))
    elif args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        ensemble(args)


if __name__ == '__main__':
    main()
