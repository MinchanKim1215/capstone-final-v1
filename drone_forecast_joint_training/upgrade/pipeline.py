"""동결한 계획을 실행합니다. 독립 학습만 별도 프로세스 두 개로 병렬 처리합니다."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import subprocess
import sys
import time

from swarm_sim.regions import ROOT, read_json, write_json
from upgrade.train import VARIANTS, freeze_training_plan


def run_training(variant, seed):
    name = f'{variant}_s{seed}'
    log = ROOT/'results'/'upgrade'/f'{name}.log'
    print(json.dumps({'stage': 'training_start', 'run': name}), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-u', '-m', 'upgrade.train', '--variant', variant, '--seed', str(seed)],
                                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} 학습 실패. {log}를 확인하십시오.')
    item = read_json(ROOT/'results'/'upgrade'/'runs'/name/'run.json')
    print(json.dumps({'stage': 'training_complete', 'run': name, 'validation_macro_ap': item['validation_macro_ap'],
                      'best_step': item['best_step'], 'elapsed_s': item['elapsed_s']}), flush=True)
    return item


def run(workers=2, stop_after_training=False):
    from upgrade.evaluate import freeze_selection, evaluate_banks, evaluate_all_dense
    started = time.perf_counter()
    plan = freeze_training_plan()
    write_json(ROOT/'results'/'upgrade'/'progress.json', {'stage': 'training_ablations', 'complete': False})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run_training, variant, 11) for variant in VARIANTS]
        for future in as_completed(jobs):
            future.result()
    selected = freeze_selection()
    print(json.dumps({'stage': 'selection_frozen', 'variant': selected['selected_variant'],
                      'validation_macro_ap': selected['validation_macro_ap']}), flush=True)
    write_json(ROOT/'results'/'upgrade'/'progress.json', {'stage': 'replicate_selected_model', 'complete': False,
                                                        'selected_variant': selected['selected_variant']})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run_training, selected['selected_variant'], s) for s in plan['training']['replication_seeds']]
        for future in as_completed(jobs):
            future.result()
    if stop_after_training:
        result = {'stage': 'training_complete_before_test', 'complete': False,
                  'selected_variant': selected['selected_variant'], 'elapsed_s': time.perf_counter()-started}
        write_json(ROOT/'results'/'upgrade'/'progress.json', result)
        return result
    write_json(ROOT/'results'/'upgrade'/'progress.json', {'stage': 'heldout_classification', 'complete': False})
    evaluate_banks()
    write_json(ROOT/'results'/'upgrade'/'progress.json', {'stage': 'heldout_full_flight_evaluation', 'complete': False})
    evaluate_all_dense()
    result = {'stage': 'experiments_complete', 'complete': True, 'elapsed_s': time.perf_counter()-started,
              'selected_variant': selected['selected_variant']}
    write_json(ROOT/'results'/'upgrade'/'progress.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.workers)
