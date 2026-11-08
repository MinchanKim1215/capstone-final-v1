"""Team entry point for Final V1 inference, evidence checks and stored evaluation."""
from pathlib import Path
import argparse
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')


def main():
    parser = argparse.ArgumentParser(description='Final V1 team handoff')
    parser.add_argument('--verify', action='store_true', help='Verify delivered file contents')
    parser.add_argument('--check-training-data', action='store_true', help='Load the 245-map supervised inputs')
    parser.add_argument('--check-forecast', action='store_true', help='Recompute same-flight forecast comparison')
    parser.add_argument('--region')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--kind', choices=('clear', 'intermittent', 'continuous'))
    parser.add_argument('--direction', choices=('east_west', 'north_south'))
    parser.add_argument('--steps', type=int)
    parser.add_argument('--command-loss', action='store_true')
    parser.add_argument('--save', help='Result prefix inside code/outputs')
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    os.chdir(ROOT)
    from team_portability import activate, read_evidence, resolve, sha, verify_files
    if args.verify:
        print(json.dumps(verify_files(), ensure_ascii=False, indent=2))
        return
    activate()
    if args.check_training_data:
        from final_v1.training import load_train
        x, labels = load_train()
        print(json.dumps(dict(rows=len(x), features=x.shape[1], labels=len(labels)), indent=2))
        return
    if args.check_forecast:
        import numpy as np
        from final_v1.evaluation import advisory_case, forecast_model, score_advisory
        from final_v1 import ARTIFACTS
        comparison = read_evidence(ARTIFACTS / 'comparison.json')
        model = forecast_model(read_evidence(ARTIFACTS / 'training.json'))
        threshold = read_evidence(ARTIFACTS / 'comparison_freeze.json')['thresholds']
        flights = []
        for record in comparison['manifest']:
            if resolve(record['path']).parent.name != 'v7':
                continue
            row = read_evidence(record['path'])
            with np.load(resolve(row['archive_path']), allow_pickle=False) as data:
                inputs = data['forecast_features']
                flights.append(advisory_case(row, np.arange(len(inputs)) * 3., inputs, data['network'], model))
        result = {key: score_advisory(flights, key, threshold[key]['threshold'], True)
                  for key in ('v20', 'v21', 'joint_early', 'joint_short')}
        for key, metrics in result.items():
            expected = comparison['alerts'][key]
            for field in ('events', 'hits', 'alerts', 'false_or_late', 'ranking_rows', 'ranking_positives',
                          'recall', 'precision', 'average_precision'):
                if abs(metrics[field] - expected[field]) > 1e-10:
                    raise ValueError(f'Recomputed {key}/{field} differs from the frozen result')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    from final_v1 import ARTIFACTS
    from final_v1.runner import initialize, run
    from final_v1.training import read_checkpoint
    release_path = ARTIFACTS / 'release.json'
    if sha(release_path) != read_evidence(ARTIFACTS / 'completed.json')['release_sha256']:
        raise ValueError('Final V1 release content changed')
    release = read_evidence(release_path)
    spec = read_evidence(ARTIFACTS / 'protocol.json')['confirmation'][0].copy()
    for key in spec:
        value = getattr(args, key)
        if value is not None:
            spec[key] = value
    state = None
    if release['model']:
        metadata = release['model']
        if sha(metadata['path']) != metadata['sha256']:
            raise ValueError('Released adapter content changed')
        state = read_checkpoint(metadata['path'])['state']
    initialize()
    row, data = run(spec, release['controller'], state_dict=state,
                    transport=release['transport_fallback'] if args.command_loss else None,
                    max_steps=args.steps)
    if args.save:
        import numpy as np
        prefix = (ROOT / 'outputs' / args.save).resolve()
        if not prefix.is_relative_to(ROOT / 'outputs'):
            parser.error('--save must stay inside code/outputs')
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prefix.with_suffix('.npz'), **data)
        prefix.with_suffix('.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
    keys = ('spec', 'arm', 'transport', 'deployment', 'duration_s', 'mission_completed',
            'persistent_onsets', 'persistent_cut_seconds', 'timely_pdr', 'audited')
    print(json.dumps(dict(version=release['version'], **{key: row[key] for key in keys}),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
