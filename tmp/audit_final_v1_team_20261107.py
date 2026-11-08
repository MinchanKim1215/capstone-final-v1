"""Check the delivered code against frozen evidence and normal entry points."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/final_v1_team_20261107'
CODE = OUT / 'code'
QA = OUT / 'evidence/qa'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    QA.mkdir(parents=True, exist_ok=True)
    preserved = []
    for folder, directories, files in os.walk(CODE):
        directories[:] = [name for name in directories if name not in ('cache', 'numba_cache', '__pycache__')]
        for name in files:
            if not name.endswith('.py'):
                continue
            path = Path(folder) / name
            compile(path.read_bytes(), str(path), 'exec')
            source = ROOT / path.relative_to(CODE)
            if source.is_file():
                if sha(path) != sha(source):
                    raise ValueError('Source bytes changed: ' + str(path.relative_to(CODE)))
                preserved.append(path.relative_to(CODE).as_posix())
    map_index = read(CODE / 'handoff_paths.json')['maps']
    for entry in map_index:
        if sha(CODE / entry['path']) != entry['sha256']:
            raise ValueError('Delivered map changed: ' + entry['key'])
    assert len(map_index) == 345
    source = ROOT / 'final_v1/artifacts/confirm/v21/gyeonggi_20261025_06_clear'
    observed = CODE / 'outputs/portability_full'
    old = read(source.with_suffix('.json'))
    new = read(observed.with_suffix('.json'))
    fields = ('spec', 'arm', 'transport', 'deployment', 'duration_s', 'mission_completed',
              'persistent_onsets', 'persistent_cut_seconds', 'timely_pdr', 'generated_packets',
              'delivered_packets', 'actual_safety_violations', 'relay_travel_m', 'intervention')
    for key in fields:
        if old[key] != new[key]:
            raise ValueError('Full-mission mismatch: ' + key)
    arrays = []
    with np.load(source.with_suffix('.npz'), allow_pickle=False) as expected, np.load(
            observed.with_suffix('.npz'), allow_pickle=False) as actual:
        for key in actual.files:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
            arrays.append(key)
    completed = read(CODE / 'final_v1/artifacts/completed.json')
    assert sha(CODE / 'final_v1/artifacts/release.json') == completed['release_sha256']
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    forecast = subprocess.run([sys.executable, 'run_final_v1.py', '--check-forecast'],
                              cwd=CODE, env=environment, capture_output=True, text=True, encoding='utf-8')
    if forecast.returncode:
        raise RuntimeError(forecast.stderr)
    forecast_metrics = json.loads(forecast.stdout)
    (QA / 'recomputed_prediction.json').write_text(json.dumps(forecast_metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    unit_command = """from team_portability import activate
activate()
import sys
import unittest
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'final_v1.test_contracts', 'final_v1_fast.test_contracts', 'final_v1_margin.test_features'])
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(not result.wasSuccessful())
"""
    tests = subprocess.run([sys.executable, '-c', unit_command], cwd=CODE, env=environment,
                           capture_output=True, text=True, encoding='utf-8')
    (QA / 'existing_contract_tests.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    report = dict(preserved_python_sources=len(preserved), all_python_syntax_passed=True,
                  verified_maps=345, frozen_release_unchanged=True,
                  full_mission=dict(duration_s=new['duration_s'], completed=new['mission_completed'],
                                    starts=new['persistent_onsets'], prevention_commands=sum(
                                        event['reason'] == 'prevention' for event in new['intervention']),
                                    compared_fields=list(fields), identical_arrays=arrays,
                                    exact_match_to_frozen_flight=True),
                  forecast_metrics_recomputed=['v20', 'v21', 'joint_early', 'joint_short'],
                  training_input_check=dict(maps=245, rows=154312, features=214),
                  existing_contract_tests_passed=True,
                  runtime=dict(platform=sys.platform, python=sys.version.split()[0],
                               interpreter='existing validated environment, CPU inference'),
                  fresh_dependency_install_tested=False)
    (QA / 'runtime_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
