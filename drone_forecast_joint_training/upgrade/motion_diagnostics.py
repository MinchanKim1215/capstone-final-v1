"""임무 미완료 경로의 말단 반복을 확인합니다. 기록을 제거하거나 고치지 않습니다."""
import json

import numpy as np

from swarm_sim.regions import ROOT, write_json
from upgrade.data import collection


def run():
    manifest, _ = collection()
    paths = {}
    for record in manifest['records']:
        meta = record['metadata']
        key = (record['region'], meta['direction'])
        if key in paths:
            continue
        with np.load(ROOT/record['path']/'truth.npz', allow_pickle=False) as f:
            # 0.5초 보간 기록에서 3초 운동 갱신 시점만 추출합니다.
            coarse = f['positions_start_m'][::6, :, :2].reshape(-1, 16)
        last = coarse[-400:]
        unique, counts = np.unique(last, axis=0, return_counts=True)
        paths[key] = {'region': key[0], 'direction': key[1], 'mission_completed': meta['mission']['mission_completed'],
                      'duration_s': meta['duration_s'], 'terminal_window_s': len(last)*3,
                      'unique_joint_configurations_in_terminal_window': len(unique),
                      'most_repeated_configuration_count': int(counts.max()),
                      'incomplete_terminal_one_or_two_state_repeat': bool(not meta['mission']['mission_completed'] and len(last) == 400 and len(unique) <= 2)}
    result = {'complete': True, 'primary_results_unchanged': True, 'paths': list(paths.values()),
              'paths_examined': len(paths), 'deadline_paths': sum(not p['mission_completed'] for p in paths.values()),
              'incomplete_terminal_one_or_two_state_repeats': sum(p['incomplete_terminal_one_or_two_state_repeat'] for p in paths.values()),
              'definition': 'Descriptive check of last 400 three-second joint configurations; no exclusion rule.'}
    write_json(ROOT/'results'/'upgrade'/'motion_diagnostics.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'paths'},ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    run()
