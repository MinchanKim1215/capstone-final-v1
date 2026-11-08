"""원시 기록·정답·비행 안전과 버전 일관성을 별도로 재검사합니다."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import shapely

from swarm_sim.city import City
from swarm_sim.config import Design, Scenario
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from upgrade.collect import source_hashes, prepare_city_without_unused_radio_cache, trajectory
from upgrade.data import collection, checked_observations, index_episode
from upgrade.map_setup import digest


def run():
    manifest, protocol = collection()
    source = source_hashes()
    maps = {m['key']: m for m in protocol['maps']}
    for entry in maps.values():
        if sha256_file(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('지도 파일이 동결 후 변경되었습니다.')
    group_splits = {}
    for entry in maps.values():
        group_splits.setdefault(entry['spatial_group'], set()).add(entry['split'])
    if any(len(parts) != 1 for parts in group_splits.values()):
        raise ValueError('인접 지도 또는 공유 건물이 다른 분할에 있습니다.')
    paths, episodes = {}, []
    rng = np.random.default_rng(5531)
    for record in manifest['records']:
        meta = record['metadata']
        if read_json(ROOT/record['path']/'episode.json') != meta or meta['source_sha256'] != source:
            raise ValueError('수집 메타데이터/코드 버전이 일치하지 않습니다.')
        x, mask = checked_observations(record)
        index, y, up = index_episode(record, protocol)
        with np.load(ROOT/record['path']/'truth.npz', allow_pickle=False) as f:
            available, times = f['link_available'], f['time_end_s']
        np.testing.assert_allclose(np.diff(times), .5, atol=0, rtol=0)
        np.testing.assert_array_equal(available, available.swapaxes(1, 2))
        for row in rng.choice(len(index), min(32, len(index)), replace=False):
            k, i, j = index[row]
            bad = ~available[k+1:k+11, i, j]
            expected = bool(np.any(bad[:-1] & bad[1:]))
            if bool(y[row]) != expected or bool(up[row]) != bool(available[k, i, j]):
                raise ValueError('독립 재계산한 미래 정답이 다릅니다.')
        if np.any(mask[..., 16:27] & ~mask[..., 15:16]):
            raise ValueError('상대 상태를 받은 적 없이 공간 입력이 유효하다고 표시되었습니다.')
        pair = (record['region'], meta['direction'])
        if pair not in paths:
            scenario = Scenario(duration_s=7200, motion_dt_s=3, direction=meta['direction']).validate()
            design = Design(**protocol['design'])
            tag = digest({'map': maps[record['region']]['sha256'], 'design': asdict(design),
                          'scenario': asdict(scenario), 'source': source})[:20]
            cache = ROOT/'cache'/'upgrade'/f'{record["region"]}_{meta["direction"]}_{tag}.npz'
            if not cache.exists():
                rebuild = prepare_city_without_unused_radio_cache(ROOT/maps[record['region']]['path'], scenario, ROOT/'cache'/'upgrade')
                trajectory(rebuild, design, scenario, ROOT/'cache'/'upgrade')
            with np.load(cache, allow_pickle=False) as f:
                a, b = f['xyz_start'][::6], f['xyz_end'][5::6]
            city = City(ROOT/maps[record['region']]['path'], scenario)
            line = shapely.linestrings(np.stack((a[..., :2], b[..., :2]), axis=2).reshape(-1, 2, 2))
            violations = int(np.count_nonzero(shapely.intersects(line, city.flight_blocker)))
            speeds = np.linalg.norm(b-a, axis=-1)/3
            minimum = np.inf
            for i in range(8):
                for j in range(i):
                    r = a[:, i]-a[:, j]
                    v = (b[:, i]-a[:, i])-(b[:, j]-a[:, j])
                    dot = np.sum(v*v, axis=1)
                    alpha = np.divide(-np.sum(r*v, axis=1), dot, out=np.zeros(len(r)), where=dot > 0)
                    closest = r+np.clip(alpha, 0, 1)[:, None]*v
                    minimum = min(minimum, float(np.linalg.norm(closest, axis=1).min()))
            if violations or minimum < 5-1e-7 or speeds.max() > 5+1e-7 or np.any(a[..., 2] != 50):
                raise ValueError(f'{pair}: 비행 안전/고도/속도 재검사 실패')
            paths[pair] = {'region': pair[0], 'direction': pair[1], 'obstacle_intersections': violations,
                           'minimum_continuous_separation_m': minimum, 'maximum_speed_mps': float(speeds.max()),
                           'mission_completed': meta['mission']['mission_completed'], 'duration_s': meta['duration_s'],
                           'all_agents_stationary_fraction': float(np.mean(np.all(speeds < 1e-8, axis=1)))}
        episodes.append({'id': record['id'], 'integrity': True, 'label_spot_checks': min(32, len(index))})
    result = {'complete': True, 'protocol_sha256': protocol['protocol_sha256'], 'map_count': len(maps),
              'all_map_hashes_match': True, 'spatial_groups': len(group_splits), 'split_overlap_groups': 0,
              'episode_count': len(episodes), 'all_episode_hashes_match': True, 'all_source_hashes_match': True,
              'independent_label_checks': sum(e['label_spot_checks'] for e in episodes),
              'unique_paths': len(paths), 'completed_paths': sum(p['mission_completed'] for p in paths.values()),
              'mission_deadline_paths': sum(not p['mission_completed'] for p in paths.values()),
              'max_obstacle_intersections': max(p['obstacle_intersections'] for p in paths.values()),
              'minimum_continuous_separation_m': min(p['minimum_continuous_separation_m'] for p in paths.values()),
              'paths': list(paths.values()), 'episodes': episodes,
              'scenario_overrides': 'episode jammer_power_dbm/jammer_window_s are effective values; scenario is baseline dataclass before these explicit overrides',
              'peer_header_source_definition': 'current radio transmitter for each hop, not original application-packet origin'}
    write_json(ROOT/'results'/'upgrade'/'integrity_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('paths', 'episodes')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    run()
