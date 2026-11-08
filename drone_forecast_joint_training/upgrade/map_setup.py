"""지도 원본 보존, 인접·공유 건물 집단 분리, 결과를 보기 전 분할 동결."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import shutil

import numpy as np
from shapely.geometry import box, shape
from shapely import union_all

from swarm_sim.regions import ROOT, validate_scene, write_json, sha256_file

ORIGINAL_KEYS = ('gangnam', 'teheran', 'bundang', 'hongdae', 'sangam_dmc')


def digest(value):
    """설정의 순서와 무관한 해시를 만들되 실험 내용 변경은 감지합니다."""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _subset(groups, count, rng):
    """같은 공간 집단을 쪼개지 않고 필요한 지도 수를 고릅니다."""
    reachable = {0: []}
    order = rng.permutation(len(groups))
    for no in order:
        size = len(groups[int(no)])
        for before, selected in list(reachable.items())[::-1]:
            after = before + size
            if after <= count and after not in reachable:
                reachable[after] = selected + [int(no)]
        if count in reachable:
            return reachable[count]
    raise ValueError(f'집단을 유지하며 {count}개 지도를 고를 수 없습니다.')


def prepare_maps(archives, root=ROOT, seed=20261025):
    """시뮬레이션 성능을 읽지 않고 지도 특성만으로 분할을 확정합니다."""
    root = Path(root).resolve()
    frozen = root / 'configs' / 'upgrade_protocol.json'
    if frozen.exists():
        protocol = json.loads(frozen.read_text(encoding='utf-8'))
        for item in protocol['maps']:
            if sha256_file(root / item['path']) != item['sha256']:
                raise ValueError('분할 동결 후 지도 파일이 바뀌었습니다.')
        return protocol
    archive_records = []
    for archive in map(Path, archives):
        archive_records.append({'name': archive.name, 'sha256': sha256_file(archive)})
        with ZipFile(archive) as z:
            scene_files = sorted(n for n in z.namelist() if n.endswith('/scene.local.json'))
            if len(scene_files) != 20:
                raise ValueError('이번 두 전달본에는 각각 20개 scene이 있어야 합니다.')
            for member in scene_files:
                scene = json.loads(z.read(member))
                key = scene['metadata']['region_key']
                destination = root / 'maps' / 'additional' / key
                destination.mkdir(parents=True, exist_ok=True)
                prefix = member.rsplit('/', 1)[0]
                for name in ('scene.local.json', 'no_fly.local.json', 'quality.csv', 'manifest.json',
                             'LICENSE-BUILDINGS.md', 'LICENSE-OSM.md'):
                    source = prefix + '/' + name
                    if source in z.namelist():
                        (destination / name).write_bytes(z.read(source))
    entries = []
    files = list((root / 'maps').glob('*.scene.local.json'))
    files += list((root / 'maps' / 'additional').glob('*/scene.local.json'))
    ids, regions = {}, {}
    for path in sorted(files):
        audit = validate_scene(path)
        scene = json.loads(path.read_text(encoding='utf-8'))
        meta, buildings = scene['metadata'], scene['buildings']
        key = audit['key']
        if key in regions:
            raise ValueError(f'중복 지역 ID: {key}')
        source_counts = Counter(b['height_source'] for b in buildings)
        geometry = box(*meta['aoi_bounds_projected_m'])
        ids[key] = {b['building_id'] for b in buildings}
        regions[key] = geometry
        high = sum(b['height_m'] >= 50 for b in buildings)
        blocked = union_all([shape(z['geometry_local_m']) for z in scene['no_fly_zones']]).area
        entries.append({
            'key': key, 'name': audit['name'], 'path': path.relative_to(root).as_posix(),
            'sha256': audit['sha256'], 'geometry_sha256': audit['geometry_sha256'],
            'bounds_epsg5179_m': meta['aoi_bounds_projected_m'], 'center_lonlat': audit['center_lonlat'],
            'buildings': len(buildings), 'known_heights': source_counts['gis'],
            'imputed_heights': source_counts['imputed'], 'high_buildings_ge50': high,
            'no_fly_area_fraction': float(blocked / 160000),
        })
    if len(entries) != 45 or set(ORIGINAL_KEYS) - set(regions):
        raise ValueError('기존 5개 + 추가 40개 지도 구성이 아닙니다.')
    parent = {key: key for key in regions}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    links = []
    keys = sorted(regions)
    for i, a in enumerate(keys):
        for b in keys[:i]:
            overlap = regions[a].intersection(regions[b]).area
            gap = regions[a].distance(regions[b])
            common = len(ids[a] & ids[b])
            # 지도 경계 간 100m 이내는 같은 집단입니다. 건물 ID 공유도 별도로 잡습니다.
            if gap < 100 or common or overlap > 1e-6:
                parent[find(a)] = find(b)
                links.append({'a': a, 'b': b, 'gap_m': float(gap),
                              'overlap_m2': float(overlap), 'shared_building_ids': common})
    groups = {}
    for key in keys:
        groups.setdefault(find(key), []).append(key)
    ordered = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    held_for_train = [g for g in ordered if set(g) & set(ORIGINAL_KEYS)]
    eligible = [g for g in ordered if not (set(g) & set(ORIGINAL_KEYS))]
    rng = np.random.default_rng(seed)
    test_numbers = _subset(eligible, 8, rng)
    test_groups = [eligible[i] for i in test_numbers]
    remaining = [g for i, g in enumerate(eligible) if i not in test_numbers]
    val_numbers = _subset(remaining, 7, rng)
    val_groups = [remaining[i] for i in val_numbers]
    train_groups = held_for_train + [g for i, g in enumerate(remaining) if i not in val_numbers]
    splits = {'train': sorted(k for g in train_groups for k in g),
              'val': sorted(k for g in val_groups for k in g),
              'test': sorted(k for g in test_groups for k in g)}
    assert [len(splits[s]) for s in ('train', 'val', 'test')] == [30, 7, 8]
    assignment = {key: split for split, members in splits.items() for key in members}
    for entry in entries:
        entry['split'] = assignment[entry['key']]
        entry['spatial_group'] = next(i for i, g in enumerate(ordered) if entry['key'] in g)
    protocol = {
        'format': 'single-transformer-upgrade-v1', 'date_kst': '2026-10-25', 'seed': seed,
        'prediction_model': 'lightweight_multivariate_transformer',
        'source_archives': archive_records, 'maps': entries, 'splits': splits,
        'spatial_group_policy': 'Shared building IDs, positive AOI overlap, or AOI gap <100m stay in one split.',
        'spatial_groups': ordered, 'spatial_links': links,
        'all_original_five_are_training': True,
        'split_selected_without_simulation_outcomes': True,
        'design': {'n_drones': 8, 'n_relays': 5, 'formation': 'v', 'spacing_m': 40},
        'radio_dt_s': 0.5, 'lookback_s': 20, 'horizon_s': 5, 'min_outage_s': 1,
        'stride_s': 1, 'minimum_observed_tx_intervals': 1,
        'model_start_spec': {'d_model': 64, 'nhead': 4, 'layers': 2, 'feedforward': 128, 'dropout': 0.1},
        'target_definition': 'At currently available directed pairs, future 5s contains >=1s bidirectional unavailability.',
        'truth_availability_used_only_for_offline_labels_and_evaluation_cohort': True,
        'original_channel_and_packet_parameters_preserved': True,
        'telemetry_contract': '32B timestamp/position/velocity fields in already modeled 64B data header; received successful inbound data only; no added frames.',
        'evaluation': {'fpr_budget': 0.05, 'minimum_lead_s': 3, 'test_used_for_tuning': False},
        'sources': [
            {'title': 'A Transformer-based Framework for Multivariate Time Series Representation Learning',
             'url': 'https://arxiv.org/abs/2010.02803', 'use': 'Masked-history reconstruction then supervised fine-tuning; adapted architecture.'},
            {'title': 'Point Cloud-based Proactive Link Quality Prediction for Millimeter-wave Communications',
             'url': 'https://arxiv.org/abs/2301.00752', 'use': 'Motivation for spatial/motion cues; 60GHz indoor results do not establish our 2.4GHz performance.'},
        ],
    }
    protocol['protocol_sha256'] = digest(protocol)
    write_json(frozen, protocol)
    registry = {'format': 'region-catalog-v1', 'regions': {
        e['key']: {'path': '../' + e['path'], 'sha256': e['sha256']} for e in entries}}
    write_json(root / 'configs' / 'regions_upgrade.json', registry)
    write_json(root / 'results' / 'upgrade' / 'map_audit.json', {
        'all_45_loader_validations_passed': True, 'maps': entries, 'spatial_links': links,
        'split_sizes': {s: len(v) for s, v in splits.items()},
        'shared_building_ids_across_splits': 0,
    })
    return protocol
