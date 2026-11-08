"""Verify the shared map archives and register disjoint study geography."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import ZipFile
import hashlib
import json
import math
import multiprocessing as mp
import re
import stat

from shapely.geometry import box

from forecast_rebuild.contracts import digest, now, read, sha, write
from training_ops.prepare_additional_maps_20261030 import Union, gap, validate_scene
from . import ROOT, WORKSPACE, LEGACY, ARTIFACTS

BASE_PLAN = WORKSPACE / 'additional_maps_20261030/expanded_plan.json'
REGISTRY = ARTIFACTS / 'map_registry.json'
STATUS = ARTIFACTS / 'status.json'
MAP_ROOT = ROOT / 'registered_maps'


def safe_parts(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or '\\' in name or not path.parts or
            any(part in ('.', '..') or part.endswith((' ', '.')) or
                PureWindowsPath(part).is_reserved() or re.search(r'[<>:"|?*\x00-\x1f]', part)
                for part in path.parts)):
        raise ValueError(f'Unsafe ZIP path: {name}')
    return path.parts


def extract_checked(archive_record):
    source = Path(archive_record['path'])
    if sha(source) != archive_record['sha256']:
        raise ValueError(f'Download changed: {source.name}')
    with ZipFile(source) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        roots = {safe_parts(item.filename)[0] for item in files}
        if len(roots) != 1 or len({item.filename.casefold() for item in files}) != len(files):
            raise ValueError(f'Ambiguous archive paths: {source.name}')
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in files):
            raise ValueError(f'Symlink in archive: {source.name}')
        prefix = roots.pop()
        index = {'/'.join(safe_parts(item.filename)[1:]): item for item in files}
        manifest = json.loads(archive.read(index['package_manifest.json']))
        expected = {entry['path']: entry for entry in manifest['files']}
        if (set(expected) != set(index) - {'package_manifest.json'} or
                len(expected) != manifest['file_count_excluding_manifest'] or
                len(expected) != len(manifest['files'])):
            raise ValueError(f'Manifest coverage differs: {source.name}')

        def checked_bytes(name):
            item = index[name]
            spec = expected[name]
            if item.file_size != spec['size_bytes'] or item.file_size > 128 * 1024**2:
                raise ValueError(f'Manifest length differs: {source.name}/{name}')
            data = archive.read(item)
            if hashlib.sha256(data).hexdigest() != spec['sha256']:
                raise ValueError(f'Manifest checksum differs: {source.name}/{name}')
            return data

        selection_path, = (key for key in index if key.startswith('config/selection') and key.endswith('.json'))
        regions_path, = (key for key in index if key.startswith('config/regions') and key.endswith('.json'))
        selection = json.loads(checked_bytes(selection_path))['candidates']
        regions = json.loads(checked_bytes(regions_path))['regions']
        keys = {record['key'] for record in selection}
        scenes = {key for key in index if key.startswith('maps/') and key.endswith('/scene.local.json')}
        if len(selection) != 50 or len(keys) != 50 or set(regions) != keys or scenes != {
                f'maps/{key}/scene.local.json' for key in keys}:
            raise ValueError(f'Expected 50 declared scenes: {source.name}')
        needed = {selection_path, regions_path}
        records = []
        destination = (MAP_ROOT / prefix).resolve()
        if not destination.is_relative_to(MAP_ROOT.resolve()):
            raise ValueError('Map destination escaped the study directory')
        for entry in selection:
            key = entry['key']
            if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', key):
                raise ValueError(f'Invalid scene key: {key}')
            region = regions[key]
            bounds = entry['aoi_bounds_epsg5179_m']
            if (len(bounds) != 4 or not all(math.isfinite(value) for value in bounds) or
                    abs(bounds[2] - bounds[0] - 400.) > 1e-3 or abs(bounds[3] - bounds[1] - 400.) > 1e-3):
                raise ValueError(f'Invalid EPSG:5179 extent: {key}')
            data = PurePosixPath(*safe_parts(region['data_dir']))
            scene = f'maps/{key}/scene.local.json'
            schema = (data / 'schema.json').as_posix()
            gpkg = (data / 'buildings.gpkg').as_posix()
            needed.update((scene, schema, gpkg))
            records.append(dict(key=key, name=region['name'], path=str(destination / scene),
                                sha256=expected[scene]['sha256'], bounds_epsg5179_m=bounds,
                                center_lonlat=entry['center_lonlat'], source_archive=source.name,
                                source_archive_sha256=archive_record['sha256'],
                                source_dataset=prefix, province=key.split('_', 1)[0],
                                source_geopackage=str(destination / gpkg), source_schema=str(destination / schema)))
        for name in expected:
            data = checked_bytes(name)
            if name not in needed:
                continue
            target = (destination / Path(*safe_parts(name))).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Scene path escaped its archive directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != expected[name]['sha256']:
                    raise ValueError(f'Existing scene changed: {target.name}')
            else:
                target.write_bytes(data)
        failures = []
        for entry in records:
            try:
                validate_scene(entry)
            except (ValueError, AssertionError, KeyError) as error:
                failures.append(dict(key=entry['key'], archive=source.name, error=f'{type(error).__name__}: {error}'))
                entry['validation_error'] = failures[-1]['error']
        return records, failures, dict(archive=source.name, sha256=archive_record['sha256'],
                                       zip_members_verified=len(files), scenes=50,
                                       geometry_verified=50-len(failures))


def identities(record):
    return {str(building['building_id']) for building in read(LEGACY / record['path'])['buildings']}


def split(old, additions):
    entries = [dict(item) for item in old + additions]
    boundary = len(old)
    union = Union(len(entries))
    old_groups = {}
    for i, entry in enumerate(old):
        group = entry['joint_spatial_group']
        if group in old_groups:
            union.join(i, old_groups[group])
        old_groups[group] = i
    good = {i for i in range(boundary, len(entries)) if 'validation_error' not in entries[i]}
    ids = {i: identities(entry) for i, entry in enumerate(entries)
           if (i < boundary and entry['joint_split'] not in ('test', 'quarantine')) or i in good}
    polygons = [box(*entry['bounds_epsg5179_m']) for entry in entries]
    links = []
    duplicates = set()
    for i in sorted(good):
        a = entries[i]
        for j in range(i):
            if j >= boundary and j not in good:
                continue
            b = entries[j]
            distance = gap(a['bounds_epsg5179_m'], b['bounds_epsg5179_m'])
            shared = not ids[i].isdisjoint(ids.get(j, ()))
            protected = j < boundary and b['joint_split'] in ('test', 'quarantine')
            footprint_touch = protected and box(*a['footprint_global_bounds']).intersects(polygons[j])
            if distance >= 100.-1e-5 and not shared and not footprint_touch:
                continue
            overlap = float(polygons[i].intersection(polygons[j]).area)
            same_region = (overlap > .98 * 160000 and ids.get(j) == ids[i]) or a['sha256'] == b['sha256']
            if same_region:
                duplicates.add(i)
            links.append(dict(a=a['key'], b=b['key'], gap_m=distance, overlap_m2=overlap,
                              shared_buildings=shared, protected_footprint_touch=bool(footprint_touch)))
            union.join(i, j)
    groups = defaultdict(list)
    for i in range(len(entries)):
        groups[union.root(i)].append(i)
    novel = []
    for _, indices in groups.items():
        previous = {entries[i]['joint_split'] for i in indices if i < boundary}
        new_good = [i for i in indices if i in good and i not in duplicates]
        if new_good and not previous:
            novel.append(indices)
    # Whole new spatial groups go into a fixed 70/15/15 allocation before any RF runs.
    novel.sort(key=lambda indices: digest(['halfsec-20261107', sorted(entries[i]['key'] for i in indices)]))
    role_by_group = {}
    n_eval = max(1, round(len(novel) * .15))
    for order, indices in enumerate(novel):
        role_by_group[union.root(indices[0])] = ('confirmation' if order < n_eval else
                                                 'calibration' if order < 2*n_eval else 'train')
    for root, indices in groups.items():
        previous = {entries[i]['joint_split'] for i in indices if i < boundary}
        guarded = previous & {'validation', 'historical_validation', 'test', 'quarantine'}
        identifier = digest(sorted(entries[i]['key'] for i in indices))[:16]
        for i in indices:
            entry = entries[i]
            entry['study_group'] = identifier
            entry['new_map'] = i >= boundary
            if i < boundary:
                role = 'train' if entry['joint_split'] == 'train' and not guarded else 'legacy_protected'
            elif i not in good or i in duplicates or guarded & {'test', 'quarantine'} or ('train' in previous and guarded):
                role = 'quarantine'
                if i in duplicates:
                    entry['exclusion_reason'] = 'duplicate delivered geography'
                elif i not in good:
                    entry['exclusion_reason'] = 'scene geometry audit failed'
                else:
                    entry['exclusion_reason'] = 'protected geography or train–validation bridge'
            elif guarded:
                role = 'legacy_protected'
            elif previous:
                role = 'train'
            else:
                role = role_by_group[root]
            entry['study_role'] = role
            if i >= boundary:
                entry['joint_split'] = 'train' if role == 'train' else (
                    'validation' if role in ('calibration', 'confirmation', 'legacy_protected') else 'quarantine')
    for indices in groups.values():
        active = {entries[i]['study_role'] for i in indices} & {'train', 'calibration', 'confirmation'}
        if len(active) > 1:
            raise ValueError('Geography split leakage')
    return entries, links


def main():
    downloaded = read(ROOT / 'map_inputs/downloads.json')
    base_hash = sha(BASE_PLAN)
    if REGISTRY.exists():
        registry = read(REGISTRY)
        if registry['base_plan_sha256'] != base_hash or registry['downloads_sha256'] != sha(ROOT / 'map_inputs/downloads.json'):
            raise ValueError('Registered map inputs changed; use a new study directory')
        print(registry['counts'], flush=True)
        return
    write(STATUS, dict(stage='validating_map_archives', state='running', updated_at=now(), archives=6))
    additions, failures, evidence = [], [], []
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn')) as pool:
        for rows, bad, checked in pool.map(extract_checked, downloaded['files']):
            additions.extend(rows)
            failures.extend(bad)
            evidence.append(checked)
            print(checked, flush=True)
    old = read(BASE_PLAN)['maps']
    if len(additions) != 300 or len({row['key'] for row in old + additions}) != len(old) + len(additions):
        raise ValueError('Missing scenes or a reused scene key')
    entries, links = split(old, additions)
    counts = dict(existing=len(old), added=len(additions), total=len(entries), invalid_geometry=len(failures),
                  roles=dict(Counter(row['study_role'] for row in entries)),
                  new_roles=dict(Counter(row['study_role'] for row in entries if row['new_map'])))
    write(ARTIFACTS / 'map_import_audit.json', dict(downloaded=downloaded, archives=evidence,
          geometry_failures=failures, spatial_links=links, counts=counts, base_plan_sha256=base_hash))
    write(REGISTRY, dict(created_at=now(), base_plan_sha256=base_hash,
                        downloads_sha256=sha(ROOT / 'map_inputs/downloads.json'),
                        boundary='100 m AOI gap, shared building IDs, prior geographic groups and protected footprints',
                        counts=counts, maps=entries))
    write(STATUS, dict(stage='maps_registered', state='ready', updated_at=now(), counts=counts))
    print(counts, flush=True)


if __name__ == '__main__':
    main()
