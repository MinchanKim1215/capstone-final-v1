"""Import supplied map data without executing archive scripts or changing v6."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
import re
import stat
import sqlite3
import tempfile
import numpy as np
from shapely import from_wkb, force_2d, make_valid
from shapely.affinity import translate
from shapely.geometry import shape, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'additional_maps_20261030'
OLD = ROOT / 'drone_forecast_joint_training'
ARCHIVES = [Path(r'C:/Users/mckim/OneDrive/문서/카카오톡 받은 파일') / name for name in (
    'capstone-additional50-maps-20261030.zip', 'capstone-additional50-maps-20261030-batch2.zip')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        f.write(payload); f.flush(); os.fsync(f.fileno()); temporary = f.name
    os.replace(temporary, path)


def extract(archive_path, batch):
    archive_sha = sha(archive_path)
    with ZipFile(archive_path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        manifest_name = next(m.filename for m in members if m.filename.endswith('/package_manifest.json'))
        prefix = PurePosixPath(manifest_name).parts[0]
        manifest = json.loads(z.read(manifest_name))
        expected = {r['path']: r for r in manifest['files']}
        assert len(expected) == manifest['file_count_excluding_manifest'] == len(members)-1
        seen = set(); checked = set()
        for member in members:
            p = PurePosixPath(member.filename)
            if '\\' in member.filename or p.is_absolute() or any(s in ('..', '.') or ':' in s for s in p.parts):
                raise ValueError('Unsafe archive path: '+member.filename)
            if p.parts[0] != prefix or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unexpected root or link in archive')
            canonical = member.filename.casefold()
            if canonical in seen: raise ValueError('Duplicate Windows archive path')
            seen.add(canonical)
            relative = PurePosixPath(*p.parts[1:]).as_posix()
            payload = z.read(member)
            if relative != 'package_manifest.json':
                record = expected[relative]
                assert len(payload) == record['size_bytes']
                assert hashlib.sha256(payload).hexdigest() == record['sha256']
                checked.add(relative)
            target = (DEST / 'raw' / Path(*p.parts)).resolve()
            if (DEST/'raw').resolve() not in target.parents: raise ValueError('Extraction escaped dataset root')
            if target.exists():
                if target.read_bytes() != payload: raise ValueError('Existing imported file changed: '+str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
        assert checked == set(expected)
        base = DEST / 'raw' / prefix
        selection = read(next((base/'config').glob('selection*.json')))
        regions = read(next((base/'config').glob('regions*.json')))['regions']
        assert len(regions) == len(selection['candidates']) == 50
        scenes = {p.parent.name:p for p in (base/'maps').glob('*/scene.local.json')}
        assert set(scenes) == set(regions) == {r['key'] for r in selection['candidates']}
        records = []
        for candidate in selection['candidates']:
            k = candidate['key']; path = scenes[k]
            assert re.fullmatch(r'[A-Za-z0-9_]+', k)
            record = dict(key=k, name=regions[k]['name'], path=str(path.resolve()), sha256=sha(path),
                          bounds_epsg5179_m=candidate['aoi_bounds_epsg5179_m'],
                          center_lonlat=candidate['center_lonlat'], source_batch=batch,
                          source_dataset=f'additional50_batch{batch}', source_archive_sha256=archive_sha,
                          source_geopackage=str((base/regions[k]['data_dir']/'buildings.gpkg').resolve()),
                          source_schema=str((base/regions[k]['data_dir']/'schema.json').resolve()),
                          province='incheon' if batch==2 else 'gyeonggi')
            records.append(record)
        return records, dict(archive=str(archive_path), sha256=archive_sha,
                             batch=batch, files_verified=len(members), scenes=50, extracted_to=str(base))


class Union:
    def __init__(self, n): self.parent=list(range(n))
    def root(self, i):
        while self.parent[i] != i:
            self.parent[i]=self.parent[self.parent[i]]; i=self.parent[i]
        return i
    def join(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b: self.parent[max(a,b)]=min(a,b)


def gap(a,b):
    return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))


def validate_scene(record):
    scene=read(record['path']); m=scene['metadata']; buildings=scene['buildings']
    assert m['region_key']==record['key'] and m['local_bounds']==[-200,-200,200,200]
    policy=m['flight_policy']; assert policy['flight_altitude_m']==50 and policy['safety_margin_m']==5
    assert policy['comparison']=='>=' and 'building geometry stays unchanged' in policy['geometry_policy'].lower()
    ids=[str(b['building_id']) for b in buildings]
    assert len(ids)==len(set(ids)) and all(ids)
    heights=np.array([b['height_m'] for b in buildings],float)
    geometry=[shape(b['geometry_local_m']) for b in buildings]
    assert np.isfinite(heights).all() and (heights>0).all()
    assert all(g.is_valid and not g.is_empty and g.area>0 for g in geometry)
    assert {b['height_source'] for b in buildings}<={'gis','imputed','ledger'}
    # Scene building polygons are AOI-clipped, but delivered safety zones were
    # buffered from FULL source footprints before clipping. Recomputing the
    # buffer from the scene alone would incorrectly shrink edge constraints.
    schema=read(record['source_schema'])
    assert schema['source_crs']=='EPSG:5179'
    id_field=schema['id_field'];table=schema['layer']
    assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',id_field) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',table)
    origin=m['dataset_provenance']['buildings']['local_origin_projected_m']
    full={}
    uri=Path(record['source_geopackage']).as_uri()+'?mode=ro&immutable=1'
    with sqlite3.connect(uri,uri=True) as connection:
        geom_field,srs=connection.execute('select column_name,srs_id from gpkg_geometry_columns where table_name=?',(table,)).fetchone()
        columns={r[1] for r in connection.execute(f'pragma table_info("{table}")')}
        assert id_field in columns and geom_field in columns and srs==5179
        assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',geom_field)
        for bid,payload in connection.execute(f'select "{id_field}","{geom_field}" from "{table}"'):
            assert payload[:2]==b'GP'
            envelope=(payload[3]>>1)&7
            offset=8+{0:0,1:32,2:48,3:48,4:64}[envelope]
            g=force_2d(from_wkb(payload[offset:]))
            if not g.is_valid:g=make_valid(g)
            if g.geom_type=='GeometryCollection':g=unary_union([v for v in g.geoms if v.geom_type in ('Polygon','MultiPolygon')])
            if str(bid) in full:raise ValueError('Duplicate source GIS identity')
            full[str(bid)]=translate(g,xoff=-origin[0],yoff=-origin[1])
    assert set(full)==set(ids)
    aoi=box(-200,-200,200,200)
    clipped_error=sum(g.symmetric_difference(full[i].intersection(aoi)).area for i,g in zip(ids,geometry))
    if clipped_error>.01:raise ValueError('Scene footprints differ from delivered GIS source: '+record['key'])
    expected=unary_union([full[i].buffer(5).intersection(aoi) for i,h in zip(ids,heights) if h>=50])
    zones=[shape(z['geometry_local_m']) for z in scene['no_fly_zones']]
    assert all(g.is_valid for g in zones)
    actual=unary_union(zones)
    error=actual.symmetric_difference(expected).area
    boundary_error=actual.hausdorff_distance(expected) if not actual.is_empty and not expected.is_empty else (0. if actual.is_empty and expected.is_empty else math.inf)
    # Buffering in million-metre projected coordinates and local coordinates
    # can differ at floating-point clipping corners. Preserve delivered zones.
    if boundary_error>.01 or error>.01:
        raise ValueError(f'Incompatible flight constraints: {record["key"]}: area={error}, boundary={boundary_error}')
    known=sum(b['height_source']!='imputed' for b in buildings)
    bounds=record['bounds_epsg5179_m']; x=(bounds[0]+bounds[2])/2; y=(bounds[1]+bounds[3])/2
    footprint=unary_union(list(full.values())).bounds
    record.update(buildings=len(buildings), known_heights=known, imputed_heights=len(buildings)-known,
                  high_buildings_ge50=int((heights>=50).sum()), no_fly_area_fraction=actual.area/160000,
                  geometry_sha256=digest([dict(id=i,geometry=b['geometry_local_m'],height=b['height_m']) for i,b in zip(ids,buildings)]),
                  source_height_policy=m['height_policy'],
                  footprint_global_bounds=[footprint[0]+x,footprint[1]+y,footprint[2]+x,footprint[3]+y],
                  source_scene_bytes_preserved=True, flight_zone_difference_m2=error,
                  flight_zone_boundary_difference_m=boundary_error,
                  full_source_geometry_verified=True,clipped_scene_source_difference_m2=clipped_error,
                  source_geopackage_sha256=sha(record['source_geopackage']))
    return set(ids)


def main():
    DEST.mkdir(exist_ok=True)
    old_plan=read(OLD/'configs/transformer_plan.json'); old_maps=old_plan['maps']
    new=[]; archives=[]
    for i,path_archive in enumerate(ARCHIVES,1):
        records,manifest=extract(path_archive,i)
        new.extend(records);archives.append(manifest)
        print(json.dumps(dict(stage='verified_extract',batch=i,files=manifest['files_verified'],maps=50)),flush=True)
    assert len(new)==100 and len({m['key'] for m in new})==100
    assert not ({m['key'] for m in new}&{m['key'] for m in old_maps})
    ids={m['key']:validate_scene(m) for m in new}
    old_ids={}
    for i,m in enumerate(old_maps):
        if m['joint_split']=='test':continue  # Never open held-out geometry.
        old_ids[i]={str(b['building_id']) for b in read(OLD/m['path'])['buildings']}
    all_maps=old_maps+new; nold=len(old_maps); u=Union(len(all_maps)); old_groups={}
    for i,m in enumerate(old_maps):
        g=m['joint_spatial_group']
        if g in old_groups:u.join(i,old_groups[g])
        old_groups[g]=i
    edges=[]; positive_overlaps=[]; near_protected=[]
    for i in range(nold,len(all_maps)):
        a=all_maps[i]; ga=box(*a['bounds_epsg5179_m'])
        for j in range(i):
            b=all_maps[j]; distance=gap(a['bounds_epsg5179_m'],b['bounds_epsg5179_m'])
            shared=bool(ids[a['key']] & (old_ids.get(j,set()) if j<nold else ids[b['key']]))
            protected_cross=j<nold and b['joint_split']=='test' and box(*a['footprint_global_bounds']).intersects(box(*b['bounds_epsg5179_m']))
            overlap=ga.intersection(box(*b['bounds_epsg5179_m'])).area
            if overlap>1e-4:positive_overlaps.append(dict(new=a['key'],other=b['key'],area_m2=overlap))
            if distance<100-1e-5 or shared or protected_cross:
                u.join(i,j);edges.append((i,j))
                if j<nold and b['joint_split'] in ('test','validation','historical_validation'):
                    near_protected.append(dict(new=a['key'],old=b['key'],split=b['joint_split'],gap_m=distance,shared_ids=shared,protected_footprint_cross=protected_cross))
    groups=defaultdict(list)
    for i in range(len(all_maps)):groups[u.root(i)].append(i)
    novel={}; quarantine=[]
    for g,indices in groups.items():
        additions=[i for i in indices if i>=nold]
        if not additions:continue
        roles={old_maps[i]['joint_split'] for i in indices if i<nold}
        protected=roles&{'test','validation','historical_validation'}
        if 'test' in roles or ('train' in roles and protected):
            for i in additions:
                new[i-nold].update(joint_split='quarantine',exclusion_reason='Protected geography or bridge between existing training and held-out geography')
                quarantine.append(new[i-nold]['key'])
        elif protected:
            for i in additions:new[i-nold]['joint_split']='validation'
        elif 'train' in roles:
            for i in additions:new[i-nold]['joint_split']='train'
        else:novel[g]=additions
    inherited=sum(m.get('joint_split')=='validation' for m in new)
    target=max(0,20-inherited)
    states={0:([],set())}
    ordered=sorted(novel,key=lambda g:digest(['expanded100-split', [all_maps[i]['key'] for i in novel[g]]]))
    for g in ordered:
        weight=len(novel[g]);provinces={all_maps[i]['province'] for i in novel[g]}
        for count,(chosen,seen) in list(states.items())[::-1]:
            candidate=(chosen+[g],seen|provinces);total=count+weight
            if total not in states or len(candidate[1])>len(states[total][1]):states[total]=candidate
    chosen_count=min(states,key=lambda n:(abs(n-target),-len(states[n][1]),n))
    selected=set(states[chosen_count][0])
    for g,indices in novel.items():
        for i in indices:new[i-nold]['joint_split']='validation' if g in selected else 'train'
    final=Union(len(all_maps))
    for i,m in enumerate(old_maps):final.join(i,old_groups[m['joint_spatial_group']])
    for i,j in edges:
        if all_maps[i].get('joint_split')!='quarantine' and all_maps[j].get('joint_split')!='quarantine':final.join(i,j)
    assigned={g:n for n,g in enumerate(sorted({final.root(i) for i in range(len(all_maps))}))}
    maps=[]
    for i,m in enumerate(all_maps):
        item=dict(m,joint_spatial_group=assigned[final.root(i)])
        item.setdefault('source_dataset','legacy_v6')
        maps.append(item)
    check=defaultdict(set)
    for m in maps:
        if m['joint_split']!='quarantine':check[m['joint_spatial_group']].add('heldout' if m['joint_split'] in ('test','validation','historical_validation') else 'train')
    assert all(not ('train' in v and 'heldout' in v) for v in check.values())
    development=list(old_plan['validation'])
    for m in maps[nold:]:
        if m['joint_split']!='validation':continue
        for direction in ('east_west','north_south'):
            for kind in ('continuous','intermittent'):
                seed=int(digest(['expanded100-validation-v7',m['key'],direction,kind])[:8],16)%(2**31-3000004)
                development.append(dict(region=m['key'],direction=direction,kind=kind,seed=seed))
    expanded=dict(old_plan,maps=maps,validation=development,spatial_groups=len(assigned),
                  format='expanded-map-plan-v7',parent_plan_sha256=sha(OLD/'configs/transformer_plan.json'),
                  expansion=dict(archives=archives,new_maps=100,quarantined=quarantine,
                                 split_rule='Inherited spatial components, protected test envelopes, shared building IDs; target 20 validation maps among additions',
                                 test_geometry_opened=False,attached_document_instructions_executed=False))
    expanded['counts']=dict(Counter(m['joint_split'] for m in maps),maps=len(maps),validation_cases=len(development),test_cases=len(expanded['test']))
    atomic(DEST/'expanded_plan.json',expanded)
    audit=dict(created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
               archives=archives,total_new_maps=100,new_split_counts=dict(Counter(m['joint_split'] for m in maps[nold:])),
               all_split_counts=dict(Counter(m['joint_split'] for m in maps)),positive_aoi_overlaps=positive_overlaps,
               new_near_protected=near_protected,quarantined=quarantine,
               test_geometry_opened=False,old_map_and_source_files_modified=False,
               scene_format_validated=100,source_scene_bytes_preserved=True,
               existing_test_identity_check='AOI and full new-building envelope metadata only; existing test scenes never read',
               attachments_used_as_data_not_instructions=True,expanded_plan_sha256=sha(DEST/'expanded_plan.json'))
    atomic(DEST/'import_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('new_near_protected','archives')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
