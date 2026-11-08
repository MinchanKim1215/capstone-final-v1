"""팀의 local-metre-scene-v1 지도를 검증하고 등록한다. 원본은 변환하지 않는다."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import re
import tempfile

import shapely
from shapely.geometry import shape, box

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / 'configs' / 'regions.json'
KEY = re.compile(r'^[a-z][a-z0-9_]{0,63}$')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f'중복 JSON 키: {key}')
            out[key] = value
        return out
    def constant(value):
        raise ValueError(f'JSON에 유한하지 않은 값이 있습니다: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=pairs, parse_constant=constant)


def write_json(path, value):
    """같은 파일 시스템의 임시 파일을 교체하여 중단 중 불완전한 JSON을 피한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def validate_scene(path):
    """400m·지면 기준 높이·미터 좌표라는 현재 엔진의 계약을 검사한다."""
    path = Path(path).resolve()
    scene = read_json(path)
    if not isinstance(scene, dict) or not str(scene.get('format', '')).startswith('local-metre-scene-v1'):
        raise ValueError('팀의 local-metre-scene-v1 형식이 필요합니다. 경위도 GeoJSON을 그대로 넣을 수 없습니다.')
    meta = scene.get('metadata', {})
    key = meta.get('region_key', '')
    if not isinstance(key, str) or not KEY.fullmatch(key):
        raise ValueError('region_key는 소문자 영문으로 시작하는 영문·숫자·밑줄 이름이어야 합니다.')
    if not isinstance(meta.get('region_name'), str) or not meta['region_name'].strip():
        raise ValueError('metadata.region_name이 필요합니다.')
    if meta.get('local_bounds') != [-200, -200, 200, 200] or meta.get('size_m') != 400:
        raise ValueError('현재 비행·채널 엔진은 중심 원점의 400×400m만 지원합니다.')
    source = meta.get('building_source', {})
    if source.get('height_unit', 'm') != 'm' or source.get('height_semantics', 'above_ground') != 'above_ground':
        raise ValueError('건물 높이는 미터 단위의 지면 기준 높이여야 합니다. 해발고도와 구분하십시오.')
    center = meta.get('center_lonlat')
    if center is not None:
        if (not isinstance(center, list) or len(center) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in center)
            or not (-180 <= center[0] <= 180 and -90 <= center[1] <= 90)):
            raise ValueError('center_lonlat은 [경도, 위도] 또는 null이어야 합니다.')
    buildings = scene.get('buildings')
    zones = scene.get('no_fly_zones')
    if not isinstance(buildings, list) or not isinstance(zones, list):
        raise ValueError('buildings와 no_fly_zones 배열이 필요합니다.')
    ids = set()
    fingerprints = []
    imputed = 0
    aoi = box(-200, -200, 200, 200)
    for b in buildings:
        ident = b.get('building_id')
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError('건물 ID는 비어 있지 않은 고유 문자열이어야 합니다.')
        ids.add(ident)
        height = b.get('height_m')
        if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
            raise ValueError(f'{ident}: 0·음수·미확인 높이를 자유공간으로 취급하지 않습니다.')
        if not isinstance(b.get('height_source'), str) or not b['height_source']:
            raise ValueError(f'{ident}: height_source가 필요합니다. 추정 높이는 imputed로 표시하십시오.')
        imputed += b['height_source'] == 'imputed'
        g = shape(b.get('geometry_local_m', {}))
        if g.geom_type not in ('Polygon', 'MultiPolygon') or g.is_empty or not g.is_valid:
            raise ValueError(f'{ident}: 유효한 건물 Polygon/MultiPolygon이 필요합니다.')
        coordinates = shapely.get_coordinates(g, include_z=True)
        if not all(math.isfinite(float(x)) for x in coordinates[:, :2].flat) or shapely.has_z(g):
            raise ValueError(f'{ident}: 지붕 높이를 좌표 Z로 넣지 말고 height_m에 따로 넣으십시오.')
        if not g.intersects(aoi):
            raise ValueError(f'{ident}: 400m 관심영역과 교차하지 않는 건물입니다.')
        fingerprints.append([shapely.normalize(g).wkb_hex, float(height)])
    for zone in zones:
        g = shape(zone.get('geometry_local_m', {}))
        if g.geom_type not in ('Polygon', 'MultiPolygon') or g.is_empty or not g.is_valid:
            raise ValueError('유효하지 않은 no_fly_zones 기하입니다.')
    policy = meta.get('flight_policy', {})
    if policy.get('flight_altitude_m') == 50 and policy.get('safety_margin_m') == 5:
        expected = shapely.union_all([shape(b['geometry_local_m']).buffer(5).intersection(aoi)
                                     for b in buildings if b['height_m'] >= 50])
        delivered = shapely.union_all([shape(z['geometry_local_m']) for z in zones])
        # 캐시된 비행 금지 구역이 실제 높이·버퍼 규칙과 다른 경우 조용히 재사용하지 않는다.
        # 팀 지도는 원본 전체 외곽선을 먼저 버퍼한 뒤 AOI로 자른다.
        # scene의 이미 잘린 외곽선을 버퍼하면 경계 5m 띠 안에서 조금 작아질 수 있다.
        tolerance = max(0.1, expected.area * 1e-5)
        missing = expected.difference(delivered).area
        extra = delivered.difference(expected)
        full_before_clip = policy.get('geometry_policy', '').startswith('Buffer original full')
        invalid_extra = (extra.difference(aoi.boundary.buffer(5.00001)).area
                         if full_before_clip else extra.area)
        if missing > tolerance or invalid_extra > tolerance:
            raise ValueError('전달된 no_fly_zones가 높이≥50m·5m 버퍼와 맞지 않습니다. 지도 생성 설정을 확인하십시오.')
    fingerprint = hashlib.sha256(json.dumps(sorted(fingerprints), separators=(',', ':')).encode()).hexdigest()
    return {'key': key, 'name': meta['region_name'], 'path': str(path),
            'sha256': sha256_file(path), 'geometry_sha256': fingerprint,
            'buildings': len(buildings), 'imputed_buildings': imputed,
            'center_lonlat': center, 'is_demo': bool(meta.get('is_demo', False)),
            'warnings': ([] if meta.get('building_source') or meta.get('dataset_provenance') else
                         ['건물 원자료 출처가 없습니다. 실제 도시 자료라는 주장을 하지 마십시오.'])}


@dataclass(frozen=True)
class Region:
    key: str
    path: Path
    audit: dict


def load_regions(registry=DEFAULT_REGISTRY):
    registry = Path(registry).resolve()
    data = read_json(registry)
    if set(data) != {'format', 'regions'} or data['format'] != 'region-catalog-v1' or not isinstance(data['regions'], dict):
        raise ValueError('region-catalog-v1 등록 목록이 필요합니다.')
    out = {}
    for key, entry in data['regions'].items():
        if set(entry) != {'path', 'sha256'}:
            raise ValueError(f'{key}: path와 sha256만 등록하십시오.')
        path = (registry.parent / entry['path']).resolve()
        audit = validate_scene(path)
        if key != audit['key'] or entry['sha256'] != audit['sha256']:
            raise ValueError(f'{key}: 등록 이름 또는 지도 해시가 다릅니다. 변경을 확인한 뒤 register --update 하십시오.')
        out[key] = Region(key, path, audit)
    return out


def register_scene(path, registry=DEFAULT_REGISTRY, update=False):
    registry = Path(registry).resolve()
    audit = validate_scene(path)
    data = read_json(registry) if registry.exists() else {'format': 'region-catalog-v1', 'regions': {}}
    if data.get('format') != 'region-catalog-v1' or not isinstance(data.get('regions'), dict):
        raise ValueError('등록 목록 형식 오류')
    key = audit['key']
    entry = {'path': Path(os.path.relpath(Path(path).resolve(), registry.parent)).as_posix(), 'sha256': audit['sha256']}
    if key in data['regions'] and data['regions'][key] != entry and not update:
        raise ValueError(f'{key}가 이미 등록되었습니다. 교체하려면 --update가 필요합니다.')
    data['regions'][key] = entry
    write_json(registry, data)
    return audit
