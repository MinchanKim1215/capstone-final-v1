"""Canonical PUBLIC geometry descriptors when registry summaries are absent."""
from functools import lru_cache
import numpy as np
from shapely.geometry import shape,box
from shapely.ops import unary_union
from . import ARTIFACTS,LEGACY
from .contracts import atomic_json,digest,entry_for,read_json,sha


@lru_cache(maxsize=1)
def feature_code():
    return sha(__file__)


def descriptors(entry):
    path=ARTIFACTS/'cache/public_maps'/(entry['sha256']+'_'+feature_code()[:12]+'.json')
    if path.exists():
        item=read_json(path)
        if item['feature_code_sha256']!=feature_code() or item['map_sha256']!=entry['sha256'] or digest(item['values'])!=item['values_sha256']:
            raise ValueError('Public feature cache integrity failure')
        return item['values']
    # Protect accidental test-map geometry access before the final-test seal.
    canonical=entry_for(entry['key'])
    scene=read_json(LEGACY/canonical['path'])
    buildings=scene['buildings']
    if not buildings:raise ValueError('Empty building catalogue is unsupported')
    geometry=[shape(b['geometry_local_m']) for b in buildings]
    heights=np.asarray([b['height_m'] for b in buildings],float)
    if not np.isfinite(heights).all() or (heights<=0).any() or any(g.is_empty or not g.is_valid for g in geometry):
        raise ValueError('Invalid delivered building geometry/heights')
    bounds=scene['metadata']['local_bounds']
    if bounds!=[-200,-200,200,200]:raise ValueError('Unexpected static map extent')
    aoi=box(*bounds)
    policy=scene['metadata'].get('flight_policy',{})
    if policy.get('flight_altitude_m')==50 and policy.get('safety_margin_m')==5:
        zones=[shape(z['geometry_local_m']) for z in scene['no_fly_zones']]
    else:
        zones=[g.buffer(5).intersection(aoi) for g,h in zip(geometry,heights) if h>=50]
    fraction=float(unary_union(zones).intersection(aoi).area/aoi.area)
    blocker=unary_union(zones)
    footprint=unary_union(geometry)
    cells=[box(-200+x*50,-200+y*50,-150+x*50,-150+y*50) for y in range(8) for x in range(8)]
    nofly_grid=[float(blocker.intersection(c).area/2500) for c in cells]
    footprint_grid=[float(footprint.intersection(c).area/2500) for c in cells]
    imputed=sum(b['height_source']=='imputed' for b in buildings)
    values=dict(buildings=len(buildings),tall_buildings=int((heights>=50).sum()),
                no_fly_area_fraction=fraction,nonimputed_heights=len(buildings)-imputed,imputed_heights=imputed,
                height_statistics_scaled=[float(heights.mean()/200),float(heights.std()/200),float(heights.max()/200)],
                nofly_grid_8x8=nofly_grid,footprint_grid_8x8=footprint_grid)
    if not 0<=fraction<=1+1e-8:raise ValueError('Invalid flight exclusion area')
    atomic_json(path,dict(map_sha256=entry['sha256'],feature_code_sha256=feature_code(),
                         values=values,values_sha256=digest(values)))
    return values
