"""45지역 × 2방향의 전체 경로/수색 완료 및 연속 분리거리 검사."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
from swarm_sim.config import Design, Scenario
from swarm_sim.regions import ROOT, read_json, write_json
from swarm_sim.forecast import atomic_npz
from upgrade.collect import prepare_city_without_unused_radio_cache
from usable.motion import full_mission


def one(entry, design):
    city=prepare_city_without_unused_radio_cache(ROOT/entry['path'],Scenario(),ROOT/'cache/usable')
    output=[]
    for direction in ('east_west','north_south'):
        s=Scenario(duration_s=7200,motion_dt_s=3,direction=direction)
        city.route=city.make_sweep(s)
        t=time.perf_counter();m=full_mission(city,Design(**design),s)
        path=ROOT/'cache/usable'/f'{city.key}_{direction}_motion.npz'
        atomic_npz(path,**{k:m[k] for k in ('nodes','phase','desired')})
        meta={k:v for k,v in m.items() if k not in ('nodes','phase','desired')}
        meta.update(region=city.key,direction=direction,elapsed_s=time.perf_counter()-t,
                    unique_last_1200_s=len(np.unique(m['nodes'][-400:],axis=0)))
        write_json(path.with_suffix('.json'),meta)
        output.append(meta)
    return output


def main():
    p=read_json(ROOT/'configs/upgrade_protocol.json');rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(one,e,p['design']) for e in p['maps']]
        for f in as_completed(jobs):
            for m in f.result():
                rows.append(m);print(m['region'],m['direction'],m['mission_completed'],m['duration_s'],flush=True)
    write_json(ROOT/'results/usable/motion_check.json',{'paths':rows,'complete_count':sum(x['mission_completed'] for x in rows),
                'minimum_separation_m':min(x['minimum_swept_separation_m'] for x in rows)})


if __name__=='__main__':main()
