"""Extend frozen outage diagnostics to 30 s; never alter flight or RF outcomes."""
from pathlib import Path
import sys
import json
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from final_v1.causes import ATLAS, LEGACY, NAV_ARTIFACTS, radio_samples
from forecast_rebuild.contracts import maps, read, sha
from forecast_prevention.diagnostics import connected, outage_runs
from swarm_sim.config import Scenario
from usable.collect import prepare_city_without_unused_radio_cache
import numpy as np

OUT = ROOT / 'output/outage_history_20261105'
CAUSES = ROOT / 'final_v1/artifacts/causes.json'
ORDER = ['abrupt', 'gradual', 'mixed', 'other', 'insufficient_history']


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def process(task):
    entry, registry, prior = task
    if sha(entry['path']) != entry['sha256']:
        raise ValueError('Atlas metadata changed')
    meta = read(entry['path'])
    if sha(meta['archive_path']) != meta['archive_sha256']:
        raise ValueError('Flight archive changed')
    map_path = LEGACY / registry['path']
    if sha(map_path) != registry['sha256']:
        raise ValueError('Map changed')
    scenario = Scenario(direction=meta['spec']['direction'], motion_dt_s=3).validate()
    city = prepare_city_without_unused_radio_cache(map_path, scenario, NAV_ARTIFACTS/'cache/city')
    city.radio.enable_cache()
    with np.load(meta['archive_path'], allow_pickle=False) as data:
        links = data['actual_links']
        positions = city.points[data['nodes']]
        runs = outage_runs(connected(links, meta['mission']['search_drones']))
        if not runs:
            return dict(region=entry['region'], events=[], samples_verified=0,
                        archive_sha256=meta['archive_sha256'], metadata_sha256=entry['sha256'])
        frames = np.unique(np.concatenate([np.arange(max(0, start-60), start+1) for start, _ in runs]))
        margin, rebuilt = radio_samples(city, scenario, positions, frames, meta['spec']['seed'],
                                       meta['spec']['kind'], meta['jammer'], meta['mission']['search_drones'])
        np.testing.assert_array_equal(rebuilt, links[frames])
    if len(runs) != len(prior['events']):
        raise ValueError('Event count changed')
    rows = []
    for idx, ((start, stop), old) in enumerate(zip(runs, prior['events'])):
        if (old['onset_s'], old['end_s']) != (start*.5, stop*.5):
            raise ValueError('Event timing changed')
        f = np.arange(max(0, start-60), start+1)
        values = margin[np.searchsorted(frames, f)]
        rounded = np.round(values, 2)
        np.testing.assert_array_equal(rounded[-len(old['margin_db']):], old['margin_db'])
        if not start:
            continue
        previous = [[max(a, int(f[0]))*.5-start*.5, min(b, start)*.5-start*.5]
                    for a, b in runs[:idx] if b > f[0]]
        row = dict(region=entry['region'], kind=meta['spec']['kind'], seed=meta['spec']['seed'],
                   direction=meta['spec']['direction'], split=entry['split'],
                   onset_s=start*.5, end_s=stop*.5, duration_s=(stop-start)*.5,
                   pattern=old['pattern'], available_history_s=(start-f[0])*.5,
                   margin_db=np.round(values, 6).tolist(),
                   previous_persistent_outages_relative_s=previous,
                   repair_factors=old['repair_factors'])
        rows.append(row)
    return dict(region=entry['region'], events=rows, samples_verified=len(frames),
                archive_sha256=meta['archive_sha256'], metadata_sha256=entry['sha256'])


def point(event, lead):
    if event['available_history_s'] < lead:
        return None
    return event['margin_db'][-1-int(round(lead*2))]


def stats(events):
    value = {}
    for kind in ORDER:
        rows = [e for e in events if e['pattern'] == kind]
        complete = [e for e in rows if e['available_history_s'] >= 30]
        uninterrupted = [e for e in complete if not e['previous_persistent_outages_relative_s']]
        points = {}
        for lead in (30, 20, 10, 6, 3, 1, .5, 0):
            x = [point(e, lead) for e in rows]
            x = np.array([a for a in x if a is not None], float)
            points[str(lead)] = dict(n=len(x), median_db=float(np.median(x)) if len(x) else None,
                p10_db=float(np.quantile(x, .1)) if len(x) else None,
                p90_db=float(np.quantile(x, .9)) if len(x) else None,
                le_3db=int((x <= 3).sum()), le_6db=int((x <= 6).sum()),
                below_zero=int((x < 0).sum()))
        # Paired common cohort avoids changing the denominator across lead times.
        paired = {str(lead): float(np.median([point(e,lead) for e in complete])) if complete else None
                  for lead in (30,20,10,6,3,.5,0)}
        clean_points = {str(lead): float(np.median([point(e,lead) for e in uninterrupted])) if uninterrupted else None
                        for lead in (30,20,10,6,3,.5,0)}
        value[kind] = dict(n=len(rows), coverage={str(w):sum(e['available_history_s']>=w for e in rows)
                                               for w in (6,10,20,30)}, points=points,
                           complete30_n=len(complete), complete30_medians=paired,
                           no_prior_persistent_outage30_n=len(uninterrupted),
                           no_prior_persistent_outage30_medians=clean_points)
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    causes = read(CAUSES)
    manifest_path = ATLAS/'manifest.json'
    manifest = read(manifest_path)
    if sha(manifest_path) != causes['source_sha256']:
        raise ValueError('Frozen causes no longer match atlas')
    registry = maps()
    prior = {r['region']:r for r in causes['flights']}
    tasks = [(e, registry[e['region']], prior[e['region']]) for e in manifest['records']]
    flights = []
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn')) as pool:
        for i, row in enumerate(pool.map(process, tasks, chunksize=2),1):
            flights.append(row)
            if i % 30 == 0 or i == len(tasks):
                print(f'long-history maps {i}/{len(tasks)}', flush=True)
    events = [e for f in flights for e in f['events']]
    if len(events) != 2572 or len(flights) != 345:
        raise ValueError('Coverage changed')
    counts = Counter(e['pattern'] for e in events)
    if counts != Counter(causes['sensitivity']['pattern']):
        raise ValueError('Classification changed')
    old_examples = read(ROOT/'output/outage_patterns_20261105/evidence.json')['examples']
    examples = {k:next(e for e in events if e['region']==v['region'] and e['onset_s']==v['onset_s'])
                for k,v in old_examples.items()}
    payload = dict(created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), maps=345, new_events=len(events),
        sample_dt_s=.5, max_history_s=30, classification_history_s=6,
        measurement_scope='Exact replay of original simulated physical truth; not receiver observations',
        not_available='Native 0.1-second RF, beacon, packet, or control samples',
        source_causes_sha256=sha(CAUSES), source_manifest_sha256=sha(manifest_path),
        script_sha256=sha(Path(__file__)),
        verified_radio_frames=sum(f['samples_verified'] for f in flights),
        all_original_6s_samples_match=True, all_original_graph_samples_match=True,
        statistics=stats(events), examples=examples, flights=flights)
    path = OUT/'long_history.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(',',':'),allow_nan=False,default=scalar),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('flights','examples')},ensure_ascii=False,indent=2,default=scalar),flush=True)
    print('saved', path, sha(path), flush=True)


if __name__=='__main__':
    main()
