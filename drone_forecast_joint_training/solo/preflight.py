"""Kinematic path feasibility over training/validation maps; no RF policy/test rollout."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import json
from solo.contracts import ROOT,load_plan,ensure_map,atomic_json,object_hash

_STOP=None

def initialize(stop):
    global _STOP
    _STOP=stop


def check(entry):
    ensure_map(entry,_STOP)
    results={}
    for direction in ['east_west','north_south']:
        row=json.loads((ROOT/'cache/usable'/f'{entry["key"]}_{direction}_motion.json').read_text())
        assert row['mission_completed'],(entry['key'],direction)
        results[direction]=dict(completed=True,duration_s=row.get('duration_s'))
    return dict(region=entry['key'],split=entry['joint_split'],directions=results)


def run(workers=4):
    if workers<1:raise ValueError('workers must be positive')
    plan=load_plan();entries=[m for m in plan['maps'] if m['joint_split'] in ['train','validation']]
    ctx=mp.get_context('spawn');stop=ctx.Event();rows=[]
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize,initargs=(stop,))
    futures=[]
    try:
        futures=[pool.submit(check,m) for m in entries]
        for f in as_completed(futures):
            rows.append(f.result())
            if len(rows)%10==0 or len(rows)==len(entries):print(json.dumps(dict(stage='kinematic_preflight',complete=len(rows),total=len(entries))),flush=True)
    except BaseException:
        stop.set()
        for f in futures:f.cancel()
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    result=dict(passed=True,plan_sha256=object_hash(plan),maps=len(rows),direction_checks=len(rows)*2,
        final_test_maps_accessed=0,scope='Full nominal kinematic paths only; no RF/jamming, policy validation, training or final-test outcomes',
        rows=sorted(rows,key=lambda r:r['region']))
    atomic_json(ROOT/'results/transformer/verification/map_preflight.json',result)
    return result
