"""Fixed-policy development data collection, isolated from earlier runs."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import ctypes
import json
import multiprocessing as mp
import msvcrt
import os
import time
from . import ARTIFACTS
from .contracts import BASELINE, SEED, prepare, maps, folds, digest, write, now


def requests():
    registry=maps();partition=folds();specs=[];forced=[];roles=[]
    # The first results cover all development folds. Confirmation is collected,
    # but predictor evaluation on it is forbidden until a separate model freeze.
    ordered=[]
    for i in range(max(map(len,partition.values()))):
        for fold in ('selection','calibration','confirmation'):
            if i<len(partition[fold]):ordered.append((fold,partition[fold][i],None))
    added=sorted(k for k,v in registry.items() if v['joint_split']=='train' and v.get('source_dataset','legacy_v6')!='legacy_v6')
    choices=('v8_s3_r5_40','cross6_s4_r2_20','cross10_s6_r4_20','v6_s4_r2_40')
    ordered.extend(('train',key,choices[i%4]) for i,key in enumerate(added))
    original=sorted(k for k,v in registry.items() if v['joint_split']=='train' and v.get('source_dataset','legacy_v6')=='legacy_v6')
    original=sorted(original,key=lambda k:digest(['new-trajectory',SEED,k]))[:16]
    ordered.extend(('train',key,choices[(i+1)%4]) for i,key in enumerate(original))
    for i,(fold,key,choice) in enumerate(ordered):
        seed=int(digest(['collect',SEED,key])[:8],16)%(2**31-3000010)
        specs.append(dict(region=key,direction=('east_west','north_south')[i%2],
                          kind=('continuous','intermittent','clear')[i%3],seed=seed))
        forced.append(choice);roles.append(fold)
    return specs,forced,roles


def main():
    protocol=prepare()
    from adaptive_joint_expanded.experiment import _initialize,_worker
    from adaptive_joint_expanded import ARTIFACTS as UNUSED_PARENT_ARTIFACTS
    from . import ARTIFACTS as OWN_ARTIFACTS
    os.environ['NUMBA_CACHE_DIR']=str(OWN_ARTIFACTS/'numba_cache')
    parent=__import__('adaptive_joint_expanded.contracts',fromlist=['read_json'])
    old=parent.read_json(UNUSED_PARENT_ARTIFACTS/'protocol.json')
    folder=ARTIFACTS/'raw/current';folder.mkdir(parents=True,exist_ok=True)
    lock_path=ARTIFACTS/'collection.lock'
    lock=open(lock_path,'a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('This collection already has a live owner')
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    status=ARTIFACTS/'collection_status.json';started=now();clock=time.monotonic()
    specs,forced,roles=requests()
    write(ARTIFACTS/'collection_plan.json',dict(created_at=started,specs=specs,forced=forced,roles=roles,
        baseline_sha256=protocol['baseline']['sha256'],new_predictor_updates=False))
    try:
        write(status,dict(state='running',stage='prepare_cached_assets',pid=os.getpid(),started_at=started,
                           completed=0,total=len(specs),system_sleep_inhibited=held))
        # All 2,312 nominal trajectories were already verified in the parent.
        # Avoid loading 139 full city graphs serially just to check their paths.
        # Each World still performs its own map and nominal content checks.
        assets=parent.read_json(UNUSED_PARENT_ARTIFACTS/'assets_manifest.json')
        available={(row['region'],row['direction'],row['deployment']):row for row in assets['rows']}
        choices=('v8_s3_r5_40','cross6_s4_r2_20','cross10_s6_r4_20','v6_s4_r2_40')
        for spec in specs:
            for choice in choices:
                row=available[(spec['region'],spec['direction'],choice)]
                if not Path(row['path']).is_file():raise FileNotFoundError(row['path'])
        write(status,dict(state='running',stage='collect',pid=os.getpid(),started_at=started,
                           completed=0,total=len(specs),cached_assets_verified=True,system_sleep_inhibited=held))
        config=old['config'];tasks=[]
        for i,spec in enumerate(specs):
            prefix=folder/f'{i:04d}_{spec["region"]}_{forced[i] or "adaptive"}'
            tasks.append((spec,config,forced[i],True,None,str(prefix)))
        rows={};ctx=mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=_initialize,
                                 initargs=(str(BASELINE),old)) as pool:
            pending={pool.submit(_worker,t):i for i,t in enumerate(tasks)}
            for future in as_completed(pending):
                i=pending[future];row=future.result()
                # The lower-level worker returns the committed metadata. Its
                # public collect() wrapper normally attaches array_path later.
                row=dict(row,array_path=str(Path(tasks[i][-1]).with_suffix('.npz')))
                rows[i]=row
                write(ARTIFACTS/'collection_manifest.json',dict(rows=[dict(role=roles[j],metadata_path=str(Path(rows[j]['array_path']).with_suffix('.json')),
                    array_path=rows[j]['array_path'],array_sha256=rows[j]['array_sha256'],region=rows[j]['region']) for j in sorted(rows)]))
                record=dict(state='running',stage='collect',pid=os.getpid(),started_at=started,updated_at=now(),
                            completed=len(rows),total=len(specs),last_region=row['region'],last_role=roles[i],
                            system_sleep_inhibited=held,wall_seconds=round(time.monotonic()-clock,1))
                write(status,record)
                print(json.dumps(record,ensure_ascii=False),flush=True)
        write(status,dict(state='completed',completed=len(rows),total=len(specs),started_at=started,ended_at=now(),
                           wall_seconds=round(time.monotonic()-clock,1),original_models_modified=False,
                           final_test_opened=False,new_predictor_training_started=False))
    except BaseException as error:
        write(status,dict(state='failed',error=repr(error),updated_at=now(),original_models_modified=False))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
