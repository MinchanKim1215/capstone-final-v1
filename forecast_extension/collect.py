"""Fresh seeds/directions; all development maps include every RF condition."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import ctypes
import json
import multiprocessing as mp
import msvcrt
import os
from . import ARTIFACTS
from .contracts import prepare,read,write,sha,digest,now,maps,folds,SEED,BASELINE,PARENT_ARTIFACTS


def requests():
    old=read(PARENT_ARTIFACTS/'collection_plan.json');old_direction={s['region']:s['direction'] for s in old['specs']}
    registry=maps();train=sorted((k for k,m in registry.items() if m['joint_split']=='train'),key=lambda k:digest(['v9-order',SEED,k]))
    deployments=('v8_s3_r5_40','cross6_s4_r2_20','cross10_s6_r4_20','v6_s4_r2_40');rows=[]
    for i,key in enumerate(train):
        seed=int(digest(['v9-train-seed',SEED,key])[:8],16)%(2**31-3000010)
        direction=('east_west','north_south')[int(digest(['v9-direction',key])[:8],16)%2]
        rows.append(dict(role='train',spec=dict(region=key,direction=direction,kind=('clear','intermittent','continuous')[i%3],seed=seed),forced=deployments[i%4]))
    for split,regions in folds().items():
        for key in regions:
            seed=int(digest(['v9-development-seed',SEED,key])[:8],16)%(2**31-3000010)
            direction='north_south' if old_direction[key]=='east_west' else 'east_west'
            for kind in ('clear','intermittent','continuous'):
                rows.append(dict(role=split,spec=dict(region=key,direction=direction,kind=kind,seed=seed),forced=None))
    if len(rows)!=377:raise ValueError('Unexpected collection size')
    return rows


def main():
    prepare()
    from adaptive_joint_expanded.experiment import _initialize,_worker
    from adaptive_joint_expanded import ARTIFACTS as OLD_ARTIFACTS
    prior=read(OLD_ARTIFACTS/'protocol.json');folder=ARTIFACTS/'raw/fresh';folder.mkdir(parents=True,exist_ok=True)
    rows=requests();write(ARTIFACTS/'collection_plan.json',dict(rows=rows,created_at=now(),baseline_sha256=sha(BASELINE),confirmation_outcomes_used=False))
    lock=open(ARTIFACTS/'collection.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));saved={};status=ARTIFACTS/'collection_status.json'
    tasks=[]
    for i,row in enumerate(rows):
        spec=row['spec'];prefix=folder/f'{i:04d}_{row["role"]}_{spec["region"]}_{spec["kind"]}'
        tasks.append((spec,prior['config'],row['forced'],True,None,str(prefix)))
    try:
        write(status,dict(state='running',complete=0,total=len(rows),pid=os.getpid(),started_at=now()))
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=_initialize,initargs=(str(BASELINE),prior)) as pool:
            pending={pool.submit(_worker,task):i for i,task in enumerate(tasks)}
            for future in as_completed(pending):
                i=pending[future];item=future.result();path=Path(tasks[i][-1]).with_suffix('.npz');meta=path.with_suffix('.json')
                if sha(path)!=item['array_sha256']:raise ValueError('Fresh rollout hash changed')
                saved[i]=dict(role=rows[i]['role'],region=item['region'],spec=item['spec'],deployment=item['deployment'],n_search=item['search_drones'],
                    source_kind='fresh_v9',key=sha(meta)[:20],metadata_path=str(meta),array_path=str(path),array_sha256=item['array_sha256'])
                write(ARTIFACTS/'collection_manifest.json',dict(rows=[saved[j] for j in sorted(saved)],created_at=now(),confirmation_prediction_evaluated=False))
                write(status,dict(state='running',complete=len(saved),total=len(rows),last_role=rows[i]['role'],last_region=item['region'],updated_at=now(),system_sleep_inhibited=held))
                print(json.dumps(dict(complete=len(saved),total=len(rows),role=rows[i]['role'],region=item['region'])),flush=True)
        write(status,dict(state='completed',complete=len(rows),total=len(rows),ended_at=now(),all_source_hashes_verified=True))
    except BaseException as error:
        write(status,dict(state='failed',error=repr(error),updated_at=now()));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
