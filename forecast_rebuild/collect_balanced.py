"""All three interference types per development map, before confirmation opens."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import ctypes
import json
import multiprocessing as mp
import msvcrt
import os
from . import ARTIFACTS
from .contracts import read,write,prepare,BASELINE,PARENT,now,role,sha


def main():
    if (ARTIFACTS/'confirmation_access.json').exists():raise ValueError('Do not redefine data after confirmation access')
    protocol=prepare();old=read(PARENT/'artifacts/protocol.json');plan=read(ARTIFACTS/'collection_plan.json')
    from adaptive_joint_expanded.experiment import _initialize,_worker
    folder=ARTIFACTS/'raw/validation_balanced';folder.mkdir(parents=True,exist_ok=True)
    tasks=[];roles=[]
    for spec,split in zip(plan['specs'],plan['roles']):
        if split=='train':continue
        for kind in ('clear','intermittent','continuous'):
            if kind==spec['kind']:continue
            new=dict(spec,kind=kind)
            prefix=folder/f'{split}_{spec["region"]}_{kind}'
            tasks.append((new,old['config'],None,True,None,str(prefix)));roles.append(split)
    lock=open(ARTIFACTS/'balanced_collection.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));rows={};started=now()
    status=ARTIFACTS/'balanced_collection_status.json'
    write(ARTIFACTS/'balanced_collection_plan.json',dict(created_at=started,tasks=[dict(spec=t[0],role=r) for t,r in zip(tasks,roles)],
        reason='Repair interference-type/fold confounding from the original round-robin schedule before confirmation is evaluated',
        final_cases_per_fold=dict(selection=45,calibration=45,confirmation=42),confirmation_previously_opened=False))
    try:
        write(status,dict(state='running',complete=0,total=len(tasks),pid=os.getpid(),started_at=started))
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=_initialize,
                                 initargs=(str(BASELINE),old)) as pool:
            pending={pool.submit(_worker,t):i for i,t in enumerate(tasks)}
            for future in as_completed(pending):
                i=pending[future];row=future.result();path=Path(tasks[i][-1]).with_suffix('.npz')
                if sha(path)!=row['array_sha256']:raise ValueError('Balanced source archive differs')
                rows[i]=dict(role=roles[i],region=row['region'],array_path=str(path),array_sha256=row['array_sha256'],
                              metadata_path=str(path.with_suffix('.json')))
                write(ARTIFACTS/'balanced_collection_manifest.json',dict(rows=[rows[j] for j in sorted(rows)]))
                write(status,dict(state='running',complete=len(rows),total=len(tasks),updated_at=now(),system_sleep_inhibited=held))
                print(json.dumps(dict(complete=len(rows),total=len(tasks))),flush=True)
        write(status,dict(state='completed',complete=len(rows),total=len(tasks),ended_at=now(),all_archive_hashes_verified=True))
    except BaseException as error:
        write(status,dict(state='failed',error=repr(error),updated_at=now()));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
