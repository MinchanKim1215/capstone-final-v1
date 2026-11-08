"""Evaluate the independent last condition in isolated storage."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import multiprocessing as mp
import os
import shutil

from adaptive_joint_expanded.contracts import lock
from forecast_rebuild.contracts import now, read, sha, write
from fast_halfsec import ARTIFACTS
from fast_halfsec import rollouts
from fast_halfsec.protocol import PLAN, verify

STAGING = ARTIFACTS / 'parallel_tail'


def worker_init():
    rollouts.ARTIFACTS = STAGING


def requests():
    plan = read(PLAN)
    verify(plan)
    freeze = read(ARTIFACTS / 'freeze.json')
    opened = read(ARTIFACTS / 'confirmation_open.json')
    if opened['freeze_sha256'] != sha(ARTIFACTS / 'freeze.json'):
        raise ValueError('Changed confirmation binding')
    spec, = (value for value in plan['confirmation']
             if value['region']=='gyeonggi_20261031_b2_05' and value['kind']=='continuous')
    return [rollouts.job('confirmation',spec,mode,
                         freeze['forecast'] if mode in ('learned_05s','ppo_05s') else None,
                         freeze['actor'] if mode=='ppo_05s' else None,collect=mode=='physics_05s')
            for mode in plan['modes']]


def stage():
    STAGING.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ARTIFACTS/'confirmation_open.json',STAGING/'confirmation_open.json')
    pending = [request for request in requests() if rollouts.cached(request) is None]
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=min(5,len(pending)),mp_context=mp.get_context('spawn'),
                             initializer=worker_init) as pool:
        tasks = [pool.submit(rollouts.one,request) for request in pending]
        for task in as_completed(tasks):
            task.result()
    write(STAGING/'staged.json',dict(created_at=now(),modes=[row['mode'] for row in pending],
                                     freeze_sha256=sha(ARTIFACTS/'freeze.json')))


def merge():
    # The primary campaign must have released its lock before any promotion.
    with lock(ARTIFACTS/'campaign.lock'):
        promoted = []
        for request in requests():
            if rollouts.cached(request) is not None:
                continue
            target_meta,target_archive = rollouts.record_paths(request)
            relative = target_meta.relative_to(ARTIFACTS)
            staged_meta = STAGING/relative
            if not staged_meta.exists():
                continue
            row = read(staged_meta)
            source = Path(row['archive_path'])
            if not source.is_relative_to(STAGING) or row['contract'] != request or sha(source)!=row['archive_sha256']:
                raise ValueError('Staged rollout does not match its frozen request')
            target_archive.parent.mkdir(parents=True,exist_ok=True)
            temporary = target_archive.with_suffix('.promote.npz')
            shutil.copyfile(source,temporary)
            os.replace(temporary,target_archive)
            row['archive_path'] = str(target_archive)
            write(target_meta,row)
            if rollouts.cached(request) is None:
                raise ValueError('Promoted rollout did not pass the checksum check')
            promoted.append(request['mode'])
        write(STAGING/'promoted.json',dict(created_at=now(),modes=promoted))
        print(dict(promoted=promoted),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step',choices=('stage','merge'))
    {'stage':stage,'merge':merge}[parser.parse_args().step]()
