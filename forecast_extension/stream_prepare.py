"""Prepare each committed non-confirmation flight while collection continues."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import time
from . import ARTIFACTS
from .contracts import read,write,now
from .data import compile_record,plan_record,training_layout
from .belief import create


def one(record):
    record=compile_record(record);variants=[]
    if record['role']=='train':
        rows,arguments=training_layout(record);plan_record(record,rows)
        for argument in arguments:variants.append(create(*argument))
    else:plan_record(record,None)
    return dict(record=record,variants=variants)


def main():
    completed={};pending={};status=ARTIFACTS/'stream_preparation_status.json';seen=set()
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        while True:
            path=ARTIFACTS/'collection_manifest.json'
            if path.exists():
                for record in read(path)['rows']:
                    if record['role']=='confirmation' or record['key'] in seen:continue
                    seen.add(record['key']);pending[pool.submit(one,record)]=record['key']
            if pending:
                ready,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for future in ready:
                    key=pending.pop(future);completed[key]=future.result()
                    if len(completed)%8==0 or len(completed)==335:
                        write(status,dict(state='running',complete=len(completed),total=335,pending=len(pending),updated_at=now()))
                        print('Prepared',len(completed),'of 335',flush=True)
            else:time.sleep(2)
            collection=ARTIFACTS/'collection_status.json'
            if collection.exists():
                state=read(collection)
                if state['state']=='failed':raise RuntimeError('Collection failed; source logs preserved')
                if state['state']=='completed' and not pending:
                    if len(completed)!=335:raise ValueError('Prepared cohort is incomplete')
                    break
    write(ARTIFACTS/'stream_preparation_manifest.json',dict(rows=list(completed.values()),created_at=now()))
    write(status,dict(state='completed',complete=335,total=335,updated_at=now()))


if __name__=='__main__':main()
