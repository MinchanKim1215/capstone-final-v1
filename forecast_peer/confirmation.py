"""Shared blind42cohort, declared before any predictive outcomes existed."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import time
from . import ARTIFACTS,GRAPH
from .contracts import read,write,sha,now,status
from .data import prepare_one

def prepare_case(item):
    parent=dict(key=item['key'],role='confirmation',region=item['region'],source_record=item['source_record'],path=item['legacy_path'],sha256=item['legacy_sha256'])
    result=prepare_one(parent)
    return dict(result,reference_path=item['reference_path'],reference_sha256=item['reference_sha256'],v14_data_path=item['path'],v14_data_sha256=item['sha256'])

def main():
    from .evaluate import verify_freeze
    f=verify_freeze();out=ARTIFACTS/'confirmation_data_manifest.json'
    if out.exists():return out
    source=GRAPH/'confirmation_data_manifest.json';began=time.monotonic()
    while not source.exists():
        if (GRAPH/'status.json').exists() and read(GRAPH/'status.json').get('state')=='failed':raise RuntimeError('Shared source study failed')
        if time.monotonic()-began>2*3600:raise TimeoutError('Shared cohort timeout')
        time.sleep(5)
    protocol=read(ARTIFACTS/'protocol.json')
    write(ARTIFACTS/'confirmation_access.json',dict(opened_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),protocol_sha256=sha(ARTIFACTS/'protocol.json'),
        shared_v14_manifest_sha256=sha(source),plan_registered_at=protocol['created_at'],raw_generation_before_V15freeze=True,no_new_V15flight_generation_claim=True,
        models_selected_without_confirmation=True,root_predictive_outcomes_blind_until_own_freeze=True,original_final_test_opened=False))
    items=read(source)['records'];done={};status('blind_shared42_peer_features',complete=0,total=len(items))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        jobs={pool.submit(prepare_case,i):k for k,i in enumerate(items)}
        for future in as_completed(jobs):done[jobs[future]]=future.result();status('blind_shared42_peer_features',complete=len(done),total=len(items))
    verify_freeze();write(out,dict(records=[done[k] for k in sorted(done)],created_at=now(),source_manifest=str(source),source_manifest_sha256=sha(source),
        freeze_sha256=sha(ARTIFACTS/'freeze.json'),shared_blind_cohort=True,known14maps=True,original_final_test_opened=False));return out

if __name__=='__main__':main()
