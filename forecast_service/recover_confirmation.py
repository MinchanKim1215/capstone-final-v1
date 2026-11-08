"""Recover post-flight logs without editing frozen code or regenerating flights.

The collector compared epsilon-tolerant deadline labels to a legacy strict
floating-point counter. Replaying the saved commands preserves both definitions
and lets the audit verify each separately. No model/threshold is refit.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from . import ARTIFACTS
from .contracts import read,write,sha,now,status,folds,BASELINE
from .evaluate import verify_freeze
from .confirmation import prepare_case
from .replay import replay
from .campaign import run


def recover_case(record):
    prepared=prepare_case(record);path=ARTIFACTS/'service_cache'/(record['key']+'.json')
    replayed=False
    if not path.exists():replay(prepared);replayed=True
    item=read(path)
    if sha(item['path'])!=item['sha256']:raise ValueError('Recovered packet record changed')
    return prepared,replayed


def main():
    verify_freeze();folder=ARTIFACTS/'recovery';folder.mkdir(exist_ok=True);records=[]
    for path in sorted((ARTIFACTS/'raw/confirmation').glob('*.json')):
        row=read(path);source=path.with_suffix('.npz')
        if sha(source)!=row['array_sha256'] or row['request']['checkpoint_sha256']!=sha(BASELINE) or not row['audit']['passed']:
            raise ValueError('Saved frozen-policy flight is not intact')
        records.append(dict(role='confirmation',region=row['region'],spec=row['spec'],deployment=row['deployment'],n_search=row['search_drones'],
            source_kind='fresh_v10',key=sha(path)[:20],metadata_path=str(path),array_path=str(source),array_sha256=row['array_sha256']))
    groups={region:sorted(r['spec']['kind'] for r in records if r['region']==region) for region in {r['region'] for r in records}}
    if len(records)!=42 or set(groups)!=set(folds()['confirmation']) or any(k!=['clear','continuous','intermittent'] for k in groups.values()):
        raise ValueError('Do not silently replace a missing confirmation flight')
    status('stage4_saved_flight_packet_recovery',complete=0,total=42);prepared={};replayed=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(recover_case,r):i for i,r in enumerate(records)}
        for future in as_completed(pending):
            record,used_replay=future.result();prepared[pending[future]]=record
            if used_replay:replayed.append(record['key'])
            status('stage4_saved_flight_packet_recovery',complete=len(prepared),total=42)
    verify_freeze();rows=[prepared[i] for i in sorted(prepared)]
    write(ARTIFACTS/'confirmation_dataset.json',dict(rows=rows,counts={'confirmation':42},created_at=now(),known_maps_only=True,
        freeze_sha256=sha(ARTIFACTS/'freeze.json'),original_final_test_opened=False,flight_regeneration=False))
    write(folder/'packet_log_recovery.json',dict(completed_at=now(),flights_preserved=42,missing_packet_logs_replayed=replayed,
        frozen_sources_unchanged=True,model_refitted=False,threshold_refitted=False,labels_changed=False,
        explanation='Original service labels already used deadline<=.25+1e-10. Legacy timely_pdr uses strict<=.25. These are checked separately; only post-flight log recovery changed.'))
    run('forecast_service.compare','comparison_resumed');run('forecast_service.post','post')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',stage='confirmation_recovery',error=repr(error),updated_at=now()));raise
