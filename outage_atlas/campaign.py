"""Resume per-map missions and verify every output before aggregation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

import numpy as np

from . import ARTIFACTS, ROOT
from .protocol import prepare, read, sha, status, now, write


def initialize():
    from .runner import initialize as load
    load()


def record(case):
    from .runner import run
    path = ARTIFACTS/'maps'/(case['spec']['region']+'.json')
    archive = path.with_suffix('.npz')
    contract = dict(case=case, protocol_sha256=sha(ARTIFACTS/'protocol.json'))
    if path.exists():
        old = read(path)
        if old['contract'] != contract or old['archive_sha256'] != sha(archive):
            raise ValueError('Completed map was modified: ' + str(path))
    else:
        result, evidence = run(case['spec'])
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = archive.with_suffix('.pending.npz')
        np.savez_compressed(pending, **evidence)
        pending.replace(archive)
        write(path, dict(result, contract=contract, archive_path=str(archive), archive_sha256=sha(archive)))
    return dict(path=str(path), sha256=sha(path), region=case['spec']['region'], split=case['split'])


def preflight():
    from .radio import minimal_masks
    from .runner import run
    initialize()
    old = read(ROOT.parent/'forecast_prevention/artifacts/protocol.json')['archived_records'][0]['source_record']
    if sha(old['array_path']) != old['array_sha256']:
        raise ValueError('Preflight archive changed')
    row, trace = run(old['spec'], max_steps=18)
    with np.load(old['array_path'], allow_pickle=False) as raw:
        for key in ('nodes', 'commands', 'links'):
            value = trace['actual_links' if key=='links' else key]
            np.testing.assert_array_equal(value, raw[key][:len(value)])
    assert minimal_masks([1, 2, 3, 6]) == [1, 2]
    write(ARTIFACTS/'preflight.json', dict(passed=True, archive_steps=18, identical_motion_and_links=True,
          identical_native_replay_packets=row['native_packet_exact'], phases_aligned=True,
          three_regimes_replayed=len(row['rf_cases'])==3, checked_at=now()))


def all_records(plan):
    path = ARTIFACTS/'manifest.json'
    if path.exists():
        old = read(path)
        if old['protocol_sha256'] != sha(ARTIFACTS/'protocol.json'):
            raise ValueError('Manifest is from another protocol')
        return old['records']
    cases = plan['cases']
    marker = ARTIFACTS/'diagnostic_access.json'
    if not marker.exists():
        write(marker, dict(protocol_sha256=sha(ARTIFACTS/'protocol.json'), opened_at=now(),
                           former_test_maps=32, geographic_quarantine_maps=1,
                           use='User requested every map. Diagnosis only; no model fitting or tuning.'))
    results = [None]*len(cases)
    status('full_missions_and_matched_rf_replays', complete=0, total=len(cases))
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        pending = {pool.submit(record, case): i for i,case in enumerate(cases)}
        for count, future in enumerate(as_completed(pending), 1):
            index = pending[future]
            try:
                results[index] = future.result()
            except Exception as error:
                raise RuntimeError('Map failed: ' + cases[index]['spec']['region']) from error
            status('full_missions_and_matched_rf_replays', complete=count, total=len(cases),
                   last_completed=results[index]['region'])
            if count % 5 == 0 or count == len(cases):
                print('Maps', count, '/', len(cases), flush=True)
    write(path, dict(records=results, protocol_sha256=sha(ARTIFACTS/'protocol.json'), created_at=now()))
    return results


def main():
    plan = prepare()
    final = ARTIFACTS/'completed.json'
    if final.exists():
        if sha(ROOT/'RESULTS.md') != read(final)['report_sha256']:
            raise ValueError('Completed report changed')
        print('Completed. See', ROOT/'RESULTS.md', flush=True)
        return
    if not (ARTIFACTS/'preflight.json').exists():
        status('native_preflight')
        preflight()
    records = all_records(plan)
    status('auditing_and_reporting', complete=len(records), total=len(plan['cases']))
    from .report import finish
    finish(plan, records)
    write(final, dict(created_at=now(), report_sha256=sha(ROOT/'RESULTS.md'),
                      map_index_sha256=sha(ROOT/'MAPS.md'), summary_sha256=sha(ARTIFACTS/'summary.json'),
                      manifest_sha256=sha(ARTIFACTS/'manifest.json'), access_sha256=sha(ARTIFACTS/'diagnostic_access.json')))
    status('audited_and_reported', state='completed', complete=len(records), total=len(records),
           replayed_rf_cases=len(records)*3, report=str(ROOT/'RESULTS.md'))


if __name__=='__main__':
    import ctypes
    from adaptive_joint_expanded.contracts import lock
    with lock(ARTIFACTS/'campaign.lock'):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            main()
        except BaseException as error:
            status('campaign', state='failed', error=repr(error))
            traceback.print_exc()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
