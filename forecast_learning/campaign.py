"""Extract training examples, fit models, and test real paired prevention."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

import numpy as np

from . import ARTIFACTS, ROOT
from .protocol import V20, now, prepare, read, sha, status, write


def initialize():
    from forecast_prevention.runner import initialize as load
    load()


def replay_source(scope,item):
    if scope=='train':
        if sha(item['path'])!=item['sha256']:
            raise ValueError('Original training metadata changed')
        row = read(item['path'])
        return row['spec'],row['archive_path'],row['archive_sha256'],'actual_links',row['mission']
    return item['spec'],item['array_path'],item['array_sha256'],'links',read(item['metadata_path'])


def trial(scope,item):
    from .runner import run
    archive = None
    source_sha = None
    if scope in ('train','calibration','selection'):
        spec,archive,source_sha,link_key,expected = replay_source(scope,item)
    else:
        spec = item
    filename = f"{spec['region']}_{spec['kind']}.json"
    path = ARTIFACTS/scope/filename
    array = path.with_suffix('.npz')
    control = scope in ('v18_control','learned_control')
    contract = dict(spec=spec,scope=scope,source=source_sha,protocol_sha256=sha(ARTIFACTS/'protocol.json'))
    if control:
        contract['freeze_sha256'] = sha(ARTIFACTS/'freeze.json')
    if path.exists():
        row = read(path)
        if row['contract']!=contract or sha(array)!=row['archive_sha256']:
            raise ValueError('Existing flight result changed')
    else:
        if archive is not None:
            if sha(archive)!=source_sha:
                raise ValueError('Original command archive changed')
            with np.load(archive,allow_pickle=False) as raw:
                row,data = run(spec,reference=raw['commands'],diagnosis=False)
                for key in ('nodes','commands','links'):
                    np.testing.assert_array_equal(data[key],raw[link_key if key=='links' else key])
            for key in ('generated_packets','delivered_packets','timely_pdr','retries','beacon_rx'):
                if row[key]!=expected[key]:
                    raise ValueError('Original packet or reception outcome changed: '+key)
        elif scope=='v18_control':
            from forecast_avoidance.runner import run as existing_control
            row,data = existing_control(spec,'joint_geometry')
        elif scope=='learned_control':
            from .control import load_profile
            row,data = run(spec,arm='learned',profile=load_profile(),diagnosis=False)
        else:
            row,data = run(spec)
        path.parent.mkdir(parents=True,exist_ok=True)
        if archive is not None:
            data = {key:data[key] for key in ('network','x','scores')}
        pending = array.with_suffix('.pending.npz')
        np.savez_compressed(pending,**data)
        pending.replace(array)
        write(path,dict(row,contract=contract,archive_path=str(array),archive_sha256=sha(array)))
    return dict(path=str(path),sha256=sha(path),region=spec['region'])


def group(task):
    return [trial(scope,item) for scope,item in task]


def grouped(scope,items):
    by_map = defaultdict(list)
    for item in items:
        region = item['region'] if scope=='train' else item.get('spec',item)['region']
        by_map[region].append((scope,item))
    return list(by_map.values())


def jobs(name,batches):
    path = ARTIFACTS/(name+'_manifest.json')
    if path.exists():
        value = read(path)
        if value['protocol_sha256']!=sha(ARTIFACTS/'protocol.json'):
            raise ValueError('Cached cohort came from another protocol')
        return value['records']
    total = sum(len(batch) for batch in batches)
    status(name,complete=0,total=total)
    rows = {}
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures = {pool.submit(group,batch):i for i,batch in enumerate(batches)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            rows[i] = future.result()
            done += len(rows[i])
            status(name,complete=done,total=total)
            if done%5==0 or done==total:
                print(name,done,'/',total,flush=True)
    output = [row for i in sorted(rows) for row in rows[i]]
    write(path,dict(records=output,protocol_sha256=sha(ARTIFACTS/'protocol.json'),created_at=now()))
    return output


def preflight():
    from .runner import run
    initialize()
    ref = read(read(V20/'artifacts'/'fresh_manifest.json')['rows'][0]['path'])
    if sha(ref['archive_path'])!=ref['archive_sha256']:
        raise ValueError('Old validation trace changed')
    row,data = run(ref['spec'],max_steps=24)
    with np.load(ref['archive_path'],allow_pickle=False) as old:
        for key in ('nodes','commands','links'):
            np.testing.assert_array_equal(data[key],old[key][:len(data[key])])
        old_scores = old['scores']
        mask = (old_scores[:,0]<row['duration_s']) & (old_scores[:,0]%3==0)
        np.testing.assert_allclose(data['scores'],old_scores[mask][:,[0,3,4,1]],rtol=1e-12,atol=1e-12,equal_nan=True)
    write(ARTIFACTS/'preflight.json',dict(passed=True,original_trace_exact=True,original_v20_scores_exact=True,
          forecast_rows=len(data['scores']),first_segments_verified=row['exact_endpoints'],created_at=now()))


def main():
    plan = prepare()
    if (ARTIFACTS/'completed.json').exists():
        if sha(ROOT/'RESULTS.md')!=read(ARTIFACTS/'completed.json')['report_sha256']:
            raise ValueError('Completed study report changed')
        return
    if not (ARTIFACTS/'preflight.json').exists():
        status('preflight')
        preflight()
    development = dict(train=jobs('train',grouped('train',plan['train'])))
    for scope,items in plan['development'].items():
        development[scope] = jobs(scope,grouped(scope,items))
    from .training import fit_and_lock, verify_freeze
    freeze = fit_and_lock(development)
    verify_freeze(freeze)
    access = ARTIFACTS/'fresh_access.json'
    if not access.exists():
        write(access,dict(freeze_sha256=sha(ARTIFACTS/'freeze.json'),opened_at=now()))
    elif read(access)['freeze_sha256']!=sha(ARTIFACTS/'freeze.json'):
        raise ValueError('Candidate changed after confirmation access')
    new = plan['fresh_specs']
    forecast = jobs('fresh',grouped('fresh',new))
    control = defaultdict(list)
    for spec in new:
        for scope in ('v18_control','learned_control'):
            control[spec['region']].append((scope,spec))
    loops = jobs('control',list(control.values()))
    verify_freeze(freeze)
    from .report import finish
    status('final_audit_and_report')
    finish(plan,freeze,forecast,loops)
    write(ARTIFACTS/'completed.json',dict(report_sha256=sha(ROOT/'RESULTS.md'),created_at=now(),
                                        freeze_sha256=sha(ARTIFACTS/'freeze.json'),
                                        results_sha256=sha(ARTIFACTS/'final_results.json')))
    status('audited_and_reported',state='completed',fresh_forecasts=len(forecast),control_flights=len(loops))


if __name__=='__main__':
    import ctypes
    from adaptive_joint_expanded.contracts import lock
    with lock(ARTIFACTS/'campaign.lock'):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            main()
        except BaseException as error:
            status('campaign',state='failed',error=repr(error))
            traceback.print_exc()
            raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
