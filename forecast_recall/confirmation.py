"""Own42new fixed-policy flights, after own freeze; unchanged radio/control."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,digest,now,status,folds,SEED
from .data import prepare_one
from .references import prepare as reference
import forecast_navigation.confirmation as parent_capture
from forecast_navigation.wire import attach_navigation
from forecast_graph.confirmation import snapshot


def recover(record):
    from adaptive_joint_expanded.world import World
    meta=read(record['metadata_path']);maximum=meta['request'].get('max_steps')
    env=World(**record['spec'],deployment=record['deployment'],observer=None,max_steps=maximum);attach_navigation(env)
    with np.load(record['array_path'],allow_pickle=False) as a:commands=a['commands'];nodes=a['nodes'];links=a['links']
    for k,c in enumerate(commands):
        env.step(c);np.testing.assert_equal(env.nodes,nodes[k+1]);np.testing.assert_equal(env.channel_available[-6:],links[k*6:k*6+6])
    assert env.done
    for key in ('generated_packets','delivered_packets','retries','ttl_drops','retry_drops','overflow_drops','timely_pdr'):
        if env.summary[key]!=meta[key]:raise ValueError('Recovery changed fixed-policy outcome')
    base,packets=snapshot(env);tx=np.asarray(env.telemetry.nav_tx)
    return base,packets,np.concatenate([tx[...,30:66],tx[...,114:126]],-1)


def collect_job(task):
    from adaptive_joint_expanded.experiment import _worker
    if shutil.disk_usage(ARTIFACTS).free<20*1024**3:raise RuntimeError('Preserve20GiB reserve before new flight')
    parent_capture._CAPTURE=None;row=_worker(task);path=Path(task[-1]).with_suffix('.npz');meta=path.with_suffix('.json');key=sha(meta)[:20]
    r=dict(key=key,role='confirmation',region=row['region'],spec=row['spec'],deployment=row['deployment'],n_search=row['search_drones'],
        source_kind='fresh_recall_v13',array_path=str(path),metadata_path=str(meta),array_sha256=row['array_sha256'])
    paths=[ARTIFACTS/folder/(key+'.npz') for folder in ('fresh_traces','service_cache','deltas')]
    if not all(p.with_suffix('.json').exists() for p in paths):
        base,packets,delta=recover(r) if parent_capture._CAPTURE is None else parent_capture._CAPTURE
        strict=(packets['packet_status']==1)&(packets['packet_delay']<=.25);np.testing.assert_allclose(strict.mean(),row['timely_pdr'],rtol=1e-12,atol=1e-12)
        for p,values in zip(paths,(base,packets,dict(delta=delta))):
            p.parent.mkdir(exist_ok=True);tmp=p.with_suffix('.working.npz');np.savez_compressed(tmp,**values);tmp.replace(p)
            write(p.with_suffix('.json'),dict(path=str(p),sha256=sha(p),key=key,source_sha256=row['array_sha256'],
                source_pins={name:sha(ROOT/name) for name in ('confirmation.py','data.py','features.py','embedding.py')},created_at=now()))
    parent_capture._CAPTURE=None;return r


def prepare_case(record):
    item=prepare_one(dict(key=record['key'],role='confirmation',region=record['region'],source_record=record))
    return reference(item)


def main():
    from .evaluate import verify_freeze
    verify_freeze();out=ARTIFACTS/'confirmation_data_manifest.json'
    if out.exists():return out
    from adaptive_joint_expanded import ARTIFACTS as OLD
    prior=read(OLD/'protocol.json');folder=ARTIFACTS/'raw/confirmation';folder.mkdir(parents=True,exist_ok=True)
    write(ARTIFACTS/'confirmation_access.json',dict(opened_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),independent_new_seeds=True,original_final_test_opened=False))
    jobs=[]
    for region in folds()['confirmation']:
        token=digest(['v13-event-recall-new-confirmation',SEED,region]);seed=int(token[:8],16)%(2**31-3000010)
        direction=('east_west','north_south')[int(token[8:12],16)%2]
        for kind in ('clear','intermittent','continuous'):
            jobs.append((dict(region=region,direction=direction,kind=kind,seed=seed),prior['config'],None,True,None,str(folder/(region+'_'+kind))))
    raw={};status('event_recall_new_confirmation_collection',complete=0,total=len(jobs))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=parent_capture.initialize) as pool:
        pending={pool.submit(collect_job,j):i for i,j in enumerate(jobs)}
        for f in as_completed(pending):
            raw[pending[f]]=f.result();status('event_recall_new_confirmation_collection',complete=len(raw),total=len(jobs));write(ARTIFACTS/'confirmation_sources.json',dict(rows=[raw[i] for i in sorted(raw)],created_at=now()))
    prepared={};status('event_recall_new_confirmation_features',complete=0,total=len(jobs))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(prepare_case,raw[i]):i for i in sorted(raw)}
        for f in as_completed(pending):prepared[pending[f]]=f.result();status('event_recall_new_confirmation_features',complete=len(prepared),total=len(jobs))
    verify_freeze();write(out,dict(records=[prepared[i] for i in sorted(prepared)],created_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),known14maps=True,original_final_test_opened=False));return out


if __name__=='__main__':main()
