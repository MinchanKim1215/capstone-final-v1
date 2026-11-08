"""New fixed-policy flights after freeze; received-plan logging cannot affect PPO."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,digest,now,status,folds,SEED,BASELINE
from .trace import attach
from forecast_service.replay import service_health

_CAPTURE=None


def snapshot(env):
    tel=env.telemetry;g=int(env.packets.ic[0]);pk=env.packets
    packets={name:np.array(value[:g]) for name,value in [('packet_born',pk.born),('packet_status',pk.status),('packet_delay',pk.delays)]}
    packets.update(service_health(packets['packet_born'],packets['packet_status'],packets['packet_delay'],env.t))
    trace=dict(tx=np.asarray(tel.plan_tx),rx=np.packbits(np.asarray(tel.plan_rx),axis=-1),
        sinr=tel.sinr[:env.t].astype(np.float16),n=np.array(env.n),ns=np.array(env.ns))
    return trace,packets


def initialize():
    from adaptive_joint_expanded import ARTIFACTS as OLD
    from adaptive_joint_expanded.experiment import _initialize
    from adaptive_joint_expanded.world import World
    from adaptive_joint_expanded import rollout
    _initialize(str(BASELINE),read(OLD/'protocol.json'))
    class LoggedWorld(World):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);attach(self)
        def metrics(self,event):
            global _CAPTURE
            result=super().metrics(event);_CAPTURE=snapshot(self);return result
    rollout.World=LoggedWorld


def recover(record):
    from adaptive_joint_expanded.world import World
    env=World(**record['spec'],deployment=record['deployment'],observer=None);attach(env)
    with np.load(record['array_path'],allow_pickle=False) as a:commands=a['commands'];nodes=a['nodes'];links=a['links']
    for k,c in enumerate(commands):
        env.step(c);np.testing.assert_equal(env.nodes,nodes[k+1]);np.testing.assert_equal(env.channel_available[-6:],links[k*6:k*6+6])
    assert env.done
    original=read(record['metadata_path'])
    for key in ('generated_packets','delivered_packets','ttl_drops','retry_drops','overflow_drops','retries','timely_pdr'):
        if env.summary[key]!=original[key]:raise ValueError('Recovery changed packet engine')
    return snapshot(env)


def collect_job(task,role='confirmation'):
    global _CAPTURE
    from adaptive_joint_expanded.experiment import _worker
    if shutil.disk_usage(ARTIFACTS).free<20*1024**3:raise RuntimeError('Insufficient safe disk reserve for fresh archive')
    _CAPTURE=None;row=_worker(task);path=Path(task[-1]).with_suffix('.npz');meta=path.with_suffix('.json');key=sha(meta)[:20]
    record=dict(role=role,region=row['region'],spec=row['spec'],deployment=row['deployment'],n_search=row['search_drones'],
        source_kind='fresh_v11',key=key,metadata_path=str(meta),array_path=str(path),array_sha256=row['array_sha256'])
    tracepath=ARTIFACTS/'traces'/(key+'.npz');servicepath=ARTIFACTS/'service_cache'/(key+'.npz')
    if not tracepath.with_suffix('.json').exists() or not servicepath.with_suffix('.json').exists():
        recovery=_CAPTURE is None;trace,packets=recover(record) if recovery else _CAPTURE
        strict=(packets['packet_status']==1)&(packets['packet_delay']<=.25)
        np.testing.assert_allclose(strict.mean(),row['timely_pdr'],atol=1e-12,rtol=1e-12)
        for target,arrays in [(tracepath,trace),(servicepath,packets)]:
            target.parent.mkdir(exist_ok=True);temp=target.with_suffix('.working.npz');np.savez_compressed(temp,**arrays);temp.replace(target)
            write(target.with_suffix('.json'),dict(key=key,role=role,path=str(target),sha256=sha(target),source_sha256=row['array_sha256'],
                source_pins={name:sha(ROOT/name) for name in ('codec.py','trace.py','confirmation.py')},
                post_flight_read_only_capture=not recovery,exact_replay_recovery=recovery,created_at=now()))
    _CAPTURE=None;return record


def prepare_case(record):
    from .episodes import RawEpisode
    from .features import LocalHistory,ReceivedTrace,build
    from .data import queries,target_values,FEATURE_KEYS
    folder=ARTIFACTS/'data';folder.mkdir(exist_ok=True);path=folder/(record['key']+'.npz');meta=path.with_suffix('.json')
    if meta.exists():return read(meta)
    ep=RawEpisode(record);q=queries(record,ep);ends=ep.ends[q[:,0]]
    tracepath=ARTIFACTS/'traces'/(record['key']+'.npz');servicepath=ARTIFACTS/'service_cache'/(record['key']+'.npz')
    with np.load(tracepath,allow_pickle=False) as a:trace=ReceivedTrace(a['tx'],a['rx'],ep.values.shape[2],record['n_search']);sinr=a['sinr']
    with np.load(servicepath,allow_pickle=False) as a:health=a['service_healthy']
    local=LocalHistory(ep.values,ep.masks,ep.intents,ep.values.shape[2],record['n_search']);parts={k:[] for k in FEATURE_KEYS}
    for start in range(0,len(q),128):
        f=build(local,trace,np.c_[ends[start:start+128],q[start:start+128,1]],ep.city)
        for key,value in f.items():parts[key].append(value)
    arrays={k:np.concatenate(v) for k,v in parts.items()};arrays.update(target_values(ep,q,sinr,health,local.n,local.ns));arrays['query']=q.astype(np.int32)
    tmp=path.with_suffix('.working.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
    item=dict(key=record['key'],role=record['role'],region=record['region'],path=str(path),sha256=sha(path),queries=len(q),source_record=record,
        n=local.n,ns=local.ns,contract=dict(source_sha256=record['array_sha256'],trace_sha256=sha(tracepath),
            features_sha256=sha(ROOT/'features.py'),confirmation_sha256=sha(ROOT/'confirmation.py')),created_at=now(),bytes=path.stat().st_size)
    write(meta,item);return item


def main():
    from .evaluate import verify_freeze
    frozen=verify_freeze();manifest=ARTIFACTS/'confirmation_data_manifest.json'
    if manifest.exists():return manifest
    from adaptive_joint_expanded import ARTIFACTS as OLD
    prior=read(OLD/'protocol.json');folder=ARTIFACTS/'raw/confirmation';folder.mkdir(parents=True,exist_ok=True)
    write(ARTIFACTS/'confirmation_access.json',dict(opened_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),fresh_seeds=True,original_final_test_opened=False))
    tasks=[]
    for region in folds()['confirmation']:
        token=digest(['v11-independent-confirmation',SEED,region]);seed=int(token[:8],16)%(2**31-3000010)
        direction=('east_west','north_south')[int(token[8:12],16)%2]
        for kind in ('clear','intermittent','continuous'):
            tasks.append((dict(region=region,direction=direction,kind=kind,seed=seed),prior['config'],None,True,None,str(folder/(region+'_'+kind))))
    status('fresh_confirmation_collection',complete=0,total=len(tasks));records={}
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        pending={pool.submit(collect_job,task):i for i,task in enumerate(tasks)}
        for future in as_completed(pending):
            records[pending[future]]=future.result();status('fresh_confirmation_collection',complete=len(records),total=len(tasks))
            write(ARTIFACTS/'confirmation_sources.json',dict(rows=[records[i] for i in sorted(records)],created_at=now()))
    prepared={};status('fresh_confirmation_features',complete=0,total=len(tasks))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(prepare_case,records[i]):i for i in sorted(records)}
        for future in as_completed(pending):
            prepared[pending[future]]=future.result();status('fresh_confirmation_features',complete=len(prepared),total=len(tasks))
    verify_freeze();write(manifest,dict(records=[prepared[i] for i in sorted(prepared)],created_at=now(),
        freeze_sha256=sha(ARTIFACTS/'freeze.json'),original_final_test_opened=False,known_maps_only=True))
    return manifest


if __name__=='__main__':main()

