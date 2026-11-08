"""Independent42flights after V12freeze, with one extended128B payload."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import shutil
from pathlib import Path
import numpy as np
from . import ROOT,ARTIFACTS
from .contracts import read,write,sha,digest,now,status,folds,SEED,BASELINE
from .wire import attach_navigation,extend_trace
from .inference import parent_choices,parent_predict
from .data import cohort_arrays,REPLACED
from forecast_graph.confirmation import snapshot
from forecast_graph.features import LocalHistory,ReceivedTrace,build
from forecast_graph.data import target_values,FEATURE_KEYS

_CAPTURE=None


def initialize():
    from adaptive_joint_expanded import ARTIFACTS as OLD
    from adaptive_joint_expanded.experiment import _initialize
    from adaptive_joint_expanded.world import World
    from adaptive_joint_expanded import rollout
    _initialize(str(BASELINE),read(OLD/'protocol.json'))
    class NavigationWorld(World):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);attach_navigation(self)
        def metrics(self,event):
            global _CAPTURE
            result=super().metrics(event);base,packets=snapshot(self);tx=np.asarray(self.telemetry.nav_tx)
            _CAPTURE=(base,packets,np.concatenate([tx[...,30:66],tx[...,114:126]],-1));return result
    rollout.World=NavigationWorld


def recover(record):
    from adaptive_joint_expanded.world import World
    env=World(**record['spec'],deployment=record['deployment'],observer=None);attach_navigation(env)
    with np.load(record['array_path'],allow_pickle=False) as a:commands=a['commands'];nodes=a['nodes'];links=a['links']
    for k,c in enumerate(commands):
        env.step(c);np.testing.assert_equal(env.nodes,nodes[k+1]);np.testing.assert_equal(env.channel_available[-6:],links[k*6:k*6+6])
    assert env.done
    original=read(record['metadata_path'])
    for key in ('generated_packets','delivered_packets','retries','ttl_drops','retry_drops','overflow_drops','timely_pdr'):
        if env.summary[key]!=original[key]:raise ValueError('Exact recovery changed data-plane results')
    base,packets=snapshot(env);tx=np.asarray(env.telemetry.nav_tx)
    return base,packets,np.concatenate([tx[...,30:66],tx[...,114:126]],-1)


def collect_job(task):
    global _CAPTURE
    from adaptive_joint_expanded.experiment import _worker
    if shutil.disk_usage(ARTIFACTS).free<20*1024**3:raise RuntimeError('Preserve20GiB reserve before fresh flight')
    _CAPTURE=None;row=_worker(task);path=Path(task[-1]).with_suffix('.npz');meta=path.with_suffix('.json');key=sha(meta)[:20]
    record=dict(key=key,role='confirmation',region=row['region'],spec=row['spec'],deployment=row['deployment'],n_search=row['search_drones'],
        source_kind='fresh_navigation_v12',array_path=str(path),metadata_path=str(meta),array_sha256=row['array_sha256'])
    basepath=ARTIFACTS/'fresh_traces'/(key+'.npz');servicepath=ARTIFACTS/'service_cache'/(key+'.npz');delta_path=ARTIFACTS/'deltas'/(key+'.npz')
    if not all(p.with_suffix('.json').exists() for p in (basepath,servicepath,delta_path)):
        base,packets,delta=recover(record) if _CAPTURE is None else _CAPTURE
        strict=(packets['packet_status']==1)&(packets['packet_delay']<=.25)
        np.testing.assert_allclose(strict.mean(),row['timely_pdr'],atol=1e-12,rtol=1e-12)
        for target,values in [(basepath,base),(servicepath,packets),(delta_path,dict(delta=delta))]:
            target.parent.mkdir(exist_ok=True);tmp=target.with_suffix('.working.npz');np.savez_compressed(tmp,**values);tmp.replace(target)
            write(target.with_suffix('.json'),dict(path=str(target),sha256=sha(target),key=key,source_sha256=row['array_sha256'],
                source_pins={name:sha(ROOT/name) for name in ('preview.py','wire.py','confirmation.py')},post_flight_capture=True,created_at=now()))
    _CAPTURE=None;return record


def prepare_case(record):
    from forecast_graph.episodes import RawEpisode
    folder=ARTIFACTS/'fresh_data';folder.mkdir(exist_ok=True);meta=folder/(record['key']+'.json')
    if meta.exists():return read(meta)
    ep=RawEpisode(record);n=ep.values.shape[2];ns=record['n_search'];t=np.flatnonzero(ep.valid.any((1,2)));q=np.c_[t,np.zeros(len(t),int)];ends=ep.ends[t]
    with np.load(ARTIFACTS/'fresh_traces'/(record['key']+'.npz'),allow_pickle=False) as a:tx=a['tx'];rx=a['rx'];sinr=a['sinr']
    with np.load(ARTIFACTS/'deltas'/(record['key']+'.npz'),allow_pickle=False) as a:revised=extend_trace(tx,a['delta'])
    with np.load(ARTIFACTS/'service_cache'/(record['key']+'.npz'),allow_pickle=False) as a:health=a['service_healthy']
    local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns);baseparts={k:[] for k in FEATURE_KEYS};missionparts={k:[] for k in REPLACED}
    views={'original':ReceivedTrace(tx,rx,n,ns),'mission':ReceivedTrace(revised,rx,n,ns)}
    for start in range(0,len(q),128):
        query=np.c_[ends[start:start+128],q[start:start+128,1]]
        old=build(local,views['original'],query,ep.city);new=build(local,views['mission'],query,ep.city)
        for key,value in old.items():baseparts[key].append(value)
        for key in ('node_local','edge_local','edge_shared'):np.testing.assert_equal(old[key],new[key])
        for key in REPLACED:missionparts[key].append(new[key])
    base={k:np.concatenate(v) for k,v in baseparts.items()};base.update(target_values(ep,q,sinr,health,n,ns));base['query']=q.astype(np.int32)
    basepath=folder/(record['key']+'_base.npz');missionpath=folder/(record['key']+'_mission.npz');cohortpath=folder/(record['key']+'_cohort.npz');referencepath=folder/(record['key']+'_reference.npz')
    np.savez_compressed(basepath,**base);np.savez_compressed(missionpath,**{k:np.concatenate(v) for k,v in missionparts.items()})
    np.savez_compressed(cohortpath,**cohort_arrays(ep,base,health,n,ns))
    references={};cache={}
    for task,items in parent_choices().items():
        for kind,item in items.items():references[task+'_'+kind]=parent_predict(item['candidate'],base,record,ep,cache)
    np.savez_compressed(referencepath,**references)
    item=dict(key=record['key'],role='confirmation',region=record['region'],source_record=record,queries=len(q),
        base_path=str(basepath),base_sha256=sha(basepath),mission_path=str(missionpath),mission_sha256=sha(missionpath),
        cohort_path=str(cohortpath),cohort_sha256=sha(cohortpath),reference_path=str(referencepath),reference_sha256=sha(referencepath),created_at=now())
    write(meta,item);return item


def main():
    from .evaluate import verify_freeze
    verify_freeze();out=ARTIFACTS/'confirmation_data_manifest.json'
    if out.exists():return out
    from adaptive_joint_expanded import ARTIFACTS as OLD
    prior=read(OLD/'protocol.json');folder=ARTIFACTS/'raw/confirmation';folder.mkdir(parents=True,exist_ok=True)
    write(ARTIFACTS/'confirmation_access.json',dict(opened_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),independent_new_seeds=True,original_final_test_opened=False))
    jobs=[]
    for region in folds()['confirmation']:
        token=digest(['v12-navigation-new-confirmation',SEED,region]);seed=int(token[:8],16)%(2**31-3000010)
        direction=('east_west','north_south')[int(token[8:12],16)%2]
        for kind in ('clear','intermittent','continuous'):
            jobs.append((dict(region=region,direction=direction,kind=kind,seed=seed),prior['config'],None,True,None,str(folder/(region+'_'+kind))))
    records={};status('navigation_new_confirmation_collection',complete=0,total=len(jobs))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        pending={pool.submit(collect_job,j):i for i,j in enumerate(jobs)}
        for f in as_completed(pending):
            records[pending[f]]=f.result();status('navigation_new_confirmation_collection',complete=len(records),total=len(jobs))
            write(ARTIFACTS/'confirmation_sources.json',dict(rows=[records[i] for i in sorted(records)],created_at=now()))
    prepared={};status('navigation_new_confirmation_features',complete=0,total=len(jobs))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(prepare_case,records[i]):i for i in sorted(records)}
        for f in as_completed(pending):prepared[pending[f]]=f.result();status('navigation_new_confirmation_features',complete=len(prepared),total=len(jobs))
    verify_freeze();write(out,dict(records=[prepared[i] for i in sorted(prepared)],created_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),known14maps=True,original_final_test_opened=False))
    return out


if __name__=='__main__':main()
