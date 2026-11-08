"""New-seed fixed-policy confirmation; packet logging occurs after flight."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,digest,now,status,folds,SEED,BASELINE
from .replay import service_health,LABEL_VERSION
from .evaluate import verify_freeze

_PACKETS=None


def packet_trace(env):
    g=int(env.packets.ic[0]);pk=env.packets
    value={name:np.array(array[:g]) for name,array in [('packet_source',pk.source),('packet_dest',pk.dest),('packet_born',pk.born),
        ('packet_status',pk.status),('packet_delay',pk.delays),('packet_hops',pk.hops)]}
    value['packet_frames']=np.asarray(env.packet_frames)
    value.update(service_health(value['packet_born'],value['packet_status'],value['packet_delay'],env.t));return value


def initialize():
    from adaptive_joint_expanded.experiment import _initialize
    from adaptive_joint_expanded import ARTIFACTS as OLD
    from adaptive_joint_expanded.world import World
    from adaptive_joint_expanded import rollout
    _initialize(str(BASELINE),read(OLD/'protocol.json'))
    class LoggedWorld(World):
        def metrics(self,event):
            global _PACKETS
            result=super().metrics(event)
            # Mission, motion, channel, packet RNG and metrics have already
            # completed. Copy arrays only; do not affect any policy input.
            _PACKETS=packet_trace(self);return result
    rollout.World=LoggedWorld


def collect_job(task):
    global _PACKETS
    from adaptive_joint_expanded.experiment import _worker
    _PACKETS=None;row=_worker(task);path=Path(task[-1]).with_suffix('.npz');meta=path.with_suffix('.json');key=sha(meta)[:20]
    record=dict(role='confirmation',region=row['region'],spec=row['spec'],deployment=row['deployment'],n_search=row['search_drones'],
        source_kind='fresh_v10',key=key,metadata_path=str(meta),array_path=str(path),array_sha256=row['array_sha256'])
    target=ARTIFACTS/'service_cache'/(key+'.npz');target.parent.mkdir(exist_ok=True)
    if not target.with_suffix('.json').exists():
        if _PACKETS is None:
            # Recover only if a prior process committed the flight but not
            # the post-flight packet copy. Exact replay is independently tested.
            from .replay import replay
            record=prepare_case(record);replay(record)
        else:
            arrays=_PACKETS
            assert len(arrays['packet_born'])==row['generated_packets']
            timely=(arrays['packet_status']==1)&(arrays['packet_delay']<=.25+1e-10)
            assert np.isclose(timely.mean(),row['timely_pdr'],atol=1e-12,rtol=1e-12)
            temporary=target.with_suffix('.working.npz');np.savez_compressed(temporary,**arrays);temporary.replace(target)
            write(target.with_suffix('.json'),dict(key=key,role='confirmation',region=row['region'],path=str(target),sha256=sha(target),frames=len(arrays['service_healthy']),
                request=dict(source_sha256=row['array_sha256'],spec=row['spec'],deployment=row['deployment'],label_version=LABEL_VERSION),
                created_at=now(),post_flight_read_only_logging=True,packet_conserved=True))
    _PACKETS=None;return record


def prepare_case(record):
    from forecast_rebuild.targets import labels,past_support
    from forecast_rebuild.sensors import simulate,VERSION as SENSOR_VERSION
    from forecast_rebuild.planning import profile,public_context
    from forecast_rebuild.dataset import Episode
    folder=ARTIFACTS/'cache'/record['key'];manifest=folder/'manifest.json'
    if not manifest.exists():
        if sha(record['array_path'])!=record['array_sha256']:raise ValueError('Confirmation source changed')
        with np.load(record['array_path'],allow_pickle=False) as a:
            values=a['rf_values'];masks=a['rf_masks'];intents=a['rf_intents'];links=a['links'];ends=a['aux_endpoints'];geometry=a['aux_geometry'];positions=a['positions']
        take=ends>=40;ends=ends[take];geometry=geometry[take];truth=labels(links,ends,record['n_search']);support=past_support(values,masks,ends,record['n_search'])
        energy,sensor=simulate(record,values,links,positions);folder.mkdir(parents=True,exist_ok=True)
        arrays=dict(values=values,masks=masks,intents=intents,links=links,ends=ends,geometry=geometry,energy=energy.astype(np.float16),
            targets=truth['targets'].astype(np.uint8),valid=truth['valid'],currently_up=truth['currently_up'],next_bin=truth['next_bin'].astype(np.uint8),legacy_support=support['legacy'],available=support['available'])
        for name,value in arrays.items():np.save(folder/(name+'.npy'),value,allow_pickle=False)
        write(manifest,dict(source_sha256=record['array_sha256'],sensor_version=SENSOR_VERSION,sensor=sensor,frames=len(values),created_at=now()))
    elif read(manifest)['source_sha256']!=record['array_sha256']:raise ValueError('Cached confirmation differs')
    record=dict(record,cache_path=str(folder),frames=read(manifest)['frames'])
    if not (folder/'planning_manifest.json').exists():
        ep=Episode(record);city,nominal=public_context(record)
        value=profile(city,nominal,ep.values,ep.masks,ep.intents,ep.ends,record['n_search'])
        np.save(folder/'planning.npy',value,allow_pickle=False)
        write(folder/'planning_manifest.json',dict(source_sha256=record['array_sha256'],planning_sha256=sha(folder/'planning.npy'),actual_future_positions_used=False))
    return record


def main():
    frozen=verify_freeze();final=ARTIFACTS/'confirmation_dataset.json'
    if final.exists():return final
    from adaptive_joint_expanded import ARTIFACTS as OLD
    prior=read(OLD/'protocol.json');folder=ARTIFACTS/'raw/confirmation';folder.mkdir(parents=True,exist_ok=True)
    write(ARTIFACTS/'confirmation_access.json',dict(opened_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),fresh_seeds=True,original_final_test_opened=False))
    tasks=[]
    for region in folds()['confirmation']:
        token=digest(['v10-new-confirmation',SEED,region]);seed=int(token[:8],16)%(2**31-3000010)
        direction=('east_west','north_south')[int(token[8:12],16)%2]
        for kind in ('clear','intermittent','continuous'):
            spec=dict(region=region,direction=direction,kind=kind,seed=seed);prefix=folder/(region+'_'+kind)
            tasks.append((spec,prior['config'],None,True,None,str(prefix)))
    status('stage4_new_confirmation_collection',complete=0,total=len(tasks));records={}
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        pending={pool.submit(collect_job,task):i for i,task in enumerate(tasks)}
        for future in as_completed(pending):
            records[pending[future]]=future.result();status('stage4_new_confirmation_collection',complete=len(records),total=len(tasks))
            write(ARTIFACTS/'new_collection_manifest.json',dict(rows=[records[i] for i in sorted(records)],created_at=now()))
    prepared={};status('stage4_new_confirmation_input_preparation',complete=0,total=len(tasks))
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(prepare_case,records[i]):i for i in sorted(records)}
        for future in as_completed(pending):
            prepared[pending[future]]=future.result();status('stage4_new_confirmation_input_preparation',complete=len(prepared),total=len(tasks))
    verify_freeze();rows=[prepared[i] for i in sorted(prepared)]
    write(final,dict(rows=rows,counts={'confirmation':len(rows)},created_at=now(),freeze_sha256=sha(ARTIFACTS/'freeze.json'),known_maps_only=True,original_final_test_opened=False))
    return final


if __name__=='__main__':main()
