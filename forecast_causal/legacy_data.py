"""Complete event prewindows plus uniform natural rows, separate truth inputs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
from . import ROOT,ARTIFACTS,NAV
PARENT=NAV
from .contracts import read,write,sha,digest,now,verify,status,SEED
from forecast_recall.features import steps
from forecast_recall.embedding import extract
from forecast_navigation.data import arrays as parent_arrays
from forecast_navigation.wire import extend_trace
from forecast_graph.contracts import V10
from forecast_graph.features import LocalHistory,ReceivedTrace,build
from forecast_graph.episodes import RawEpisode
from forecast_rebuild.dataset import Episode
from forecast_rebuild.targets import labels,onsets
from forecast_rebuild.planning import public_context


def target(ep,health):
    links=np.zeros((len(health),2,2),bool);links[:,1,0]=health
    gt=labels(links,ep.ends,1);support=ep.legacy_support[:,0].any(1)
    return links,gt,support


def query_plan(ep,health,key):
    links,gt,support=target(ep,health);good=np.flatnonzero(gt['valid'][:,0,0]);times=ep.ends*.5;end=len(health)*.5-8.5
    rng=np.random.default_rng(int(digest(['v13-uniform',SEED,key])[:16],16));uniform=np.sort(rng.choice(good,min(512,len(good)),replace=False))
    event=np.flatnonzero(onsets(links)[:,1,0])*.5;event=event[(event>=24)&(event<=end)];extra=[];bags=[];event_times=[]
    for when in event:
        window=good[(times[good]>when-7)&(times[good]<=when-3)&support[good]&(times[good]>=21)]
        extra.extend(good[(times[good]>=when-10)&(times[good]<=when+4)].tolist())
        if len(window):bags.append(window);event_times.append(float(when))
    q=np.unique(np.r_[uniform,extra]).astype(np.int32);lookup={int(t):i for i,t in enumerate(q)}
    mapped=np.full((len(bags),4),-1,np.int32)
    for i,bag in enumerate(bags):
        if len(bag)>4:raise ValueError('More than four1Hz opportunities in strict4s event window')
        mapped[i,:len(bag)]=[lookup[int(t)] for t in bag]
    return q,np.isin(q,uniform),mapped,np.asarray(event_times),links,gt,support


def prepare_one(item):
    started=time.monotonic();verify();r=item['source_record'];key=item['key'];folder=ARTIFACTS/'legacy_data';folder.mkdir(exist_ok=True)
    path=folder/(key+'.npz');meta=path.with_suffix('.json');role=item['role'];fresh=role=='confirmation'
    pins={name:sha(ROOT/name) for name in ('legacy_data.py','features.py','references.py')}
    if meta.exists():
        old=read(meta)
        if old['source_pins']!=pins or sha(path)!=old['sha256']:raise ValueError('Input cache changed')
        return old
    if shutil.disk_usage(ARTIFACTS).free<18*1024**3:raise RuntimeError('Preserve18GiB reserve')
    ep=RawEpisode(r) if fresh else Episode(r);n=ep.values.shape[2];ns=r['n_search']
    service=(ARTIFACTS if fresh else V10)/'service_cache'/(key+'.npz')
    with np.load(service,allow_pickle=False) as a:health=a['service_healthy']
    if role=='train':q,uniform,bags,event_times,links,gt,support=query_plan(ep,health,key)
    else:
        links,gt,support=target(ep,health);q=np.flatnonzero(gt['valid'][:,0,0]);uniform=np.ones(len(q),bool);bags=np.empty((0,4),np.int32);event_times=np.array([])
    endpoints=np.asarray(ep.ends)[q];history=steps(ep.values[:,0],ep.masks[:,0],ep.intents[:,0],ep.energy[:,0],n,ns)
    if role in ('selection','calibration'):
        old=parent_arrays(item,'mission');np.testing.assert_equal(old['query'],np.c_[q,np.zeros(len(q),int)])
        rep=extract(old,device='cpu')
    else:
        trace=(ARTIFACTS/'fresh_traces' if fresh else PARENT.parent.parent/'forecast_graph/artifacts/traces')/(key+'.npz')
        delta=(ARTIFACTS if fresh else PARENT)/'deltas'/(key+'.npz')
        with np.load(trace,allow_pickle=False) as a:tx=a['tx'];rx=a['rx']
        with np.load(delta,allow_pickle=False) as a:revised=extend_trace(tx,a['delta'])
        city,_=public_context(r);city.radio.enable_cache();local=LocalHistory(ep.values,ep.masks,ep.intents,n,ns);received=ReceivedTrace(revised,rx,n,ns)
        parts={k:[] for k in ('embedding','logits')}
        for start in range(0,len(q),128):
            ends=endpoints[start:start+128];a=build(local,received,np.c_[ends,np.zeros(len(ends),int)],city)
            a.update(n=np.full(len(ends),n),ns=np.full(len(ends),ns),observer=np.full(len(ends),ns))
            out=extract(a,device='cpu')
            for k in parts:parts[k].append(out[k])
        rep={k:np.concatenate(v) for k,v in parts.items()}
    values=dict(**rep,history=history,ends=endpoints.astype(np.int32),query=q,uniform=uniform,bags=bags,event_times=event_times,
        y=gt['targets'][q,0,0].astype(np.uint8),up=gt['currently_up'][q,0,0],support=support[q],links=links)
    temporary=path.with_suffix('.working.npz');np.savez_compressed(temporary,**values);temporary.replace(path)
    value=dict(key=key,role=role,region=r['region'],source_record=r,path=str(path),sha256=sha(path),queries=len(q),
        uniform_queries=int(uniform.sum()),train_event_bags=len(bags),source_pins=pins,service_sha256=sha(service),source_sha256=r['array_sha256'],
        parent_item=item if role in ('selection','calibration') else None,created_at=now(),bytes=path.stat().st_size,wall_seconds=time.monotonic()-started,
        local_cache_sha256={} if fresh else {name:sha(Path(r['cache_path'])/(name+'.npy')) for name in ('values','masks','intents','energy','ends','legacy_support')})
    write(meta,value);return value

