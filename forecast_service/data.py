"""Service target views and fixed, flight-balanced training feature tables."""
from pathlib import Path
from collections import Counter
import time
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,digest,now,status,verify,V9,SEED
from .features import build,BASE_DIM,GRAPH_DIM,VERSION
from forecast_rebuild.dataset import Episode
from forecast_rebuild.targets import labels
from forecast_rebuild.planning import public_context


def truth(record):
    key=record.get('clean_record_key',record['key']);path=ARTIFACTS/'service_cache'/(key+'.npz')
    with np.load(path,allow_pickle=False) as a:health=a['service_healthy']
    links=np.zeros((len(health),2,2),bool);links[:,1,0]=health
    episode=Episode(record);value=labels(links,episode.ends,1)
    return links,value


class ServiceEpisode:
    def __init__(self,record):
        physical=Episode(record);self.physical=physical;self.original_record=record;self.record=dict(record,n_search=1)
        self.links,label=truth(record);self.ends=physical.ends
        for key,value in label.items():setattr(self,key,value)
        self.legacy_support=np.zeros(self.valid.shape,bool);self.legacy_support[:,0,0]=np.asarray(physical.legacy_support[:,0]).any(1)
        self.available=self.legacy_support;self.energy=physical.energy


def query_rows(record):
    physical=Episode(record);_,value=truth(record);valid=value['valid'][:,0,0];endpoints=np.flatnonzero(valid)
    if record['role']!='train':return np.c_[endpoints,np.zeros(len(endpoints),int)]
    ns=record['n_search'];nr=physical.values.shape[1];eligible=np.array(np.meshgrid(endpoints,np.arange(nr),indexing='ij')).reshape(2,-1).T
    rng=np.random.default_rng(int(digest(['v10-service-query',SEED,record['key']])[:16],16))
    # The clean/augmented views share one source quota; query identities are
    # disjoint by alternating a single clean-source permutation.
    if record['source_kind']=='augmented_v9':
        source=record['clean_record_key'];rng=np.random.default_rng(int(digest(['v10-service-query',SEED,source])[:16],16))
        return eligible[rng.choice(len(eligible),min(1200,len(eligible)),replace=False)[1::2]]
    augmented={r['clean_record_key'] for r in read(V9/'dataset.json')['rows'] if r['source_kind']=='augmented_v9'}
    chosen=rng.choice(len(eligible),min(1200,len(eligible)),replace=False)
    return eligible[chosen[::2] if record['key'] in augmented else chosen]


def main():
    verify()
    while not (ARTIFACTS/'replay_status.json').exists():
        if read(ARTIFACTS/'status.json').get('state')=='failed':raise RuntimeError('Packet replay failed')
        time.sleep(5)
    rows=[r for r in read(V9/'dataset.json')['rows'] if r['source_kind'] in ('fresh_v9','augmented_v9') and r['role'] in ('train','selection','calibration')]
    for r in rows:
        if not (ARTIFACTS/'service_cache'/(r.get('clean_record_key',r['key'])+'.npz')).exists():raise ValueError('Missing settled service source')
    folder=ARTIFACTS/'samples';folder.mkdir(exist_ok=True);manifest=folder/'manifest.json'
    if manifest.exists():return
    indices={}
    for split in ('train','selection','calibration'):
        parts=[]
        for e,r in enumerate(rows):
            if r['role']==split:
                q=query_rows(r);parts.append(np.c_[np.full(len(q),e),q])
        indices[split]=np.concatenate(parts).astype(np.int32)
    counts={key:len(value) for key,value in indices.items()};arrays={}
    for split,index in indices.items():
        np.save(folder/(split+'_index.npy'),index,allow_pickle=False)
        arrays[split]={name:np.lib.format.open_memmap(folder/(split+'_'+name+'.npy'),mode='w+',dtype=dtype,shape=(len(index),*tail))
            for name,dtype,tail in [('x',np.float32,(BASE_DIM+GRAPH_DIM,)),('y',np.uint8,(6,)),('up',bool,()),('support',bool,())]}
    for e,r in enumerate(rows):
        split=r['role'];index=indices[split];take=np.flatnonzero(index[:,0]==e)
        if not len(take):continue
        physical=Episode(r);city,_=public_context(r);query=index[take,1:];_,label=truth(r);t,observer=query.T
        for start in range(0,len(query),256):arrays[split]['x'][take[start:start+256]]=build(physical,query[start:start+256],r['n_search'],city)
        arrays[split]['y'][take]=label['targets'][t,0,0];arrays[split]['up'][take]=label['currently_up'][t,0,0]
        arrays[split]['support'][take]=np.asarray(physical.legacy_support)[t,observer].any(1)
        if e%8==0 or e==len(rows)-1:
            status('stage2_service_feature_tables',complete=e+1,total=len(rows));print(e+1,'/',len(rows),flush=True)
    for part in arrays.values():
        for a in part.values():a.flush()
    write(manifest,dict(records=rows,counts=counts,dimension=BASE_DIM+GRAPH_DIM,base_dimension=BASE_DIM,feature_version=VERSION,
        source_dataset_sha256=sha(V9/'dataset.json'),experiment_plan_sha256=sha(ARTIFACTS/'experiment_plan.json'),
        replay_manifest_sha256=sha(ARTIFACTS/'replay_manifest.json'),created_at=now(),contains_confirmation=False,
        physical_training_flights=245,augmented_training_variants=84,query_primary_observer='first relay for selection/calibration'))
    print(manifest)


if __name__=='__main__':main()
