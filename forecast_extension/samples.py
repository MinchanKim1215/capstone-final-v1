"""Balanced fixed validation probe; original training arrays remain read-only."""
from pathlib import Path
import argparse
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,digest,SEED,OLD_DEVELOPMENT,OLD_SAMPLES,now,prepare
from forecast_rebuild.dataset import Samples,Episode


def make_probe():
    prepare();folder=ARTIFACTS/'samples/balanced_probe';path=folder/'manifest.json'
    if path.exists():return folder
    rows=[r for r in read(OLD_DEVELOPMENT)['rows'] if r['role']=='selection']
    if len(rows)!=45:raise ValueError('Balanced selection requires 45 flights')
    parts=[]
    for e,record in enumerate(rows):
        candidates=Episode(record).eligible(legacy=True)
        rng=np.random.default_rng(int(digest(['probe-v9',SEED,record['key']])[:16],16))
        chosen=candidates[rng.choice(len(candidates),min(900,len(candidates)),replace=False)]
        parts.append(np.c_[np.full(len(chosen),e),chosen])
    index=np.concatenate(parts).astype(np.int32);folder.mkdir(parents=True,exist_ok=True);n=len(index)
    arrays={name:np.lib.format.open_memmap(folder/('selection_'+suffix+'.npy'),mode='w+',dtype=dtype,shape=shape)
            for name,suffix,dtype,shape in [('x','x',np.float32,(n,728)),('y','y',np.uint8,(n,6)),('up','up',bool,(n,)),
                ('next','next',np.uint8,(n,)),('legacy','legacy',bool,(n,))]}
    for e,record in enumerate(rows):
        positions=np.flatnonzero(index[:,0]==e);local=index[positions,1:];t,r,j=local.T;episode=Episode(record)
        for start in range(0,len(local),512):arrays['x'][positions[start:start+512]]=episode.features(local[start:start+512],True,True)
        for name,source in [('y','targets'),('up','currently_up'),('next','next_bin'),('legacy','legacy_support')]:arrays[name][positions]=getattr(episode,source)[t,r,j]
    np.save(folder/'selection_index.npy',index,allow_pickle=False)
    for array in arrays.values():array.flush()
    write(path,dict(contract=dict(records=rows,planning=True,source_dataset_sha256=sha(OLD_DEVELOPMENT),seed=SEED),
        dimension=728,counts={'selection':n},created_at=now(),purpose='Fixed balanced validation probe only; no calibration or confirmation'))
    print('Balanced probe:',n,flush=True);return folder


class CombinedSamples:
    """Concatenation without copying the immutable 1.6GB predecessor matrix."""
    def __init__(self,old_folder=OLD_SAMPLES,new_folder=None):
        self.parts=[Samples(old_folder,'train')]
        if new_folder is not None:self.parts.append(Samples(new_folder,'train'))
        self.offsets=np.r_[0,np.cumsum([len(part.y) for part in self.parts])]
        self.groups={}
        for part_id,part in enumerate(self.parts):
            for e in np.unique(part.index[:,0]):
                record=part.records[int(e)];key=record.get('source_flight_key',record['key'])
                ids=np.flatnonzero(part.index[:,0]==e)+self.offsets[part_id]
                self.groups.setdefault(key,[]).append(ids)
        self.groups={key:np.concatenate(parts) for key,parts in self.groups.items()}
        self.group_keys=list(self.groups);self.n=int(self.offsets[-1]);self.dimension=self.parts[0].x.shape[1]
        if any(p.x.shape[1]!=self.dimension for p in self.parts):raise ValueError('Feature dimensions differ')

    def draw(self,rng):
        keys=rng.choice(len(self.group_keys),4,replace=False)
        return np.concatenate([rng.choice(self.groups[self.group_keys[i]],64,replace=len(self.groups[self.group_keys[i]])<64) for i in keys])

    def neural_batch(self,positions,use_energy=True):
        positions=np.asarray(positions);sequences=np.empty((len(positions),40,111),np.float32)
        static=np.empty((len(positions),self.dimension),np.float32);y=np.empty((len(positions),6),np.float32)
        up=np.empty(len(positions),bool);next_bin=np.empty(len(positions),np.int64)
        for k,part in enumerate(self.parts):
            take=np.flatnonzero((positions>=self.offsets[k])&(positions<self.offsets[k+1]))
            if not len(take):continue
            result=part.neural_batch(positions[take]-self.offsets[k],use_energy)
            for out,value in zip((sequences,static,y,up,next_bin),result):out[take]=value
        return sequences,static,y,up,next_bin


if __name__=='__main__':print(make_probe())

