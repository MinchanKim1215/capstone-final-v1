"""Bounded prediction samples; the full RF trajectory is never given to a model."""
from functools import lru_cache
from pathlib import Path
import argparse
import json
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,digest,SEED,now
from .features import build,sequence,VERSION as FEATURE_VERSION

ARRAYS=('values','masks','intents','ends','geometry','targets','valid','currently_up','next_bin','legacy_support','available','energy','links')


class Episode:
    def __init__(self,record):
        self.record=record;folder=Path(record['cache_path'])
        for name in ARRAYS:setattr(self,name,np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False))

    def features(self,rows,use_energy=True,planning=False):
        base=build(self.values,self.masks,self.intents,self.ends,self.geometry,rows,self.energy,use_energy)
        if not planning:return base
        if not hasattr(self,'planning'):
            self.planning=np.load(Path(self.record['cache_path'])/'planning.npy',mmap_mode='r',allow_pickle=False)
        t,r,j=np.asarray(rows,int).T
        return np.concatenate([base,self.planning[t,r,j]],-1)

    def sequence(self,rows,use_energy=True):
        return sequence(self.values,self.masks,self.intents,self.ends,rows,self.energy,use_energy)

    def eligible(self,legacy=False):
        return np.argwhere(self.valid & (self.legacy_support if legacy else self.available))


def without_energy(features):
    out=np.asarray(features).copy()
    out[:,620:644]=np.tile([np.nan,np.nan,np.nan,0,0,np.nan],(len(out),4))
    return out


def make(tag='engineering',train_per_old=1200,train_per_new=2400,planning=False):
    source=read(ARTIFACTS/'dataset.json')
    rows=[r for r in source['rows'] if r['role'] in ('train','selection','calibration')]
    if not any(r['role']=='selection' for r in rows):raise ValueError('No selection cases ready')
    folder=ARTIFACTS/'samples'/tag
    contract=dict(feature_version=FEATURE_VERSION,records=rows,train_per_old=train_per_old,train_per_new=train_per_new,seed=SEED)
    if planning:contract['planning']=True
    if (folder/'manifest.json').exists():
        saved=read(folder/'manifest.json')
        if saved['contract_sha256']!=digest(contract):raise ValueError('An existing sample set cannot change silently')
        return folder
    folder.mkdir(parents=True,exist_ok=True)
    indices={key:[] for key in ('train','selection','calibration')}
    for index,record in enumerate(rows):
        episode=Episode(record);eligible=episode.eligible()
        cap=(train_per_old if record['source_kind']=='bootstrap' else train_per_new) if record['role']=='train' else 9000
        if not len(eligible):continue
        rng=np.random.default_rng(int(digest([SEED,record['key']])[:16],16))
        selected=eligible[rng.choice(len(eligible),size=min(cap,len(eligible)),replace=False)]
        indices[record['role']].append(np.c_[np.full(len(selected),index),selected])
    first=Episode(rows[0]);example=first.features(first.eligible()[:1],planning=planning);dimension=example.shape[1]
    counts={}
    for split,parts in indices.items():
        if not parts:continue
        index=np.concatenate(parts).astype(np.int32);counts[split]=len(index)
        x=np.lib.format.open_memmap(folder/(split+'_x.npy'),mode='w+',dtype=np.float32,shape=(len(index),dimension))
        y=np.lib.format.open_memmap(folder/(split+'_y.npy'),mode='w+',dtype=np.uint8,shape=(len(index),6))
        live=np.lib.format.open_memmap(folder/(split+'_up.npy'),mode='w+',dtype=bool,shape=(len(index),))
        next_bin=np.lib.format.open_memmap(folder/(split+'_next.npy'),mode='w+',dtype=np.uint8,shape=(len(index),))
        legacy=np.lib.format.open_memmap(folder/(split+'_legacy.npy'),mode='w+',dtype=bool,shape=(len(index),))
        for e,record in enumerate(rows):
            positions=np.flatnonzero(index[:,0]==e)
            if not len(positions):continue
            episode=Episode(record);local=index[positions,1:];t,r,j=local.T
            for start in range(0,len(local),512):
                stop=start+512;x[positions[start:stop]]=episode.features(local[start:stop],planning=planning)
            y[positions]=episode.targets[t,r,j];live[positions]=episode.currently_up[t,r,j]
            next_bin[positions]=episode.next_bin[t,r,j];legacy[positions]=episode.legacy_support[t,r,j]
        np.save(folder/(split+'_index.npy'),index,allow_pickle=False)
        for array in (x,y,live,next_bin,legacy):array.flush()
        print(json.dumps(dict(stage='features',split=split,samples=len(index),dimension=dimension)),flush=True)
    write(folder/'manifest.json',dict(contract_sha256=digest(contract),contract=contract,counts=counts,
                                     dimension=dimension,created_at=now()))
    return folder


class Samples:
    def __init__(self,folder,split):
        self.folder=Path(folder);self.manifest=read(self.folder/'manifest.json')
        self.records=self.manifest['contract']['records']
        for name,suffix in [('x','x'),('y','y'),('up','up'),('next','next'),('legacy','legacy'),('index','index')]:
            setattr(self,name,np.load(self.folder/(split+'_'+suffix+'.npy'),mmap_mode='r',allow_pickle=False))

    @lru_cache(maxsize=48)
    def episode(self,index):
        return Episode(self.records[index])

    def neural_batch(self,positions,use_energy=True):
        positions=np.asarray(positions);index=self.index[positions]
        sequences=np.empty((len(index),40,111),np.float32)
        for e in np.unique(index[:,0]):
            selected=np.flatnonzero(index[:,0]==e)
            sequences[selected]=self.episode(int(e)).sequence(index[selected,1:],use_energy)
        static=np.asarray(self.x[positions])
        if not use_energy:static=without_energy(static)
        return sequences,np.nan_to_num(static,nan=0.,posinf=0.,neginf=0.),np.asarray(self.y[positions],np.float32),np.asarray(self.up[positions]),np.asarray(self.next[positions],np.int64)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='engineering');parser.add_argument('--planning',action='store_true');args=parser.parse_args()
    print(make(args.tag,planning=args.planning))
