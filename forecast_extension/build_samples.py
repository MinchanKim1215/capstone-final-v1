"""New-only training matrix, plus a merged matrix for tree fitting."""
from pathlib import Path
import argparse
import numpy as np
from . import ARTIFACTS
from .contracts import read,write,sha,digest,SEED,now,OLD_SAMPLES
from forecast_rebuild.dataset import Samples,Episode


NAMES=[('x',np.float32,(728,)),('y',np.uint8,(6,)),('up',bool,()),('next',np.uint8,()),('legacy',bool,())]


def new_only():
    folder=ARTIFACTS/'samples/new_only';manifest=folder/'manifest.json'
    if manifest.exists():return folder
    data=read(ARTIFACTS/'dataset.json')
    rows=[r for r in data['rows'] if (r['source_kind'] in ('fresh_v9','augmented_v9') and r['role'] in ('train','selection'))]
    indices={'train':[],'selection':[]}
    for e,record in enumerate(rows):
        if record['role']=='train':local=np.load(ARTIFACTS/'row_indices'/(record['key']+'.npy'),allow_pickle=False)
        else:
            eligible=Episode(record).eligible(legacy=True)
            rng=np.random.default_rng(int(digest(['fresh-selection-probe',SEED,record['key']])[:16],16))
            local=eligible[rng.choice(len(eligible),min(900,len(eligible)),replace=False)]
        indices[record['role']].append(np.c_[np.full(len(local),e),local])
    folder.mkdir(parents=True,exist_ok=True);counts={}
    for split,parts in indices.items():
        index=np.concatenate(parts).astype(np.int32);counts[split]=len(index)
        arrays={name:np.lib.format.open_memmap(folder/(split+'_'+name+'.npy'),mode='w+',dtype=dtype,shape=(len(index),*tail)) for name,dtype,tail in NAMES}
        for e,record in enumerate(rows):
            take=np.flatnonzero(index[:,0]==e)
            if not len(take):continue
            episode=Episode(record);local=index[take,1:];t,r,j=local.T
            for start in range(0,len(local),512):arrays['x'][take[start:start+512]]=episode.features(local[start:start+512],True,True)
            for name,source in [('y','targets'),('up','currently_up'),('next','next_bin'),('legacy','legacy_support')]:arrays[name][take]=getattr(episode,source)[t,r,j]
            if e%16==0:print(dict(stage='new_samples',split=split,record=e,total=len(rows)),flush=True)
        np.save(folder/(split+'_index.npy'),index,allow_pickle=False)
        for value in arrays.values():value.flush()
    write(manifest,dict(contract=dict(records=rows,planning=True,source_dataset_sha256=sha(ARTIFACTS/'dataset.json'),seed=SEED),
                        dimension=728,counts=counts,created_at=now(),contains_confirmation=False))
    return folder


def merged():
    new=new_only();folder=ARTIFACTS/'samples/merged_trees';manifest=folder/'manifest.json'
    if manifest.exists():return folder
    old=Samples(OLD_SAMPLES,'train');fresh=Samples(new,'train');old_probe=Samples(ARTIFACTS/'samples/balanced_probe','selection')
    new_probe=Samples(new,'selection');records=old.records+fresh.records+old_probe.records
    # The predecessor manifest also lists unused dev records. Row offsets are
    # explicit so no calibration row can accidentally become a train index.
    folder.mkdir(parents=True,exist_ok=True);counts={}
    for split,parts in [('train',[(old,0),(fresh,len(old.records))]),
                        ('selection',[(old_probe,len(old.records)+len(fresh.records)),(new_probe,len(old.records))])]:
        total=sum(len(part.y) for part,_ in parts);counts[split]=total
        arrays={name:np.lib.format.open_memmap(folder/(split+'_'+name+'.npy'),mode='w+',dtype=dtype,shape=(total,*tail)) for name,dtype,tail in NAMES}
        indices=[];offset=0
        for part,record_offset in parts:
            for start in range(0,len(part.y),8192):
                stop=min(start+8192,len(part.y))
                for name,_,_ in NAMES:arrays[name][offset+start:offset+stop]=getattr(part,name)[start:stop]
            index=np.array(part.index);index[:,0]+=record_offset;indices.append(index);offset+=len(part.y)
        np.save(folder/(split+'_index.npy'),np.concatenate(indices).astype(np.int32),allow_pickle=False)
        for array in arrays.values():array.flush()
    for e in np.unique(np.load(folder/'train_index.npy',mmap_mode='r')[:,0]):
        if records[int(e)]['role']!='train':raise ValueError('Non-training row in merged training matrix')
    write(manifest,dict(contract=dict(records=records,planning=True,old_manifest_sha256=sha(OLD_SAMPLES/'manifest.json'),new_manifest_sha256=sha(new/'manifest.json')),
                        dimension=728,counts=counts,created_at=now(),calibration_used=False,confirmation_used=False))
    return folder


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--merged',action='store_true');a=p.parse_args();print(merged() if a.merged else new_only())

