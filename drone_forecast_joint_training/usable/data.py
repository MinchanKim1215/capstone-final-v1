"""지역·전체 비행 단위 분할 및 지도학습 자료. 미래 채널은 정답 전용."""
from pathlib import Path
import json
import numpy as np
from swarm_sim.forecast_data import window_labels
from swarm_sim.regions import ROOT, read_json, write_json, sha256_file
from upgrade.map_setup import digest
from usable.beacons import data_threshold_db

DATA=ROOT/'datasets/usable45';PREPARED=DATA/'prepared';F=48;W=40


def collection():
    m=read_json(DATA/'collection_manifest.json');p=read_json(ROOT/'configs/usable_protocol.json')
    if not m['complete'] or len(m['records'])!=270 or m['protocol_sha256']!=p['protocol_sha256']:
        raise ValueError('45지역 × 2방향 × 3조건 수집 완료 필요')
    return m,p


def load_observation(r,check=False):
    folder=ROOT/r['path']
    if check and sha256_file(folder/'observations.npz')!=r['metadata']['files_sha256']['observations.npz']:
        raise ValueError('관측 파일 해시 불일치')
    with np.load(folder/'observations.npz') as z:x,m=z['features'],z['mask']
    if x.shape[-1]!=F or not np.isfinite(x).all() or np.any(x[~m]!=0):raise ValueError('입력 오류')
    return x,m


def index_episode(r,p):
    path=PREPARED/'index'/(r['id']+'.npz')
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    x,_=load_observation(r,True)
    folder=ROOT/r['path']
    if sha256_file(folder/'truth.npz')!=r['metadata']['files_sha256']['truth.npz']:raise ValueError('정답 해시 오류')
    with np.load(folder/'truth.npz') as z:up=z['link_available'];sinr=z['minimum_bidirectional_sinr_db']
    # 20초 중 성공 비콘 관측 구간 최소 3회. 정답을 보지 않는 관측 가능 조건입니다.
    idx,y10=window_labels(up,x[...,33]>0,W,20,2,2,3)
    k,i,j=idx.T
    pair=(~up[:-1])&(~up[1:])
    prefix=np.concatenate((np.zeros((1,*up.shape[1:]),np.int32),np.cumsum(pair,axis=0)),axis=0)
    y5=(prefix[k+10,i,j]-prefix[k+1,i,j]>0).astype(np.uint8)
    # 향후 5초 중 최소 SINR는 보조 정답이며 입력 배열에 삽입하지 않습니다.
    future=np.min(sinr[k[:,None]+np.arange(1,11),i[:,None],j[:,None]],axis=1)
    aux=np.clip((future-data_threshold_db())/10.,-3,3).astype(np.float32)
    out={'index':idx,'y':y5,'y10':y10,'sinr_target':aux,'current_available':up[k,i,j]}
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**out)
    return out


def validation_partition(records):
    # 같은 비행의 겹치는 창을 선택/보정 양쪽에 넣지 않습니다. 각 지역에서 3비행씩 분리합니다.
    a=[];b=[]
    regions=sorted({r['region'] for r in records})
    for r in records:
        d=0 if r['metadata']['direction']=='east_west' else 1
        kind={'clear':0,'continuous':1,'intermittent':2}[r['metadata']['kind']]
        (a if d==(kind+regions.index(r['region']))%2 else b).append(r)
    return a,b


def materialize(name,records,p,per_region,seed):
    dest=PREPARED/name
    if (dest/'bank.json').exists():return read_json(dest/'bank.json')
    rng=np.random.default_rng(seed);parts=[]
    for region in sorted({r['region'] for r in records}):
        candidates=[]
        for ep,r in enumerate(records):
            if r['region']!=region:continue
            ix=index_episode(r,p);q=np.flatnonzero(ix['current_available'])
            if len(q):candidates.append(np.column_stack((np.full(len(q),ep),ix['index'][q],ix['y'][q],ix['y10'][q],ix['sinr_target'][q])))
        c=np.concatenate(candidates);parts.append(c[rng.choice(len(c),min(per_region,len(c)),replace=False)])
    selected=np.concatenate(parts);dest.mkdir(parents=True,exist_ok=True)
    n=len(selected);values=np.lib.format.open_memmap(dest/'values.npy',mode='w+',dtype=np.float32,shape=(n,W,F))
    mask=np.lib.format.open_memmap(dest/'mask.npy',mode='w+',dtype=bool,shape=(n,W,F))
    for ep in np.unique(selected[:,0]).astype(int):
        x,m=load_observation(records[ep]);rows=np.flatnonzero(selected[:,0]==ep)
        k,i,j=selected[rows,1:4].astype(int).T;t=k[:,None]+np.arange(-W+1,1)
        values[rows]=x[t,i[:,None],j[:,None]];mask[rows]=m[t,i[:,None],j[:,None]]
    values.flush();mask.flush();np.save(dest/'selected.npy',selected)
    meta={'name':name,'samples':n,'features':F,'positive_fraction':float(selected[:,4].mean()),
          'records':records,'regions':sorted({r['region'] for r in records}),'per_region_max':per_region,
          'sampling':'label independent uniform sample per region of currently available, beacon-observed links',
          'protocol_sha256':p['protocol_sha256'],'seed':seed}
    write_json(dest/'bank.json',meta);return meta


def fit_normalizer():
    dest=PREPARED/'normalizer.json'
    if dest.exists():return read_json(dest)
    x=np.load(PREPARED/'train30/values.npy',mmap_mode='r');m=np.load(PREPARED/'train30/mask.npy',mmap_mode='r')
    total=np.zeros(F);square=np.zeros(F);count=np.zeros(F,np.int64)
    for first in range(0,len(x),2048):
        xx=np.asarray(x[first:first+2048],np.float64);mm=m[first:first+2048]
        total+=np.where(mm,xx,0).sum(axis=(0,1));square+=np.where(mm,xx*xx,0).sum(axis=(0,1));count+=mm.sum(axis=(0,1))
    mean=total/np.maximum(count,1);std=np.sqrt(np.maximum(square/np.maximum(count,1)-mean**2,0));std[std<1e-6]=1
    out={'mean':mean.tolist(),'std':std.tolist(),'counts':count.tolist(),
         'fit':'training bank observed values only; repeated history frames weighted by sampled windows',
         'training_regions':read_json(PREPARED/'train30/bank.json')['regions']}
    write_json(dest,out);return out


def prepare():
    m,p=collection();r=m['records'];train=[x for x in r if x['split']=='train'];va,vc=validation_partition([x for x in r if x['split']=='val'])
    for name,part,num,seed in [('train30',train,8192,2411),('val_select',va,2048,2412),('val_cal',vc,4096,2413)]:
        meta=materialize(name,part,p,num,seed);print(name,meta['samples'],meta['positive_fraction'],flush=True)
    fit_normalizer()
    write_json(PREPARED/'split_audit.json',{'train_ids':[x['id'] for x in train],
              'validation_select_ids':[x['id'] for x in va],'validation_calibration_ids':[x['id'] for x in vc],
              'test_ids':[x['id'] for x in r if x['split']=='test'],'test_previously_inspected_geography':True})


class Bank:
    def __init__(self,name,features):
        self.meta=read_json(PREPARED/name/'bank.json');self.values=np.load(PREPARED/name/'values.npy',mmap_mode='r')
        self.mask=np.load(PREPARED/name/'mask.npy',mmap_mode='r');self.selected=np.load(PREPARED/name/'selected.npy',mmap_mode='r')
        self.features=np.asarray(features,int);norm=read_json(PREPARED/'normalizer.json')
        self.mean=np.asarray(norm['mean'],np.float32)[self.features];self.std=np.asarray(norm['std'],np.float32)[self.features]
        self.region=np.array([self.meta['records'][int(e)]['region'] for e in self.selected[:,0]])
    def __len__(self):return len(self.selected)
    def batch(self,idx):
        import torch
        x=self.values[idx][:,:,self.features].copy();m=self.mask[idx][:,:,self.features].copy()
        x=np.where(m,np.clip((x-self.mean)/self.std,-8,8),0).astype(np.float32)
        return torch.from_numpy(x),torch.from_numpy(m),torch.from_numpy(self.selected[idx,4:7].astype(np.float32))


if __name__=='__main__':prepare()
