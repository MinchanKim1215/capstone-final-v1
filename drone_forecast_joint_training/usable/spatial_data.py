"""48개 기본 관측을 보존하고 지도 선행 입력 12개를 별도 파일에 추가."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from swarm_sim.forecast import atomic_npz
from usable.data import DATA,PREPARED,collection,load_observation
from usable.spatial import context,features,NAMES


def one_region(entry,records):
    ctx=context(ROOT/entry['path']);out=[]
    for r in records:
        path=ROOT/r['path']/'spatial_observations.npz'
        if path.exists():
            with np.load(path) as f:
                if 'source_sha256' not in f.files or str(f['source_sha256'])!=sha256_file(ROOT/'usable/spatial.py'):
                    raise ValueError('공간 파생 입력의 코드 버전 불일치')
            continue
        t=time.perf_counter();x,m=load_observation(r);z,zm=features(ctx,x,m);atomic_npz(path,features=z,mask=zm,source_sha256=sha256_file(ROOT/'usable/spatial.py'))
        out.append({'id':r['id'],'sha256':sha256_file(path),'elapsed_s':time.perf_counter()-t})
    print('spatial',entry['key'],len(out),flush=True);return out


def materialize_spatial_bank(name):
    for name in (name,):
        folder=PREPARED/name;meta=read_json(folder/'bank.json');selected=np.load(folder/'selected.npy');n=len(selected)
        xo=folder/'spatial_values.npy';mo=folder/'spatial_mask.npy'
        if xo.exists() and mo.exists():continue
        x=np.lib.format.open_memmap(xo,mode='w+',dtype=np.float32,shape=(n,40,12));m=np.lib.format.open_memmap(mo,mode='w+',dtype=bool,shape=(n,40,12))
        for ep in np.unique(selected[:,0]).astype(int):
            r=meta['records'][ep]
            with np.load(ROOT/r['path']/'spatial_observations.npz') as z:xx,mm=z['features'],z['mask']
            rows=np.flatnonzero(selected[:,0]==ep);k,i,j=selected[rows,1:4].astype(int).T;t=k[:,None]+np.arange(-39,1)
            x[rows]=xx[t,i[:,None],j[:,None]];m[rows]=mm[t,i[:,None],j[:,None]]
        x.flush();m.flush()


def prepare_banks():
    for name in ('train30','val_select','val_cal'):materialize_spatial_bank(name)
    x=np.load(PREPARED/'train30/spatial_values.npy',mmap_mode='r');m=np.load(PREPARED/'train30/spatial_mask.npy',mmap_mode='r')
    count=np.zeros(12,np.int64);s=np.zeros(12);ss=np.zeros(12)
    for first in range(0,len(x),2048):
        a=x[first:first+2048].astype(np.float64);mm=m[first:first+2048];count+=mm.sum((0,1));s+=np.where(mm,a,0).sum((0,1));ss+=np.where(mm,a*a,0).sum((0,1))
    mean=s/np.maximum(count,1);std=np.sqrt(np.maximum(ss/np.maximum(count,1)-mean*mean,0));std[std<1e-6]=1
    norm=read_json(PREPARED/'normalizer.json');norm['mean']+=mean.tolist();norm['std']+=std.tolist();norm['spatial_names']=NAMES
    write_json(PREPARED/'normalizer_spatial.json',norm)


def main():
    manifest,p=collection()
    # 학습과 검증만 선행 처리합니다. 시험 지역의 공간 입력도 선택 동결 뒤에 별도 생성.
    records=[r for r in manifest['records'] if r['split']!='test'];rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(one_region,e,[r for r in records if r['region']==e['key']]) for e in p['maps'] if e['split']!='test']
        for f in as_completed(jobs):rows+=f.result()
    write_json(ROOT/'results/usable/spatial_training_inputs.json',{'files':rows,'test_used':False,'source_sha256':sha256_file(ROOT/'usable/spatial.py')})
    prepare_banks()


if __name__=='__main__':main()
