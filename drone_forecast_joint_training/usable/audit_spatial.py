"""캐시 최적화 전 임시 산출물의 재계산·배열 동일성 확인과 출처 갱신."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from swarm_sim.regions import ROOT,read_json,write_json,sha256_file
from swarm_sim.forecast import atomic_npz
from usable.data import collection,load_observation
from usable.spatial import context,features


def one(entry,records):
    ctx=context(ROOT/entry['path']);rows=[];source=sha256_file(ROOT/'usable/spatial.py')
    for r in records:
        path=ROOT/r['path']/'spatial_observations.npz'
        if not path.exists():continue
        with np.load(path) as f:
            valid='source_sha256' in f.files and str(f['source_sha256'])==source
            if valid:continue
            old,oldmask=f['features'],f['mask']
        x,m=load_observation(r);z,zm=features(ctx,x,m)
        np.testing.assert_array_equal(old,z);np.testing.assert_array_equal(oldmask,zm)
        atomic_npz(path,features=z,mask=zm,source_sha256=source)
        rows.append({'id':r['id'],'same_arrays_after_cache_optimization':True})
    return rows


def main():
    manifest,p=collection();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(one,e,[r for r in manifest['records'] if r['region']==e['key']]) for e in p['maps'] if e['split']!='test']
        for f in as_completed(jobs):rows+=f.result()
    write_json(ROOT/'results/usable/spatial_cache_equivalence.json',{'verified_recomputed_files':rows,'source_sha256':sha256_file(ROOT/'usable/spatial.py')})


if __name__=='__main__':main()
