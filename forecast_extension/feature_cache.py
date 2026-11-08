"""Lossless static feature reuse across frozen predictors, with disk reserve."""
from pathlib import Path
import os
import shutil
import numpy as np
from . import ARTIFACTS,ROOT
from .contracts import read,write,sha,now
from forecast_rebuild import ROOT as PARENT_ROOT
from forecast_rebuild.dataset import Episode,without_energy


class CachedEpisode(Episode):
    def __init__(self,record):
        super().__init__(record);self._static=None;self._lookup=None;self._cache_attempted=False

    def _prepare_static(self):
        self._cache_attempted=True;source=Path(self.record['cache_path']);folder=ARTIFACTS/'static_features'/self.record['key']
        manifest=folder/'manifest.json';contract=dict(version='lossless-v9-static-v1',raw_source_sha256=self.record['array_sha256'],
            RF_manifest_sha256=sha(source/'manifest.json'),plan_manifest_sha256=sha(source/'planning_manifest.json'),
            feature_source_sha256=sha(PARENT_ROOT/'features.py'),adapter_sha256=sha(__file__))
        if manifest.exists():
            if read(manifest)['contract']!=contract:raise ValueError('Static feature cache contract changed')
        else:
            query=np.argwhere(self.valid);required=len(query)*728*4+int(np.prod(self.valid.shape))*4
            if shutil.disk_usage(ARTIFACTS).free<required+20*2**30:return
            folder.mkdir(parents=True,exist_ok=True)
            x=np.lib.format.open_memmap(folder/'x.working.npy',mode='w+',dtype=np.float32,shape=(len(query),728))
            for start in range(0,len(query),768):x[start:start+768]=super().features(query[start:start+768],True,True)
            x.flush();del x
            lookup=np.full(self.valid.shape,-1,np.int32);t,r,j=query.T;lookup[t,r,j]=np.arange(len(query),dtype=np.int32)
            np.save(folder/'lookup.working.npy',lookup,allow_pickle=False)
            os.replace(folder/'x.working.npy',folder/'x.npy');os.replace(folder/'lookup.working.npy',folder/'lookup.npy')
            write(manifest,dict(contract=contract,rows=len(query),bytes=required,created_at=now(),labels_are_not_features=True))
        self._static=np.load(folder/'x.npy',mmap_mode='r',allow_pickle=False);self._lookup=np.load(folder/'lookup.npy',mmap_mode='r',allow_pickle=False)

    def features(self,rows,use_energy=True,planning=False):
        if not self._cache_attempted:self._prepare_static()
        if self._static is None:return super().features(rows,use_energy,planning)
        t,r,j=np.asarray(rows,int).T;indices=self._lookup[t,r,j]
        if np.any(indices<0):return super().features(rows,use_energy,planning)
        x=np.asarray(self._static[indices])
        if not planning:x=x[:,:646]
        if not use_energy:x=without_energy(x)
        return x

