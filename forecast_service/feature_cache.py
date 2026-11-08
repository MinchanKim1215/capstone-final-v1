"""Reuse parent input caches read-only; any new cache lives in this study."""
import os
import shutil
import numpy as np
from pathlib import Path
from . import ARTIFACTS
from .contracts import read,write,sha,V9
from forecast_rebuild.dataset import Episode
from forecast_rebuild import ROOT as PARENT


class FeatureEpisode(Episode):
    def __init__(self,record):
        super().__init__(record);self.cached=None;self.lookup=None;self.attempted=False

    def features(self,rows,use_energy=True,planning=False):
        if not self.attempted:
            self.attempted=True;old=V9/'static_features'/self.record['key'];own=ARTIFACTS/'static_features'/self.record['key']
            contract=dict(raw_source_sha256=self.record['array_sha256'],RF_manifest_sha256=sha(Path(self.record['cache_path'])/'manifest.json'),
                plan_manifest_sha256=sha(Path(self.record['cache_path'])/'planning_manifest.json'),feature_source_sha256=sha(PARENT/'features.py'))
            if (old/'manifest.json').exists():
                saved=read(old/'manifest.json')['contract']
                if any(saved[k]!=v for k,v in contract.items()):raise ValueError('Parent static cache input changed')
                folder=old
            else:
                folder=own
                if not (folder/'manifest.json').exists():
                    q=np.argwhere(self.valid);required=len(q)*728*4
                    if shutil.disk_usage(ARTIFACTS).free>required+20*2**30:
                        folder.mkdir(parents=True,exist_ok=True)
                        x=np.lib.format.open_memmap(folder/'x.working.npy',mode='w+',dtype=np.float32,shape=(len(q),728))
                        for i in range(0,len(q),768):x[i:i+768]=super().features(q[i:i+768],True,True)
                        x.flush();del x;lookup=np.full(self.valid.shape,-1,np.int32);t,r,j=q.T;lookup[t,r,j]=np.arange(len(q))
                        np.save(folder/'lookup.working.npy',lookup,allow_pickle=False)
                        os.replace(folder/'x.working.npy',folder/'x.npy');os.replace(folder/'lookup.working.npy',folder/'lookup.npy')
                        write(folder/'manifest.json',dict(contract=contract))
                if (folder/'manifest.json').exists() and read(folder/'manifest.json')['contract']!=contract:raise ValueError('Study input cache changed')
            if (folder/'manifest.json').exists():
                self.cached=np.load(folder/'x.npy',mmap_mode='r');self.lookup=np.load(folder/'lookup.npy',mmap_mode='r')
        if self.cached is None:return super().features(rows,use_energy,planning)
        t,r,j=np.asarray(rows,int).T;index=self.lookup[t,r,j]
        if (index<0).any():return super().features(rows,use_energy,planning)
        x=np.asarray(self.cached[index])
        if not planning:x=x[:,:646]
        if not use_energy:
            from forecast_rebuild.dataset import without_energy
            x=without_energy(x)
        return x
