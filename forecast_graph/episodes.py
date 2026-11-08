"""Immutable source readers; matched first-relay physical and service cohorts."""
from types import SimpleNamespace
from functools import lru_cache
from pathlib import Path
import numpy as np
from . import ARTIFACTS
from .contracts import read,sha,V10
from forecast_rebuild.dataset import Episode
from forecast_rebuild.targets import labels,past_support
from forecast_rebuild.features import build,sequence
from forecast_rebuild.planning import public_context,profile


class RawEpisode:
    def __init__(self,record):
        self.record=record
        if sha(record['array_path'])!=record['array_sha256']:raise ValueError('Raw flight changed')
        with np.load(record['array_path'],allow_pickle=False) as a:
            self.values=a['rf_values'];self.masks=a['rf_masks'];self.intents=a['rf_intents'];self.links=a['links']
            ends=a['aux_endpoints'];take=ends>=40;self.ends=ends[take];self.geometry=a['aux_geometry'][take];positions=a['positions']
        from forecast_rebuild.sensors import simulate
        self.energy,meta=simulate(record,self.values,self.links,positions)
        self.energy=self.energy.astype(np.float16)
        for key,value in labels(self.links,self.ends,record['n_search']).items():setattr(self,key,value)
        support=past_support(self.values,self.masks,self.ends,record['n_search'])
        self.legacy_support=support['legacy'];self.available=support['available']
        self.city,self.nominal=public_context(record);self.city.radio.enable_cache()

    def features(self,rows,use_energy=True,planning=False):
        rows=np.asarray(rows,int)
        base=build(self.values,self.masks,self.intents,self.ends,self.geometry,rows,self.energy,use_energy)
        if not planning:return base
        if not hasattr(self,'planning'):
            q=np.argwhere(self.valid[:,:1]);q[:,1]=0
            self.planning=profile(self.city,self.nominal,self.values,self.masks,self.intents,self.ends,self.record['n_search'],only_rows=q)
        t,r,j=rows.T
        return np.concatenate([base,self.planning[t,r,j]],-1)

    def sequence(self,rows,use_energy=True):return sequence(self.values,self.masks,self.intents,self.ends,rows,self.energy,use_energy)


def physical(record):return Episode(record) if record.get('cache_path') else RawEpisode(record)


def cohort(record,task,ep=None):
    ep=physical(record) if ep is None else ep
    if task=='link':
        value=SimpleNamespace(record=record,links=ep.links,ends=ep.ends,energy=ep.energy[:,:1])
        for key in ('targets','valid','currently_up','legacy_support','available'):setattr(value,key,np.asarray(getattr(ep,key))[:,:1])
        return value
    folder=ARTIFACTS if record['role']=='confirmation' else V10
    with np.load(folder/'service_cache'/(record['key']+'.npz'),allow_pickle=False) as a:health=a['service_healthy']
    links=np.zeros((len(health),2,2),bool);links[:,1,0]=health
    value=SimpleNamespace(record=dict(record,n_search=1),links=links,ends=ep.ends,energy=ep.energy[:,:1])
    for key,item in labels(links,ep.ends,1).items():setattr(value,key,item)
    value.legacy_support=np.zeros(value.valid.shape,bool);value.legacy_support[:,0,0]=np.asarray(ep.legacy_support[:,0]).any(1)
    value.available=value.legacy_support;return value

