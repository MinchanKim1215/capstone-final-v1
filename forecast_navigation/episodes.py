"""Small exact metric cohorts; avoid repeatedly rebuilding full RF histories."""
from types import SimpleNamespace
import numpy as np


def cohort(item,task):
    with np.load(item['cohort_path'],allow_pickle=False) as a:d={k:a[k] for k in a.files}
    ends=d['ends'];n=int(d['n']);ns=int(d['ns']);q=len(ends)
    if task=='link':
        valid=np.ones((q,1,n),bool);valid[:,0,ns]=False
        energy=np.full((len(d['links']),1,n,6),np.nan,np.float16);energy[...,3:5]=0;energy[ends-1,0]=d['energy']
        return SimpleNamespace(record=dict(item['source_record'],n_search=ns),ends=ends,links=d['links'],
            targets=d['link_y'][:,None],valid=valid,currently_up=d['link_up'][:,None],
            legacy_support=d['link_common'][:,None],energy=energy)
    links=np.zeros((len(d['health']),2,2),bool);links[:,1,0]=d['health']
    return SimpleNamespace(record=dict(item['source_record'],n_search=1),ends=ends,links=links,
        targets=d['service_y'][:,None,None],valid=np.ones((q,1,1),bool),currently_up=d['service_up'][:,None,None],
        legacy_support=d['service_common'][:,None,None],energy=None)

