"""Exact service cohorts in the unchanged square-link event metric."""
from types import SimpleNamespace
import numpy as np


def cohort(item):
    with np.load(item['path'],allow_pickle=False) as a:
        return SimpleNamespace(record=dict(item['source_record'],n_search=1),ends=a['ends'],links=a['links'],
            targets=a['y'][:,None,None],valid=np.ones((len(a['ends']),1,1),bool),currently_up=a['up'][:,None,None],
            legacy_support=a['support'][:,None,None],energy=None)
