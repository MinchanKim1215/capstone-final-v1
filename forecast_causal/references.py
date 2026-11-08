"""Prepare new flight inputs while preserving old probabilities and packettruth."""
import numpy as np
from . import ARTIFACTS
from .contracts import read,sha
from .legacy_data import prepare_one as legacy_prepare
from .replay import one as instrument
from .data import prepare_one
from .wire import SLOTS
from forecast_navigation.wire import extend_trace

def prepare_case(record):
    parent=legacy_prepare(dict(key=record['key'],role='confirmation',region=record['region'],source_record=record))
    replay=instrument(parent)
    with np.load(replay['path']) as a:
        tx=a['tx'].copy();tx[...,SLOTS]=0;tx[...,127]=0
        with np.load(ARTIFACTS/'service_cache'/(record['key']+'.npz')) as old:
            for k in ('packet_born','packet_status','packet_delay'):np.testing.assert_equal(a[k],old[k])
    with np.load(ARTIFACTS/'fresh_traces'/(record['key']+'.npz')) as a:original=a['tx'];rx=a['rx']
    with np.load(ARTIFACTS/'deltas'/(record['key']+'.npz')) as a:original=extend_trace(original,a['delta'])
    np.testing.assert_equal(tx,original)
    with np.load(replay['path']) as a:np.testing.assert_equal(a['rx'],rx)
    return prepare_one(replay)
