"""Only local measurements and actually received summaries are exposed here."""
import numpy as np
from .wire import Received

def received_history(tx,rx,observed,n,ns):
    received=np.unpackbits(np.asarray(rx,np.uint8),axis=-1)[...,:n].astype(bool);cache=Received(n,ns);parts=[];known=[]
    for q,local in enumerate(observed):
        packets=[tx[q//2,j] for j in range(n) if j!=ns and received[q//2,ns,j]] if q%2==0 else []
        own=tx[q//2,ns] if q%2==0 else None;now=(q+1)*.5;cache.observe(now,packets,own);a,m=cache.snapshot(now,local[ns]);parts.append(a);known.append(m)
    return np.asarray(parts,np.float32),np.asarray(known,bool)

def from_replay(path):
    with np.load(path,allow_pickle=False) as a:return received_history(a['tx'],a['rx'],a['observed'],int(a['n']),int(a['ns']))
