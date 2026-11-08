"""Past-only graph snapshots; no modifications to parent HISTORY globals."""
import numpy as np
from forecast_graph.features import frame,physics,NODE,EDGE
OFFSETS=np.arange(-38,1,2)
SHORT=np.array([16,17,18,19])
MAX_N=10

def snapshots(local,trace,ends):
    nodes=[];edges=[];states=[]
    for end in ends:
        node,edge,state=frame(local,trace,int(end),0,True)
        nodes.append(node[:MAX_N]);edges.append(edge[:MAX_N,:MAX_N]);states.append(state)
    return np.asarray(nodes,np.float16),np.asarray(edges,np.float16),states

def build(local,trace,ends,city):
    ends=np.asarray(ends,int)
    times=np.unique((ends[:,None]+OFFSETS).ravel())
    if times.min()<1:raise ValueError('Not enough causal history')
    node,edge,_=snapshots(local,trace,times)
    index=np.searchsorted(times,ends[:,None]+OFFSETS).astype(np.int32)
    states=[frame(local,trace,int(end),0,True)[2] for end in ends]
    ph=[];gl=[]
    for start in range(0,len(states),32):
        a,b=physics(city,states[start:start+32]);ph.append(a[:,:,:MAX_N,:MAX_N]);gl.append(b)
    return dict(node=node,edge=edge,index=index,times=times.astype(np.int32),physics=np.concatenate(ph).astype(np.float16),global_features=np.concatenate(gl).astype(np.float16))
