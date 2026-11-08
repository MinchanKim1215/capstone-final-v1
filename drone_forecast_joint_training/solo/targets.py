"""Future arrays are offline labels only; never policy inputs."""
import numpy as np


def prediction_targets(links,n_decisions):
    up=np.asarray(links,bool)
    if up.ndim!=3 or up.shape[1:]!=(8,8):raise ValueError('Expected RF availability [T,8,8]')
    y=np.zeros((n_decisions,5,8,4),np.float32)
    valid=np.zeros_like(y,bool)
    bad=~up
    start=bad.copy()
    start[1:] &= up[:-1]
    start[:-1] &= bad[1:]
    if len(start):start[-1]=False
    pair=bad[:-1]&bad[1:]
    for decision in range(1,n_decisions):
        k=6*decision-1
        for relay in range(5):
            i=relay+3
            for j in range(8):
                if j==i or k>=len(up) or not up[k,i,j]:continue
                if k+10<len(up):
                    y[decision,relay,j,0]=np.any(pair[k+1:k+10,i,j])
                    y[decision,relay,j,1]=np.any(start[k+7:k+10,i,j])
                    valid[decision,relay,j,:2]=True
                if k+16<len(up):
                    y[decision,relay,j,2]=np.any(start[k+7:k+16,i,j])
                    y[decision,relay,j,3]=np.any(start[k+1:k+16,i,j])
                    valid[decision,relay,j,2:]=True
    return y,valid
