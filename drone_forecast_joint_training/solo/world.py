from collections import deque
import numpy as np
from cooperative.continuity_world import ContinuityWorld
from solo.contracts import HISTORY


class Recorder:
    def __init__(self):self.values=[];self.masks=[];self.intents=[]
    def observe(self,time,values,masks,intents):
        # Sensor contract: half precision at rest, float32 in the network. No future states.
        x=values[3:8].astype(np.float16);c=intents[3:8].astype(np.float16)
        if not np.isfinite(x).all() or not np.isfinite(c).all():raise FloatingPointError('Observation overflow/nonfinite input')
        self.values.append(x);self.masks.append(masks[3:8].copy());self.intents.append(c)
        return dict(status='recording_only',links=[])


class World(ContinuityWorld):
    def __init__(self,**spec):
        super().__init__(**spec,forecast=False,capture=False)
        self.recorder=Recorder();self.predictor=self.recorder


def history(array,decision):
    end=decision*6
    part=np.asarray(array[max(0,end-HISTORY):end])
    if end==0:
        if isinstance(array,list):raise ValueError('Use initialized empty array shape for first history')
    if len(part)<HISTORY:part=np.concatenate([np.zeros((HISTORY-len(part),*array.shape[1:]),dtype=array.dtype),part])
    return np.swapaxes(part,0,1)


def live_history(recorder):
    shapes=[(5,8,48),(5,8,48),(5,8,9)];out=[]
    for rows,shape,dtype in zip([recorder.values,recorder.masks,recorder.intents],shapes,[np.float16,bool,np.float16]):
        part=np.asarray(rows[-HISTORY:],dtype=dtype).reshape(-1,*shape)
        if len(part)<HISTORY:part=np.concatenate([np.zeros((HISTORY-len(part),*shape),dtype=dtype),part])
        out.append(np.swapaxes(part,0,1))
    return out


def supported(values,masks,frames):
    good=masks[...,41]&masks[...,46]&(values[...,46]>.5)&(frames>=HISTORY)
    good=good.copy();good[np.arange(5),np.arange(3,8)]=False
    return good
