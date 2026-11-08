"""Streaming inference: 0.5-second public observations, 3-second policy decisions."""
from collections import deque
import numpy as np
import torch
from solo.contracts import HISTORY,sources,load_plan,object_hash
from solo.model import load_controller


class StreamingController:
    def __init__(self,checkpoint):
        self.model,self.ck=load_controller(checkpoint)
        if self.ck['source_sha256']!=sources() or self.ck['plan_sha256']!=object_hash(load_plan()):raise ValueError('Checkpoint contract changed')
        self.reset()
    def reset(self):
        self.values=deque(maxlen=HISTORY);self.masks=deque(maxlen=HISTORY);self.intents=deque(maxlen=HISTORY);self.last_time=0.;self.last_action=-3.
    def observe(self,time_s,values,masks,intents):
        if abs(float(time_s)-self.last_time-.5)>1e-5:raise ValueError('Consecutive 0.5-second observations required')
        if np.shape(values)!=(5,8,48) or np.shape(masks)!=(5,8,48) or np.shape(intents)!=(5,8,9):raise ValueError('Expected [5,8,48/9] public observations')
        x=np.asarray(values,np.float16);m=np.asarray(masks,bool);c=np.asarray(intents,np.float16)
        if not np.isfinite(x).all() or not np.isfinite(c).all():raise ValueError('Finite observations required')
        self.values.append(x.copy());self.masks.append(m.copy());self.intents.append(c.copy());self.last_time=float(time_s)
    def act(self,time_s,context,candidates,mask,physics_score,gain,geometry):
        if abs(float(time_s)-self.last_time)>1e-5 or abs(float(time_s)-self.last_action-3)>1e-5:raise ValueError('Decision must follow latest observations at consecutive 3-second intervals')
        shapes=[(5,91),(5,25,12),(5,25),(5,25),(5,25,8),(5,8,46)]
        for a,shape in zip([context,candidates,mask,physics_score,gain,geometry],shapes):
            if np.shape(a)!=shape or not np.isfinite(a).all():raise ValueError('Invalid physical input shape/value: '+str(shape))
        arrays=[]
        for rows,shape,dtype in [(self.values,(5,8,48),np.float16),(self.masks,(5,8,48),bool),(self.intents,(5,8,9),np.float16)]:
            a=np.asarray(rows,dtype=dtype).reshape(-1,*shape)
            if len(a)<HISTORY:a=np.concatenate([np.zeros((HISTORY-len(a),*shape),dtype=dtype),a])
            arrays.append(np.swapaxes(a,0,1))
        support=np.zeros((5,8),bool)
        if len(self.values)==HISTORY:
            x=self.values[-1];m=self.masks[-1];support=m[...,41]&m[...,46]&(x[...,46]>.5)
            support[np.arange(5),np.arange(3,8)]=False
        with torch.no_grad():
            logits,p=self.model.policy(torch.as_tensor(context,dtype=torch.float32),torch.as_tensor(candidates,dtype=torch.float32),
                torch.as_tensor(mask,dtype=torch.bool),torch.as_tensor(physics_score,dtype=torch.float32),torch.as_tensor(gain,dtype=torch.float32),
                *[torch.from_numpy(a) for a in arrays],torch.as_tensor(np.asarray(geometry,np.float16)),torch.from_numpy(support))
        probability=p.numpy().copy();probability[~support]=np.nan;self.last_action=float(time_s)
        return dict(actions=logits.argmax(-1).numpy(),probability=probability,supported=support,
            alarm_enabled=False,lightgbm_used=False,gru_used=False)
