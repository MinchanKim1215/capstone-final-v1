"""A128B extension; both old and revised plans are available in ONE reception."""
import struct
import numpy as np
from .preview import capture,preview
from forecast_graph.trace import attach


def revise(base,old_nodes,new_nodes,stamp,issued):
    b=np.array(base,copy=True);future=stamp-issued+np.arange(9)
    left=np.minimum((future//3).astype(int),len(new_nodes)-2);alpha=np.clip((future-left*3)/3,0,1)
    xy=new_nodes[left]*(1-alpha[:,None,None])+new_nodes[left+1]*alpha[:,None,None]
    b[:,30:66]=np.rint(xy.transpose(1,0,2)*4).astype('<i2').reshape(len(base),-1).view(np.uint8)
    b[:,114:126]=np.rint(old_nodes[2:].transpose(1,0,2)*4).astype('<i2').reshape(len(base),-1).view(np.uint8)
    b[:,126]=1
    return b


def original_packet(blob):
    b=np.asarray(blob,np.uint8).copy()
    if b.shape!=(128,) or b[126]!=1:raise ValueError('Not a mission-plan extension')
    stamp,issued=struct.unpack_from('<II',b.tobytes(),4);elapsed=(stamp-issued)/1000
    mission=b[30:66].copy().view('<i2').reshape(9,2).astype(float)/4
    commit=3-elapsed
    if not 0<=commit<=3 or commit!=int(commit):raise ValueError('Unsupported issuing phase')
    endpoint=mission[int(commit)];tail=b[114:126].copy().view('<i2').reshape(3,2).astype(float)/4
    knots=np.vstack([endpoint,tail]);old=mission.copy()
    for h in range(9):
        t=elapsed+h
        if t<=3:continue
        left=min(int((t-3)//3),2);fraction=(t-3-left*3)/3
        old[h]=knots[left]*(1-fraction)+knots[left+1]*fraction
    b[30:66]=np.rint(old*4).astype('<i2').reshape(-1).view(np.uint8);b[114:128]=0
    return b


def extend_trace(base_tx,delta):
    b=np.array(base_tx,copy=True);b[...,30:66]=delta[...,:36];b[...,114:126]=delta[...,36:48];b[...,126]=1;return b


def attach_navigation(env):
    tel=attach(env);issue=tel.issue;push=tel.push;tel.nav_tx=[]
    def issued(time_s,desired):
        issue(time_s,desired);tel.original_nodes=np.array(tel.planned_nodes)
        future=preview(env.city,env.design,env.scenario,capture(env.stream),env.baseline_duration,env.k)
        tel.mission_nodes=np.concatenate([tel.original_nodes[:1],future],axis=0)
    def pushed(*args,**kwargs):
        t=tel.t;result=push(*args,**kwargs)
        if t%2==0:
            packets=revise(tel.plan_tx[-1],tel.original_nodes,tel.mission_nodes,t*.5,tel.issued_time)
            for j,p in enumerate(packets):np.testing.assert_equal(original_packet(p),tel.plan_tx[-1][j])
            tel.nav_tx.append(packets)
        return result
    tel.issue=issued;tel.push=pushed;return tel

