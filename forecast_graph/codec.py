"""An explicit128-byte beacon, with received RF reports and a tentative8s path."""
import struct
import numpy as np

BYTES=128
MAX_N=16
H=np.arange(9,dtype=float)
VERSION=11


def pack(sender,n,ns,stamp,issued,velocity,queue,attempts,successes,desired,path,last,sinr,power):
    if not 1<=n<=MAX_N or not 0<=sender<n:raise ValueError('Unsupported fleet')
    b=bytearray(BYTES)
    struct.pack_into('<BBBBIIhhHHHIhh',b,0,VERSION,sender,n,ns,
        int(round(stamp*1000)),int(round(issued*1000)),
        *np.clip(np.rint(np.asarray(velocity)*100),-32768,32767).astype(int),
        int(np.clip(queue,0,65535)),int(np.clip(attempts,0,65535)),int(np.clip(successes,0,65535)),
        int(round((issued+3)*1000)),*np.rint(np.asarray(desired)*4).astype(int))
    header=struct.calcsize('<BBBBIIhhHHHIhh')
    if header!=30:raise AssertionError(header)
    xy=np.clip(np.rint(np.asarray(path)*4),-32768,32767).astype('<i2')
    if xy.shape!=(9,2):raise ValueError('Nine timestamped plan positions required')
    b[30:66]=xy.tobytes()
    for j in range(n):
        fresh=last[j]>=0 and stamp>=last[j]-1e-8
        #255 is a missing age;127 a missing signed SINR. Zero is valid SINR.
        s=int(np.clip(np.rint(sinr[j]*2),-128,126)) if fresh else 127
        age=int(np.clip(np.ceil(max(0,stamp-last[j])*2),0,254)) if fresh else 255
        struct.pack_into('<bB',b,66+2*j,s,age)
        struct.pack_into('<b',b,98+j,int(np.clip(np.rint(power[j]),-127,0)) if fresh else -128)
    return np.frombuffer(b,dtype=np.uint8).copy()


def unpack(blob):
    b=bytes(blob)
    if len(b)!=BYTES:raise ValueError('Incorrect beacon length')
    fields=struct.unpack_from('<BBBBIIhhHHHIhh',b,0)
    version,sender,n,ns,stamp,issued,vx,vy,queue,attempts,successes,commit,dx,dy=fields
    if version!=VERSION or not 1<=n<=MAX_N or not 0<=sender<n:raise ValueError('Unsupported beacon')
    reports=np.array([struct.unpack_from('<bB',b,66+2*j) for j in range(n)],float)
    received_power=np.frombuffer(b[98:98+n],dtype=np.int8).astype(float)
    good=(reports[:,0]!=127)&(reports[:,1]!=255)
    return dict(sender=sender,n=n,ns=ns,stamp=stamp/1000,issued=issued/1000,commit_until=commit/1000,
        velocity=np.array([vx,vy])/100,queue=float(queue),attempts=float(attempts),successes=float(successes),
        desired=np.array([dx,dy])/4,path=np.frombuffer(b[30:66],dtype='<i2').reshape(9,2).astype(float)/4,
        sinr=np.where(good,reports[:,0]/2,np.nan),report_age=np.where(good,reports[:,1]/2,np.inf),
        power=np.where(good&(received_power!=-128),received_power,np.nan))


def at(message,absolute_times):
    elapsed=np.asarray(absolute_times)-message['stamp']
    path=message['path']
    return np.stack([np.interp(elapsed,H,path[:,k]) for k in range(2)],-1)

