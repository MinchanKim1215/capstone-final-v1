"""Received-only temporal graphs; no packet truth, RF truth or future action API."""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from .codec import MAX_N
from usable.beacons import data_threshold_db

NODE=34
EDGE=4
PHYS=3
GLOBAL=24
HISTORY=np.array([-6,-4,-2,0])
H=np.arange(9,dtype=float)
THRESHOLD=float(data_threshold_db())


@dataclass
class LocalHistory:
    values: object
    masks: object
    intents: object
    n: int
    ns: int


class ReceivedTrace:
    def __init__(self,tx,rx,n,ns):
        self.n=n;self.ns=ns
        b=np.asarray(tx,np.uint8)
        self.stamp=b[...,4:8].copy().view('<u4').reshape(b.shape[:2])/1000
        self.issued=b[...,8:12].copy().view('<u4').reshape(b.shape[:2])/1000
        self.velocity=b[...,12:16].copy().view('<i2').reshape(*b.shape[:2],2)/100
        self.queue=b[...,16:18].copy().view('<u2').reshape(b.shape[:2]).astype(float)
        self.attempts=b[...,18:20].copy().view('<u2').reshape(b.shape[:2]).astype(float)
        self.successes=b[...,20:22].copy().view('<u2').reshape(b.shape[:2]).astype(float)
        self.commit=b[...,22:26].copy().view('<u4').reshape(b.shape[:2])/1000
        self.desired=b[...,26:30].copy().view('<i2').reshape(*b.shape[:2],2)/4
        self.paths=b[...,30:66].copy().view('<i2').reshape(*b.shape[:2],9,2)/4
        reports=b[...,66:98].reshape(*b.shape[:2],16,2)[:,:,:n]
        s=reports[...,0].astype(np.int16);s=np.where(s>=128,s-256,s)
        good=(s!=127)&(reports[...,1]!=255)
        self.sinr=np.where(good,s/2,np.nan)
        self.age=np.where(good,reports[...,1]/2,np.inf)
        self.power=b[...,98:98+n].copy().view(np.int8).astype(float)
        self.power=np.where(good&(self.power!=-128),self.power,np.nan)
        received=np.unpackbits(np.asarray(rx,np.uint8),axis=-1)[...,:n].astype(bool)
        for j in range(n):received[:,j,j]=True
        self.latest=np.maximum.accumulate(np.where(received,np.arange(len(b))[:,None,None],-1),axis=0)

    def snapshot(self,end,observer):
        #A beacon sent at integer s is received before s+.5. Only frames<end exist.
        k=(int(end)-1)//2
        if k>=len(self.latest):raise ValueError('Unreceived future trace access')
        ix=self.latest[k,observer];safe=np.maximum(ix,0);j=np.arange(self.n)
        now=end*.5;stamp=self.stamp[safe,j];age=now-stamp
        known=(ix>=0)&(age<=10)
        paths=self.paths[safe,j]
        absolute=now+H
        elapsed=np.clip(absolute[None,:]-stamp[:,None],0,8)
        lo=np.floor(elapsed).astype(int);hi=np.minimum(lo+1,8);f=elapsed-lo
        future=paths[j[:,None],lo]*(1-f[...,None])+paths[j[:,None],hi]*f[...,None]
        return dict(known=known,age=age,position=future[:,0],future=future,
            velocity=self.velocity[safe,j],queue=self.queue[safe,j],attempts=self.attempts[safe,j],
            successes=self.successes[safe,j],commit=np.maximum(0,self.commit[safe,j]-now),
            issued_age=np.maximum(0,now-self.issued[safe,j]),desired=self.desired[safe,j],
            sinr=self.sinr[safe,j].T,report_age=(self.age[safe,j]+age[:,None]).T)


def frame(local,trace,end,observer,shared):
    n,ns=local.n,local.ns;i=ns+observer
    x=np.asarray(local.values[end-1,observer],float);m=np.asarray(local.masks[end-1,observer],bool)
    own=x[0,12:14];ownv=x[0,:2]
    known=m[:,[34,35,36,37,30]].all(-1)&(x[:,30]<=10)
    position=own+x[:,34:36];velocity=ownv+x[:,36:38];age=x[:,30].copy()
    known[i]=True;position[i]=own;velocity[i]=ownv;age[i]=0
    future=np.clip(position[:,None,:]+H[None,:,None]*velocity[:,None,:],-195,195)
    queues=np.zeros(n);attempts=np.zeros(n);successes=np.zeros(n);commit=np.zeros(n);issued_age=np.zeros(n)
    desired=position.copy();queues[i]=x[0,11];attempts[i]=x[:,5].sum();successes[i]=x[:,6].sum()
    c=np.asarray(local.intents[end-1,observer],float)
    plan_known=(c[:,8]>.5)&known
    desired[plan_known]=position[plan_known]+c[plan_known,2:4]
    commit[plan_known]=np.maximum(0,3-c[plan_known,4]);issued_age[plan_known]=c[plan_known,4]
    desired[i]=own+c[0,:2];commit[i]=c[0,5];issued_age[i]=max(0,3-c[0,5]);plan_known[i]=True
    sinr=np.full((n,n),np.nan);report_age=np.full((n,n),np.inf)
    inbound=m[:,29]&m[:,30];outbound=m[:,31]&m[:,32]
    sinr[np.arange(n)[inbound],i]=x[inbound,29];report_age[np.arange(n)[inbound],i]=x[inbound,30]
    sinr[i,np.arange(n)[outbound]]=x[outbound,31];report_age[i,np.arange(n)[outbound]]=x[outbound,32]
    if shared:
        s=trace.snapshot(end,i);take=s['known'];plan_known|=take
        for target,name in [(position,'position'),(velocity,'velocity'),(future,'future'),(age,'age'),
                (queues,'queue'),(attempts,'attempts'),(successes,'successes'),(commit,'commit'),
                (issued_age,'issued_age'),(desired,'desired')]:target[take]=s[name][take]
        known|=take
        good=np.isfinite(s['sinr'])&(s['report_age']<=10)&take[None,:]
        good &= ~np.isfinite(sinr)|(s['report_age']<report_age)
        sinr[good]=s['sinr'][good];report_age[good]=s['report_age'][good]
    node=np.zeros((MAX_N,NODE),np.float32)
    node[:n,0]=np.arange(n)>=ns;node[i,1]=1;node[:n,2]=known;node[:n,3]=np.where(known,age/10,1)
    node[:n,4:6]=np.where(known[:,None],position/200,0)
    node[:n,6:8]=np.where(known[:,None],velocity/5,0)
    node[:n,8]=queues/64;node[:n,9]=attempts/100
    node[:n,10]=successes/np.maximum(attempts,1);node[:n,11]=commit/3
    node[:n,12:14]=np.where(plan_known[:,None],desired/200,0);node[:n,14]=plan_known
    node[:n,15]=np.minimum(issued_age/10,1)
    node[:n,16:]=np.where(plan_known[:,None,None],(future-position[:,None])/40,0).reshape(n,18)
    edge=np.zeros((MAX_N,MAX_N,EDGE),np.float32)
    sk=np.isfinite(sinr)&(report_age<=10);np.fill_diagonal(sk,False)
    edge[:n,:n,0]=np.where(sk,(sinr-THRESHOLD)/30,0)
    edge[:n,:n,1]=sk;edge[:n,:n,2]=np.where(sk,report_age/10,1)
    ack=m[:,7]&(x[:,5]>0);edge[i,np.arange(n)[ack],3]=x[ack,7]
    return node,edge,dict(position=position,velocity=velocity,future=future,known=known,
        sinr=sinr,report_age=report_age,observer=i,n=n,ns=ns)


def physics(city,states):
    b=len(states);n=states[0]['n'];ns=states[0]['ns']
    if any(s['n']!=n for s in states):raise ValueError('Physics batches are one fleet')
    pos=np.stack([s['position'] for s in states]);vel=np.stack([s['velocity'] for s in states])
    plan=np.stack([s['future'] for s in states]).transpose(0,2,1,3)
    cv=np.clip(pos[:,None]+H[None,:,None,None]*vel[:,None],-195,195)
    known=np.stack([s['known'] for s in states]);pair=known[:,:,None]&known[:,None,:]
    def loss(paths):
        a=np.broadcast_to(paths[:,:,:,None,:],(b,9,n,n,2));z=np.broadcast_to(paths[:,:,None,:,:],a.shape)
        valid=np.broadcast_to(pair[:,None],(b,9,n,n)).copy()
        ii=np.arange(n);valid[:,:,ii,ii]=False
        out=np.zeros((b,9,n,n),float)
        if valid.any():
            aa=a[valid];bb=z[valid]
            result=city.radio.propagation(np.c_[aa,np.full(len(aa),50)],np.c_[bb,np.full(len(bb),50)],city._forecast_scenario)[0]
            out[valid]=result
        return out
    lp=loss(plan);lc=loss(cv)
    observed=np.stack([s['sinr'] for s in states])-THRESHOLD
    ages=np.stack([s['report_age'] for s in states]);ok=np.isfinite(observed)&(ages<=10)
    margin=np.where(ok,observed,10-lp[:,0]+107-THRESHOLD)
    mp=margin[:,None]-(lp-lp[:,:1]);mc=margin[:,None]-(lc-lc[:,:1])
    #Observation age and OU channel variation are explicit study assumptions.
    sigma=np.sqrt(.5**2+1.5**2*(1-np.exp(-2*(H[None,:,None,None]+np.where(ok,ages,10)[:,None])/4)))
    alive=.5*(ndtr(mp/sigma)+ndtr(mc/sigma))
    alive*=pair[:,None]
    out=np.zeros((b,9,MAX_N,MAX_N,PHYS),np.float32)
    out[:,:,:n,:n,0]=np.clip(mp,-90,90)/30
    out[:,:,:n,:n,1]=np.clip(mc,-90,90)/30
    out[:,:,:n,:n,2]=alive
    out[:,:,:n,:n,:2]*=pair[:,None,:,:,None]
    #A fixed, seed-independent common-random-number rollout. Hidden case seeds
    #and true jammer/shadow/future motion never enter this computation.
    rng=np.random.default_rng(1104);noise=rng.normal(size=(16,9,n,n))
    noise=(noise+noise.transpose(0,1,3,2))/np.sqrt(2)
    noisy=mp[:,None]+sigma[:,None]*noise[None]
    noisy_cv=mc[:,None]+sigma[:,None]*noise[None]
    adj=(np.minimum(noisy,noisy.transpose(0,1,2,4,3))>=0)&pair[:,None,None]
    adj_cv=(np.minimum(noisy_cv,noisy_cv.transpose(0,1,2,4,3))>=0)&pair[:,None,None]
    ii=np.arange(n);adj[:,:,:,ii,ii]=True;adj_cv[:,:,:,ii,ii]=True
    for k in range(n):
        adj |= adj[:,:,:, :,k,None]&adj[:,:,:,None,k,:]
        adj_cv |= adj_cv[:,:,:, :,k,None]&adj_cv[:,:,:,None,k,:]
    healthy=adj[:,:,:,:ns,:ns].all((-1,-2));healthy_cv=adj_cv[:,:,:,:ns,:ns].all((-1,-2))
    globals=np.zeros((b,GLOBAL),np.float32)
    globals[:,:9]=1-healthy.mean(1);globals[:,9:18]=1-healthy_cv.mean(1)
    globals[:,18]=known.mean(1);globals[:,19]=ok.mean((1,2));globals[:,20]=n/16;globals[:,21]=ns/10
    globals[:,22]=pair.mean((1,2));globals[:,23]=np.where(ok,ages,10).mean((1,2))/10
    return out,globals


def build(local,trace,queries,city):
    queries=np.asarray(queries,int);count=len(queries)
    result={name:np.empty((count,4,MAX_N,NODE if name.startswith('node') else MAX_N,)+( () if name.startswith('node') else (EDGE,)),np.float16)
            for name in ('node_local','node_shared','edge_local','edge_shared')}
    states=[]
    for q,(end,observer) in enumerate(queries):
        if end+HISTORY[0]<1:raise ValueError('Not enough causal graph history')
        for k,offset in enumerate(HISTORY):
            for shared in (False,True):
                node,edge,state=frame(local,trace,int(end+offset),int(observer),shared)
                suffix='shared' if shared else 'local'
                result['node_'+suffix][q,k]=node;result['edge_'+suffix][q,k]=edge
                if k==3 and shared:states.append(state)
    ph=[];gl=[]
    for start in range(0,len(states),32):
        a,b=physics(city,states[start:start+32]);ph.append(a);gl.append(b)
    result['physics']=np.concatenate(ph).astype(np.float16);result['global']=np.concatenate(gl).astype(np.float16)
    return result
