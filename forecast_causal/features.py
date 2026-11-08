"""186measurement summaries; only receiver-visible queue/route reports."""
import numpy as np
FIELDS=np.r_[np.arange(14),26,27]
FIELD_SCALE=np.array([1,1,64,64,64,100,10,100]+[64]*6+[100,2],np.float32)
DIM=186
SCALE=np.r_[np.tile(np.r_[np.tile(FIELD_SCALE,3),1],3),FIELD_SCALE,np.ones(12),np.ones(8),np.ones(3)].astype(np.float32)
if len(SCALE)!=DIM:raise AssertionError('Network scale dimension')

def route_summary(v,known,n,ns,observer):
    fresh=(v[:,20:20+ns]>.5)&known[:,None]&(v[:,27,None]<=1.)
    complete=unknown=loops=observer_paths=0;hops=[];total=ns*(ns-1)
    for source in range(ns):
        for destination in range(ns):
            if source==destination:continue
            node=source;seen=set();path=[]
            for step in range(n+1):
                if node==destination:
                    complete+=1;hops.append(step);observer_paths+=int(observer in path);break
                if node in seen:loops+=1;break
                if not 0<=node<n or not fresh[node,destination]:unknown+=1;break
                seen.add(node);path.append(node);node=int(v[node,14+destination])
            else:loops+=1
    queues=np.where(known[:,None],np.nan_to_num(v[:,8:8+ns]),0);weighted=float((queues*fresh).sum()/queues.sum()) if queues.sum() else np.nan
    return [float(fresh.mean()),complete/total,unknown/total,loops/total,float(np.mean(hops))/10 if hops else np.nan,max(hops)/10 if hops else np.nan,observer_paths/max(1,complete),weighted]

def one(v,known,n,ns):
    ids=np.arange(n);parts=[];values=v[:,FIELDS]
    for group in (np.ones(n,bool),ids<ns,(ids>=ns)&(ids!=ns)):
        take=known&group;count=int(take.sum())
        if count:
            x=values[take];parts.extend([x.mean(0),x.max(0),x.std(0)])
        else:parts.extend([np.full(16,np.nan)]*3)
        parts.append(np.array([count/max(int(group.sum()),1)]))
    own=v[ns];route=own[20:26]>.5;hoprole=np.where(route,np.where(own[14:20]<ns,0,1),-1)
    parts.extend([own[FIELDS],route.astype(float),hoprole,route_summary(v,known,n,ns,ns),[n/10,ns/6,(n-ns)/7]])
    out=np.concatenate(parts).astype(np.float32)
    if out.shape!=(DIM,):raise ValueError('Network summary dimension changed')
    return out

def summarize(network,known,n,ns):return np.stack([one(v,m,n,ns) for v,m in zip(network,known)])

def tree_features(summary,ends):
    x=np.asarray(summary,np.float32);end=np.asarray(ends,int);good=np.isfinite(x);z=np.nan_to_num(x,nan=0.)
    sums=np.vstack([np.zeros(DIM),np.cumsum(z,axis=0,dtype=np.float64)]);squares=np.vstack([np.zeros(DIM),np.cumsum(z*z,axis=0,dtype=np.float64)]);counts=np.vstack([np.zeros(DIM),np.cumsum(good,axis=0)])
    def rolling(width):
        c=counts[end]-counts[end-width];mean=(sums[end]-sums[end-width])/np.maximum(c,1);sd=np.sqrt(np.maximum(0,(squares[end]-squares[end-width])/np.maximum(c,1)-mean*mean));return np.where(c>0,mean,np.nan),np.where(c>0,sd,np.nan)
    mean3,std3=rolling(6);mean10,_=rolling(20);last=x[end-1];lag3=x[end-7]
    return np.concatenate([last,lag3,mean3,std3,mean10,(last-lag3)/3],-1).astype(np.float32)/np.tile(SCALE,6)

def sequence(summary,ends):
    ix=np.asarray(ends,int)[:,None]+np.arange(-39,0,2)[None]
    if ix.min()<0 or ix.max()>=len(summary):raise ValueError('Network history causality boundary')
    return np.clip(np.nan_to_num(np.asarray(summary[ix],np.float32)/SCALE,nan=0.,posinf=0.,neginf=0.),-8,8)
