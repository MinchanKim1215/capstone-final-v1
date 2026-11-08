"""54features: received path, map losses and TRAIN-scaled geometry scenarios."""
import numpy as np
import shapely
from shapely.strtree import STRtree
from . import ARTIFACTS
from .contracts import read
from forecast_graph.features import frame,THRESHOLD
HORIZONS=(1,3,5,7)
DIM=54
RNG=np.random.default_rng(1601)
COMMON=RNG.normal(size=(8,1,2));INDIVIDUAL=RNG.normal(size=(8,10,2))
COMMON-=COMMON.mean(0);INDIVIDUAL-=INDIVIDUAL.mean(0)

def connectivity(adj,known,ns):
    a=np.asarray(adj,bool).copy();n=a.shape[-1];a[...,np.arange(n),np.arange(n)]=True
    for k in range(n):a |= a[..., :,k,None]&a[...,None,k,:]
    return np.where(np.asarray(known)[...,:ns].all(-1),~a[...,:ns,:ns].all((-1,-2)),np.nan).astype(float)

class Geometry:
    def __init__(self,city):
        self.city=city;self.rms=np.asarray(read(ARTIFACTS/'path_uncertainty.json')['rms_error_m']);self.tree=STRtree(city.radio.geometries[city.radio.heights>=50.])
    def blocked(self,position):
        shape=position.shape[:-2];n=position.shape[-2];a=np.broadcast_to(position[...,:,None,:],shape+(n,n,2));b=np.broadcast_to(position[...,None,:,:],a.shape)
        lines=shapely.linestrings(np.stack([a,b],-2).reshape(-1,2,2));out=np.zeros(len(lines),bool)
        if len(self.tree.geometries):
            hits=self.tree.query(lines,predicate='intersects')
            if hits.shape[1]:out[hits[0]]=True
        out=out.reshape(shape+(n,n));out[...,np.arange(n),np.arange(n)]=False;return out
    def losses(self,position):
        n=len(position);a=np.repeat(position,n,axis=0);b=np.tile(position,(n,1));xyz_a=np.c_[a,np.full(n*n,50.)];xyz_b=np.c_[b,np.full(n*n,50.)]
        return self.city.radio.propagation(xyz_a,xyz_b,self.city._forecast_scenario)[0].reshape(n,n)
    def one(self,local,trace,end):
        n=local.n;ns=local.ns;_,_,s=frame(local,trace,int(end),0,True);known=s['known'];pair=known[:,None]&known[None,:];np.fill_diagonal(pair,False)
        current=s['future'][:,0];loss0=self.losses(current);observed=np.isfinite(s['sinr'])&(s['report_age']<=10);commit=trace.snapshot(int(end),ns)['commit']
        initial=np.where(observed,s['sinr']-THRESHOLD,117-loss0-THRESHOLD);result=[];role=(np.arange(n)>=ns).astype(int)
        for h in HORIZONS:
            nominal=s['future'][:,h];loss=self.losses(nominal);margin=initial+loss0-loss;mutual=np.minimum(margin,margin.T);blocked=self.blocked(nominal)
            nominal_radio=connectivity((mutual>=0)&pair,known,ns);nominal_los=connectivity(~blocked&pair,known,ns)
            displacement=nominal-current;norm=np.linalg.norm(displacement,axis=-1);tangent=np.divide(displacement,np.maximum(norm[:,None],1e-8));fallback=s['velocity']/np.maximum(np.linalg.norm(s['velocity'],axis=-1,keepdims=True),1e-8);fallback[np.linalg.norm(fallback,axis=-1)<.5]=[1,0];tangent[norm<1e-6]=fallback[norm<1e-6];normal=np.stack([-tangent[:,1],tangent[:,0]],-1)
            committed=(h<=commit+1e-9).astype(int);sigma=self.rms[role,h,committed];z=(COMMON+INDIVIDUAL[:,:n])/np.sqrt(2.)
            sample=nominal[None]+z[...,0,None]*sigma[None,:,0,None]*tangent[None]+z[...,1,None]*sigma[None,:,1,None]*normal[None]
            sample=np.clip(sample,-195,195);unsafe=shapely.intersects_xy(self.city.flight_blocker,sample[...,0],sample[...,1])
            if unsafe.any():sample[unsafe]=self.city.points[self.city.tree.query(sample[unsafe])[1]]
            # Point scenarios obey obstacle exclusion. They are geometry features,
            # not executable full trajectories or hidden future controller actions.
            mc=self.blocked(sample);fraction=mc.mean(0);mc_disconnected=connectivity(~mc&pair[None],np.broadcast_to(known,(8,n)),ns)
            clear=pair&~blocked;values=mutual[pair];positive=float((values>=0).mean()) if values.size else np.nan
            result.extend([float(nominal_radio),float(nominal_los),float(np.quantile(values,.1))/30 if values.size else np.nan,float(values.mean())/30 if values.size else np.nan,positive,
                float((~blocked[pair]).mean()) if pair.any() else np.nan,float(np.mean(mc_disconnected)),float(fraction[clear].max()) if clear.any() else np.nan,
                float(fraction[clear].mean()) if clear.any() else np.nan,float(fraction[pair].mean()) if pair.any() else np.nan,float(sigma[known].mean())/10 if known.any() else np.nan,float(committed[known].mean()) if known.any() else np.nan])
        result.extend([known.mean(),known[:ns].mean(),observed[pair].mean() if pair.any() else 0,float(np.minimum(s['report_age'][observed].max(),10))/10 if observed.any() else 1,ns/6,n/10])
        out=np.asarray(result,np.float32)
        if out.shape!=(DIM,):raise ValueError('Physicalfeature dimension changed')
        return out
    def build(self,local,trace,ends):return np.stack([self.one(local,trace,int(e)) for e in ends])
