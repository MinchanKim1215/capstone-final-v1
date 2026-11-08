"""Learn prospective-path error from TRAIN flights only; no test future input."""
import numpy as np
from . import ARTIFACTS,PARENT,NAV,WORKSPACE,ROOT
from .contracts import read,write,sha,now,SEED
from forecast_graph.features import ReceivedTrace
from forecast_navigation.wire import extend_trace

def main():
    target=ARTIFACTS/'path_uncertainty.json'
    if target.exists():return read(target)
    source=[p for p in read(PARENT/'data_manifest.json')['records'] if p['role']=='train'];sums=np.zeros((2,9,2,2));counts=np.zeros((2,9,2),int);sources=[]
    for number,parent in enumerate(source):
        r=parent['source_record'];ns=r['n_search'];key=r['key']
        with np.load(parent['path']) as a:ends=a['ends'][a['uniform']]
        rng=np.random.default_rng(int(key[:8],16)^SEED);ends=ends[np.sort(rng.choice(len(ends),min(64,len(ends)),replace=False))]
        with np.load(WORKSPACE/'forecast_graph/artifacts/traces'/(key+'.npz')) as a:tx=a['tx'];rx=a['rx'];n=int(a['n'])
        with np.load(NAV/'deltas'/(key+'.npz')) as a:tx=extend_trace(tx,a['delta'])
        trace=ReceivedTrace(tx,rx,n,ns)
        with np.load(r['array_path']) as a:positions=a['positions']
        for end in ends:
            s=trace.snapshot(int(end),ns);times=end*.5+np.arange(9);k=np.floor(times/3).astype(int);good_h=k+1<len(positions)
            if not good_h.all():continue
            f=(times-k*3)/3;actual=positions[k]*(1-f[:,None,None])+positions[k+1]*f[:,None,None];actual=actual.transpose(1,0,2)
            displacement=s['future']-s['future'][:,:1];norm=np.linalg.norm(displacement,axis=-1)
            fallback=s['velocity']/np.maximum(np.linalg.norm(s['velocity'],axis=-1,keepdims=True),1e-8);fallback[np.linalg.norm(fallback,axis=-1)<.5]=[1,0]
            tangent=np.where((norm>1e-6)[...,None],displacement/np.maximum(norm[...,None],1e-8),fallback[:,None]);normal=np.stack([-tangent[...,1],tangent[...,0]],-1);error=actual-s['future']
            component=np.stack([(error*tangent).sum(-1),(error*normal).sum(-1)],-1)
            for node in np.flatnonzero(s['known']):
                role=int(node>=ns)
                for h in range(9):
                    committed=int(h<=s['commit'][node]+1e-9);sums[role,h,committed]+=component[node,h]**2;counts[role,h,committed]+=1
        sources.append(dict(key=key,array_sha256=r['array_sha256'],queries=len(ends)))
        if number%40==0:print('TRAIN path uncertainty',number+1,'/',len(source),flush=True)
    rms=np.sqrt(sums/np.maximum(counts[...,None],1))
    for role in range(2):
        for h in range(9):
            for c in range(2):
                if counts[role,h,c]<20:rms[role,h,c]=rms[role,h,1-c]
    rms=np.clip(rms,.1,30.)
    value=dict(created_at=now(),training_only=True,train_flights=245,sources=sources,source_pins=dict(path_uncertainty=sha(ROOT/'path_uncertainty.py')),
        axes=['search_or_relay','horizon0to8','uncommitted_or_committed','along_or_normal'],rms_error_m=rms.tolist(),counts=counts.tolist(),
        use='Scale deterministic geometry-feature scenarios; not known future positions or a calibrated probability of trajectories.',selection_or_confirmation_fit=False)
    write(target,value);print('TRAIN-only path-error scales ready',flush=True);return value

if __name__=='__main__':main()
