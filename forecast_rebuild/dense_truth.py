"""Privileged continuous RF TARGETS only; never an input feature provider."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import json
import multiprocessing as mp
import numpy as np
from scipy.special import erfc
from . import ARTIFACTS
from .contracts import read,write,sha,now
from .dataset import Episode

HORIZONS=np.arange(1,9)*2


def packet_success(sinr,bits):
    ber=.5*erfc(np.sqrt(10**(np.clip(sinr,-100,100)/10)))
    return np.exp(bits*np.log1p(-np.minimum(ber,.5)))


def one(record,rows):
    from adaptive_joint_expanded.world import World
    ep=Episode(record);t,r,j=np.asarray(rows).T
    future=np.asarray(ep.ends)[t,None]+HORIZONS[None,:]
    i=np.broadcast_to((r+record['n_search'])[:,None],future.shape)
    peer=np.broadcast_to(j[:,None],future.shape)
    requests=np.c_[future.ravel(),i.ravel(),peer.ravel()]
    unique,inverse=np.unique(requests,axis=0,return_inverse=True);q,i,j=unique.T
    with np.load(record['array_path'],allow_pickle=False) as archive:positions=archive['positions']
    env=World(**record['spec'],deployment=record['deployment'],observer=None,max_steps=1)
    step=q//6;fraction=(q%6)/6
    axy=positions[step,i]*(1-fraction[:,None])+positions[step+1,i]*fraction[:,None]
    bxy=positions[step,j]*(1-fraction[:,None])+positions[step+1,j]*fraction[:,None]
    a=np.c_[axy,np.full(len(q),50.)];b=np.c_[bxy,np.full(len(q),50.)]
    result=np.empty(len(q),np.float32)
    for start in range(0,len(q),4096):
        take=slice(start,start+4096);aa=a[take];bb=b[take];tt=q[take];ii=i[take];jj=j[take]
        loss=env.city.radio.propagation(aa,bb,env.scenario)[0]
        reverse=env.city.radio.propagation(bb,aa,env.scenario)[0]
        ja=env.city.radio.propagation(env.jammer[None],aa,env.scenario)[0]
        jb=env.city.radio.propagation(env.jammer[None],bb,env.scenario)[0]
        active=(tt*.5>=env.jam_start)&(tt*.5<env.jam_end)
        dena=10*np.log10(10**(-107/10)+10**((env.jam_power-ja)/10)*active)
        denb=10*np.log10(10**(-107/10)+10**((env.jam_power-jb)/10)*active)
        forward=10-loss-env.shadow[tt,ii,jj]-denb
        backward=10-reverse-env.shadow[tt,jj,ii]-dena
        p=packet_success(forward,4608)*packet_success(backward,112)
        pr=packet_success(backward,4608)*packet_success(forward,112)
        actual=np.asarray(ep.links[tt,ii,jj])
        if not np.array_equal((p>=.9)&(pr>=.9),actual):
            raise AssertionError('Continuous future RF targets do not reproduce stored binary truth')
        result[take]=np.minimum(forward,backward)
    return result[inverse].reshape(len(rows),8),len(unique)


def main():
    folder=ARTIFACTS/'samples/expanded_planned';manifest=read(folder/'manifest.json')
    output=folder/'train_margin.npy';pointer=folder/'margin_manifest.json'
    if pointer.exists():
        saved=read(pointer)
        if saved['samples_sha256']!=sha(folder/'manifest.json') or saved['sha256']!=sha(output):raise ValueError('Dense target cache changed')
        print(json.dumps(saved));return
    index=np.load(folder/'train_index.npy',mmap_mode='r',allow_pickle=False);records=manifest['contract']['records']
    target=np.lib.format.open_memmap(output,mode='w+',dtype=np.float32,shape=(len(index),8));target[:]=np.nan
    tasks=[]
    for e in np.unique(index[:,0]):
        record=records[int(e)]
        if record['source_kind']=='bootstrap':continue
        positions=np.flatnonzero(index[:,0]==e);tasks.append((record,positions,np.asarray(index[positions,1:])))
    count=0;checked=0
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(one,record,rows):positions for record,positions,rows in tasks}
        for future in as_completed(pending):
            values,n=future.result();target[pending[future]]=values;checked+=n;count+=1
            if count%8==0 or count==len(tasks):
                target.flush();state=dict(state='preparing',complete=count,total=len(tasks),verified_rf_pairs=checked,updated_at=now())
                write(ARTIFACTS/'dense_truth_status.json',state);print(json.dumps(state),flush=True)
    target.flush()
    value=dict(samples_sha256=sha(folder/'manifest.json'),sha256=sha(output),verified_rf_pairs=checked,
               rows_with_truth=int(np.isfinite(target).all(-1).sum()),future_horizons_s=list(range(1,9)),
               target_only=True,feature_api_changed=False,created_at=now())
    write(pointer,value);write(ARTIFACTS/'dense_truth_status.json',dict(state='completed',**value));print(json.dumps(value))


if __name__=='__main__':main()
