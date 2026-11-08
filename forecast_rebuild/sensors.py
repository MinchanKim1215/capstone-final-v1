"""Offline measurement simulator, deliberately separated from model feature APIs."""
import numpy as np

VERSION='idle_energy_causal_beacon_sample_time_v2'


def received_energy(own,own_valid,beacon_counts,n_search):
    """Peer measurements arrive only via received beacons, sent before this frame.

    own[q] is measured during RF interval q and available at its end. A beacon
    sent during q can contain own[q-1], never own[q]. Returned age is relative
    to the prediction frame's end. No true peer interference bypass is allowed.
    """
    own=np.asarray(own,np.float32);own_valid=np.asarray(own_valid,bool)
    count=np.asarray(beacon_counts)
    frames,n=own.shape;nr=n-n_search
    if count.shape!=(frames,nr,n):raise ValueError('Local beacon reception shape differs')
    out=np.full((frames,nr,n,6),np.nan,np.float32)
    cache=np.full((nr,n),np.nan,np.float32)
    stamp=np.full((nr,n),-1,int)
    for q in range(frames):
        if q>0:
            received=(count[q]>0)&own_valid[q-1][None,:]
            cache[received]=np.broadcast_to(own[q-1],(nr,n))[received]
            stamp[received]=q-1
        for r in range(nr):
            observer=r+n_search
            out[q,r,:,3]=float(own_valid[q,observer])
            out[q,r,:,4]=((stamp[r]>=0)&((q-stamp[r]+1)*.5<=10)).astype(np.float32)
            if own_valid[q,observer]:
                out[q,r,:,0]=own[q,observer]
                # RF values represent the start of an interval and become
                # available at its end. Preserve physical measurement age.
                out[q,r,:,5]=.5
            fresh=(stamp[r]>=0)&((q-stamp[r]+1)*.5<=10)
            out[q,r,fresh,1]=cache[r,fresh]
            out[q,r,fresh,2]=(q-stamp[r,fresh]+1)*.5
            out[q,r,observer]=np.array([np.nan,np.nan,np.nan,0,0,np.nan])
    return out


def simulate(record,values,links,positions):
    """Same RF world as the stored trajectory; only noisy current measurements.

    Hidden source parameters below belong to the measurement generator, not
    to any feature or inference API. Baseline RF labels remain unchanged.
    """
    if record['source_kind']=='bootstrap':
        result=np.full((*values.shape[:3],6),np.nan,np.float32);result[...,3:5]=0
        return result,dict(enabled=False,reason='Historical archive has no equivalent validated sensor replay')
    from adaptive_joint_expanded.world import World
    from cooperative.packets import hop_probabilities
    env=World(**record['spec'],deployment=record['deployment'],observer=None,max_steps=1)
    frames,n=links.shape[:2]
    ticks=np.arange(frames);step=ticks//6;fraction=(ticks%6)/6
    positions=np.asarray(positions,np.float64)
    if positions.shape[0]<=int(step.max())+1:raise ValueError('Recorded movement does not cover RF frames')
    xy=positions[step]*(1-fraction[:,None,None])+positions[step+1]*fraction[:,None,None]
    xyz=np.concatenate([xy,np.full((*xy.shape[:-1],1),50.)],-1)
    loss=env.city.radio.propagation(env.jammer[None],xyz.reshape(-1,3),env.scenario)[0].reshape(frames,n)
    active=(ticks*.5>=env.jam_start)&(ticks*.5<env.jam_end)
    interference=10**((env.jam_power-loss)/10)*active[:,None]
    # Reconstruct a fixed set of RF frames before enabling this replay. This
    # catches a different jammer schedule, coordinate convention or RF engine.
    checked=np.unique(np.linspace(0,frames-1,min(24,frames)).astype(int))
    for q in checked:
        a=xyz[q];tx=np.repeat(a,n,axis=0);rx=np.tile(a,(n,1))
        pair_loss=env.city.radio.propagation(tx,rx,env.scenario)[0].reshape(n,n)
        signal=np.empty((n,n));sinr=np.empty_like(signal)
        probability=hop_probabilities(np.ascontiguousarray(pair_loss),np.ascontiguousarray(interference[q]),
                                      env.shadow[q],signal,sinr)
        if not np.array_equal((probability>=.9)&(probability.T>=.9),links[q]):
            raise ValueError('Measurement replay differs from the original RF truth')
    noise=10**(-107/10)
    energy=10*np.log10(noise+interference)
    rng=np.random.default_rng((int(record['spec']['seed'])+20261102)%2**32)
    measured=np.round((energy+rng.normal(0,.75,energy.shape))*10)/10
    valid=rng.random(energy.shape)>=.01
    counts=values[...,33]
    output=received_energy(measured,valid,counts,record['n_search'])
    return output,dict(enabled=True,measurement='Simulated idle-slot I+N energy; not hidden source coordinates',
                       sigma_db=.75,quantization_db=.1,drop_probability=.01,rf_frames_reconstruction_checked=len(checked),
                       hardware_support_confirmed=False,version=VERSION)
