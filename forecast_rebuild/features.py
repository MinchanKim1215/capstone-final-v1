"""Features accept observed histories only; no truth/seed/map identifier API."""
import numpy as np
from scipy.special import ndtr

SCALE=np.asarray([5,5,50,1,1,16,16,1,100,30,20,64,200,200,16,20,400,400,10,10,400,10,150,150,1,1,50,400,
 100,30,20,30,20,16,400,400,10,10,400,10,30,30,10,10,16,32,1,10],np.float32)
INTENT_SCALE=np.asarray([400,400,400,400,30,3,400,400,1],np.float32)
GEO_SCALE=np.asarray([30]*13+[1]*13+[30]*19+[1],np.float32)
HORIZONS=np.asarray([.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8],np.float32)
VERSION='causal_local_history_geometry_energy_v1'
RAW_NAMES=('own_vx','own_vy','altitude','own_relay','peer_relay','attempts','acks','ack_rate','ack_power','ack_sinr',
 'ack_age','queue','own_x','own_y','received_data','data_age','data_dx','data_dy','data_dvx','data_dvy','data_distance',
 'data_radial_speed','loss_now','loss_5s','los_now','los_5s','loss_change_5s','nearest_tall_building','beacon_power',
 'beacon_sinr','beacon_age','reverse_sinr','reverse_age','beacon_count','beacon_dx','beacon_dy','beacon_dvx','beacon_dvy',
 'beacon_distance','beacon_radial_speed','minimum_sinr','sinr_margin','sinr_slope','reverse_slope','beacons_5s',
 'beacons_10s','fresh_bidirectional','acceleration')


def feature_names():
    names=[group+'_'+field for group in ('last','lag1','lag3','mean3','std3','mean10','std10','coverage3','coverage10','slope3') for field in RAW_NAMES]
    names += ['known_'+field for field in RAW_NAMES]+['intent_'+str(i) for i in range(9)]
    names += ['geometry_'+str(i) for i in range(46)]+['physics_outage_'+str(i) for i in range(13)]
    names += ['local_'+group+'_'+str(i) for group in ('mean','std','min','max') for i in range(6)]
    names += ['energy_'+group+'_'+str(i) for group in ('last','lag3','mean3','std3') for i in range(6)]
    return names+['relay_budget','fleet_budget']


def summary(a,known,axis):
    count=known.sum(axis)
    total=np.where(known,a,0).sum(axis)
    mean=np.divide(total,np.maximum(count,1))
    square=np.where(known,a*a,0).sum(axis)/np.maximum(count,1)
    std=np.sqrt(np.maximum(0,square-mean*mean))
    return np.where(count>0,mean,np.nan),np.where(count>0,std,np.nan),count/a.shape[axis]


def windows(values,masks,intents,ends,rows):
    rows=np.asarray(rows,int)
    if rows.ndim!=2 or rows.shape[1]!=3:raise ValueError('Rows are time-index, local observer, peer')
    t,r,j=rows.T
    end=np.asarray(ends)[t]
    ix=end[:,None]+np.arange(-40,0)[None,:]
    if np.any(ix<0) or np.any(ix>=end[:,None]):raise ValueError('Future or unobserved history access')
    return tuple(np.asarray(a[ix,r[:,None],j[:,None]]) for a in (values,masks,intents))


def energy_windows(energy,ends,rows):
    t,r,j=np.asarray(rows,int).T;end=np.asarray(ends)[t]
    ix=end[:,None]+np.arange(-40,0)[None,:]
    return np.asarray(energy[ix,r[:,None],j[:,None]],np.float32)


def normalize_energy(e):
    out=np.asarray(e,np.float32).copy()
    out[...,:2]=(out[...,:2]+100)/20
    out[...,2]/=10
    out[...,5]/=10
    return out


def build(values,masks,intents,ends,geometry,rows,energy=None,use_energy=True):
    rows=np.asarray(rows,int);t,r,j=rows.T
    v,m,c=windows(values,masks,intents,ends,rows)
    v=v.astype(np.float32)/SCALE
    m=m.astype(bool)
    known=np.where(m,v,np.nan)
    mean3,std3,coverage3=summary(v[:,-6:],m[:,-6:],1)
    mean10,std10,coverage10=summary(v[:,-20:],m[:,-20:],1)
    last=known[:,-1]
    parts=[last,known[:,-3],known[:,-7],mean3,std3,mean10,std10,coverage3,coverage10,
           (last-known[:,-7])/3,m[:,-1].astype(np.float32),c[:,-1].astype(np.float32)/INTENT_SCALE]
    g=np.asarray(geometry[t,r,j],np.float32)
    geo=np.where(g[:,-1:]>.5,g/GEO_SCALE,np.nan)
    geo[:,-1]=g[:,-1]
    parts.append(geo)
    raw_margin=np.asarray(values[np.asarray(ends)[t]-1,r,j,41],np.float32)
    margin_known=np.asarray(masks[np.asarray(ends)[t]-1,r,j,41],bool)&(g[:,-1]>.5)
    # A conditional Gaussian physics reference, not hidden future RF truth.
    sigma=np.sqrt(1.5**2*(1-np.exp(-2*HORIZONS/4))+.5**2)
    reference=ndtr((g[:,:13]-raw_margin[:,None])/sigma[None])
    parts.append(np.where(margin_known[:,None],reference,np.nan))
    # Other peers of THIS observer only. Never aggregate another drone's private
    # observations; those would require an actually received message.
    frame=np.asarray(ends)[t]-1
    fields=[29,31,30,41,46,38]
    local=np.asarray(values[frame,r],np.float32)[...,fields]/SCALE[fields]
    local_known=np.asarray(masks[frame,r],bool)[...,fields]
    mean,std,cov=summary(local,local_known,1)
    low=np.where(local_known,local,np.inf).min(1)
    high=np.where(local_known,local,-np.inf).max(1)
    low=np.where(cov>0,low,np.nan);high=np.where(cov>0,high,np.nan)
    parts.extend([mean,std,low,high])
    if energy is not None and use_energy:
        ew=normalize_energy(energy_windows(energy,ends,rows))
    else:
        ew=np.full((len(rows),40,6),np.nan,np.float32)
        ew[...,3:5]=0
    em,es,_=summary(np.nan_to_num(ew[:,-6:]),np.isfinite(ew[:,-6:]),1)
    parts.extend([ew[:,-1],ew[:,-7],em,es])
    parts.append(np.tile(np.array([values.shape[1]/5,values.shape[2]/10],np.float32),(len(rows),1)))
    result=np.concatenate(parts,-1).astype(np.float32)
    result[~np.isfinite(result)]=np.nan
    return result


def sequence(values,masks,intents,ends,rows,energy=None,use_energy=True):
    v,m,c=windows(values,masks,intents,ends,rows)
    v=np.where(m,v,0).astype(np.float32)/SCALE
    out=[np.clip(v,-8,8),m.astype(np.float32),np.clip(c/INTENT_SCALE,-8,8)]
    if energy is not None and use_energy:
        e=normalize_energy(energy_windows(energy,ends,rows))
        out.append(np.nan_to_num(e,nan=0.,posinf=0.,neginf=0.))
    else:out.append(np.zeros((len(rows),40,6),np.float32))
    return np.concatenate(out,-1).astype(np.float32)
