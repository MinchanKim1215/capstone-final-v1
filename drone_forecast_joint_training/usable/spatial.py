"""성공 수신 비콘으로 추정한 위치·속도와 정적 지도에서 생성한 선행 공간 입력.

현재/3초/5초 뒤의 등속 외삽 전파 손실 추정입니다. 실제 미래 궤적,
미래 채널, 간섭원의 위치/상태를 사용하지 않습니다. 외삽 비행선이
비행 금지 영역과 교차하는지도 함께 표시해 외삽의 한계를 입력에 명시합니다.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union
from swarm_sim.radio import BuildingRadio
from swarm_sim.config import Scenario

NAMES=['beacon_map_loss_now_db','beacon_map_loss_3s_db','beacon_map_loss_change_3s_db',
       'beacon_map_loss_5s_db','beacon_map_loss_change_5s_db','beacon_map_los_now',
       'beacon_map_los_3s','beacon_map_los_5s','self_extrapolation_blocked_3s',
       'peer_extrapolation_blocked_3s','self_extrapolation_blocked_5s','peer_extrapolation_blocked_5s']


def context(path):
    scene=json.loads(Path(path).read_text());g=[shape(b['geometry_local_m']) for b in scene['buildings']]
    h=np.array([b['height_m'] for b in scene['buildings']],float)
    return SimpleNamespace(radio=BuildingRadio(g,h),flight_blocker=unary_union([shape(z['geometry_local_m']) for z in scene['no_fly_zones']]))


def features(ctx,x,mask):
    original_shape=x.shape[:-1];out=np.zeros((*original_shape,12),np.float32);m=np.zeros_like(out,bool)
    good=mask[...,34]&mask[...,36]&mask[...,30]&(x[...,30]<=2.)
    if not good.any():return out,m
    xx=x[good];a=np.c_[xx[:,12:14],xx[:,2]];b=a+np.c_[xx[:,34:36],np.zeros(len(xx))]
    va=np.c_[xx[:,:2],np.zeros(len(xx))];vb=va+np.c_[xx[:,36:38],np.zeros(len(xx))]
    # 동일 기하 입력의 중복만 제거합니다. 측정 SINR나 정답을 이용한 선택 없음.
    packed=np.c_[a,b,va,vb];unique,inverse=np.unique(packed,axis=0,return_inverse=True)
    result=np.empty((len(unique),12),np.float32);scenario=Scenario()
    # 같은 지역/방향의 간섭 조건은 기하 입력을 많이 공유합니다. 동일 입력만 캐시합니다.
    if not hasattr(ctx,'_spatial_cache'):ctx._spatial_cache={}
    keys=[u.tobytes() for u in unique];missing=[]
    for q,key in enumerate(keys):
        if key in ctx._spatial_cache:result[q]=ctx._spatial_cache[key]
        else:missing.append(q)
    for first in range(0,len(missing),4096):
        take=missing[first:first+4096];u=unique[take];aa,bb,v1,v2=u[:,:3],u[:,3:6],u[:,6:9],u[:,9:12]
        now,los,_=ctx.radio.propagation(aa,bb,scenario);r=np.zeros((len(u),12),np.float32)
        r[:,0]=now;r[:,5]=los
        for horizon,lc,dc,lsc,bc in [(3,1,2,6,8),(5,3,4,7,10)]:
            af=aa+horizon*v1;bf=bb+horizon*v2;af[:,:2]=np.clip(af[:,:2],-195,195);bf[:,:2]=np.clip(bf[:,:2],-195,195)
            loss,lf,_=ctx.radio.propagation(af,bf,scenario);r[:,lc]=loss;r[:,dc]=loss-now;r[:,lsc]=lf
            r[:,bc]=shapely.intersects(shapely.linestrings(np.stack((aa[:,:2],af[:,:2]),axis=1)),ctx.flight_blocker)
            r[:,bc+1]=shapely.intersects(shapely.linestrings(np.stack((bb[:,:2],bf[:,:2]),axis=1)),ctx.flight_blocker)
        result[take]=r
        for q,row in zip(take,r):ctx._spatial_cache[keys[q]]=row.copy()
    out[good]=result[inverse];m[good]=True
    return out,m
