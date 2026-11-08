"""유료 주기 관측: 실제 수신한 비콘/이웃이 보고한 역방향 측정만 입력.

1Hz/드론, 128B BPSK 브로드캐스트, ACK 없음. 5ms 슬롯 × 8/초=4% 예약.
프레임에 들어 있는 위치/속도는 해당 0.5초 구간 시작의 값입니다.
잡음 표준편차 0.5dB는 계측 오차에 대한 프로젝트 가정입니다.
실패한 수신의 SINR, 간섭원의 위치/활성 상태는 입력으로 공개하지 않습니다.
"""
import math
import numpy as np
from numba import njit
from swarm_sim.fast_network import bpsk_probability

EXTRA_NAMES=[
 'beacon_received_power_dbm','beacon_received_sinr_db','beacon_age_s',
 'peer_reported_reverse_sinr_db','reverse_measurement_age_s','beacons_received_this_interval',
 'beacon_relative_x_m','beacon_relative_y_m','beacon_relative_vx_mps','beacon_relative_vy_mps',
 'beacon_estimated_distance_m','beacon_estimated_distance_rate_mps','observed_minimum_sinr_db',
 'observed_bidirectional_margin_db','inbound_sinr_slope_3s','reverse_sinr_slope_3s',
 'beacons_received_last_5s','beacons_received_last_10s','both_measurements_fresh_2s',
 'self_acceleration_mps2',
]


def data_threshold_db():
    """대칭 SINR의 데이터 576B+ACK14B 홉 성공확률 0.9에 해당하는 값."""
    low,high=-20.,30.
    for _ in range(80):
        mid=(low+high)/2
        if bpsk_probability(mid,4608)*bpsk_probability(mid,112)>=.9:high=mid
        else:low=mid
    return (low+high)/2


@njit(cache=True)
def observed_beacons(received,sinr,xyz_start,xyz_end,ns,seed,threshold_db,noise_sigma_db=.5):
    np.random.seed(seed)
    nt,n,_=sinr.shape;f=20;dt=.5
    z=np.zeros((nt,n,n,f),np.float32);m=np.zeros((nt,n,n,f),np.bool_)
    last=np.full((n,n),-1.);rxdb=np.zeros((n,n));rxsin=np.zeros((n,n))
    revstamp=np.full((n,n),-1.);revsin=np.zeros((n,n))
    poses=np.zeros((n,n,3));vels=np.zeros((n,n,3));pose_stamp=np.full((n,n),-1.)
    counts=np.zeros((nt,n,n),np.int16);tx_count=0;rx_count=0
    for t in range(nt):
        velocity=(xyz_end[t]-xyz_start[t])/dt
        # 0,1,... 秒마다 8개 슬롯의 서로 다른 송신자. 같은 슬롯의 전역 동시 송신 없음.
        if t%2==0:
            for slot in range(n):
                j=slot;now=t*dt+slot*.005
                # 송신 순간의 과거 캐시만 메시지에 넣습니다. 나중에 받은 내용을 소급하지 않습니다.
                old_last=last[j].copy();old_sin=rxsin[j].copy()
                tx_count+=1
                for i in range(n):
                    if i==j:continue
                    probability=bpsk_probability(sinr[t,j,i],1024)
                    if np.random.random()<probability:
                        rx_count+=1;counts[t,i,j]+=1
                        last[i,j]=now+.001024
                        # 표시 해상도 0.01dB. 측정 잡음과 실제 채널 확률의 명시적 분리.
                        rxsin[i,j]=round((sinr[t,j,i]+np.random.normal(0,noise_sigma_db))*100)/100
                        rxdb[i,j]=round((received[t,j,i]+np.random.normal(0,noise_sigma_db))*100)/100
                        pose_stamp[i,j]=t*dt;poses[i,j]=xyz_start[t,j];vels[i,j]=velocity[j]
                        if old_last[i]>=0:
                            revstamp[i,j]=old_last[i];revsin[i,j]=old_sin[i]
        end=(t+1)*dt
        for i in range(n):
            acceleration=0.
            if t>0:
                prev=(xyz_end[t-1,i]-xyz_start[t-1,i])/dt
                acceleration=np.sqrt(np.sum((velocity[i]-prev)**2))/dt
            for j in range(n):
                if i==j:continue
                z[t,i,j,5]=counts[t,i,j];m[t,i,j,5]=True
                z[t,i,j,16]=np.sum(counts[max(0,t-9):t+1,i,j]);m[t,i,j,16]=True
                z[t,i,j,17]=np.sum(counts[max(0,t-19):t+1,i,j]);m[t,i,j,17]=True
                z[t,i,j,18]=0.;m[t,i,j,18]=True
                z[t,i,j,19]=acceleration;m[t,i,j,19]=True
                if last[i,j]<0:continue
                age=end-last[i,j];z[t,i,j,0]=rxdb[i,j];z[t,i,j,1]=rxsin[i,j];z[t,i,j,2]=age
                m[t,i,j,0:3]=True
                if revstamp[i,j]>=0:
                    revage=end-revstamp[i,j]
                    z[t,i,j,3]=revsin[i,j];z[t,i,j,4]=revage;m[t,i,j,3:5]=True
                    minimum=min(rxsin[i,j],revsin[i,j])
                    z[t,i,j,12]=minimum;z[t,i,j,13]=minimum-threshold_db;m[t,i,j,12:14]=True
                    z[t,i,j,18]=1. if age<=2. and revage<=2. else 0.
                poseage=end-pose_stamp[i,j]
                if poseage<=10.:
                    predicted=poses[i,j]+vels[i,j]*poseage
                    for q in range(2):predicted[q]=min(195.,max(-195.,predicted[q]))
                    rel=predicted-xyz_end[t,i];dv=vels[i,j]-velocity[i]
                    distance=np.sqrt(rel[0]**2+rel[1]**2)
                    z[t,i,j,6:8]=rel[:2];z[t,i,j,8:10]=dv[:2];z[t,i,j,10]=distance
                    z[t,i,j,11]=(rel[0]*dv[0]+rel[1]*dv[1])/max(distance,1.)
                    m[t,i,j,6:12]=True
                if t>=6:
                    if m[t-6,i,j,1] and age<=2. and z[t-6,i,j,2]<=2.:
                        z[t,i,j,14]=(rxsin[i,j]-z[t-6,i,j,1])/3.;m[t,i,j,14]=True
                    if m[t-6,i,j,3] and revstamp[i,j]>=0 and end-revstamp[i,j]<=2. and z[t-6,i,j,4]<=2.:
                        z[t,i,j,15]=(revsin[i,j]-z[t-6,i,j,3])/3.;m[t,i,j,15]=True
    return z,m,tx_count,rx_count
