"""미래 정답 없이 실행하는 0.5초 단위 위험 예측 인터페이스.

입력은 성공적으로 받은 데이터/비콘과 자신의 위치·속도에서 만든 48항목입니다.
시뮬레이터용 관측 생성은 usable.collect, 입력 항목 이름은 usable.collect.FEATURES.
예측은 드론별 링크 위험이며 자동 비행 명령을 생성하지 않습니다.
"""
from collections import deque
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from swarm_sim.regions import read_json,sha256_file
from usable.model import RiskTransformer


class LinkRiskPredictor:
    def __init__(self,checkpoint,calibration=None,threads=2,map_path=None):
        path=Path(checkpoint);state=torch.load(path,map_location='cpu',weights_only=False)
        torch.set_num_threads(threads)
        self.model=RiskTransformer(**state['spec']);self.model.load_state_dict(state['state_dict']);self.model.eval()
        self.cal=read_json(Path(calibration) if calibration else path.with_name('calibration.json'))
        if self.cal.get('checkpoint_sha256',sha256_file(path))!=sha256_file(path):raise ValueError('보정값과 가중치 불일치')
        policy_path=path.with_name('alarm_policy.json')
        self.alarm_policy=read_json(policy_path) if policy_path.exists() else None
        if self.alarm_policy and self.alarm_policy.get('checkpoint_sha256',sha256_file(path))!=sha256_file(path):
            raise ValueError('알림 정책과 가중치 불일치')
        self.features=np.asarray(state['feature_indices'],int)
        self.spatial_context=None
        if max(self.features)>=48 and map_path is not None:
            from usable.spatial import context
            self.spatial_context=context(map_path)
        self.mean=np.asarray(state['normalizer']['mean'],np.float32)[self.features]
        self.std=np.asarray(state['normalizer']['std'],np.float32)[self.features]
        self.values=deque(maxlen=40);self.masks=deque(maxlen=40)
        self.last_time=None;self.shape=None;self.previous_warning=None

    def reset(self):
        self.values.clear();self.masks.clear();self.last_time=None;self.shape=None;self.previous_warning=None

    def observe_local(self,time_end_s,observer_id,values,mask):
        """한 드론이 자신의 이웃별 관측 [N,F]만 제공하는 분산 추론용 진입점.

        다른 드론의 비공개 관측 행은 모두 결측입니다. 인스턴스 하나당
        관측 드론 한 대를 고정하고, 그 드론의 결과 행만 반환합니다.
        """
        x=np.asarray(values,np.float32);m=np.asarray(mask,bool)
        if x.ndim!=2 or x.shape!=m.shape or not 0<=observer_id<len(x):raise ValueError('N×F 로컬 관측 필요')
        full=np.zeros((len(x),len(x),x.shape[-1]),np.float32);valid=np.zeros_like(full,bool)
        full[observer_id]=x;valid[observer_id]=m
        result=self.observe(time_end_s,full,valid)
        result['links']=[r for r in result['links'] if r['source']==observer_id]
        return result

    @torch.inference_mode()
    def observe(self,time_end_s,values,mask):
        """과거 끝 시각까지 확보한 [N,N,48] 값/관측 여부 입력. 1초마다 위험 계산."""
        if not np.isfinite(time_end_s) or time_end_s<=0 or not np.isclose(time_end_s/.5,round(time_end_s/.5),atol=1e-6):
            raise ValueError('시뮬레이션 기준 양수 시각, 0.5초 정렬 필요')
        values=np.asarray(values,np.float32);mask=np.asarray(mask,bool)
        if values.shape!=mask.shape or values.ndim!=3 or values.shape[0]!=values.shape[1] or values.shape[-1] not in (48,60):
            raise ValueError('N×N×48 기본 관측 또는 N×N×60 지도 보강 관측과 같은 크기의 마스크 필요')
        if max(self.features)>=48 and values.shape[-1]==48:
            if self.spatial_context is None:raise ValueError('이 모델은 정적 지도 경로 map_path 또는 미리 계산한 60항목 관측 필요')
            from usable.spatial import features
            z,zm=features(self.spatial_context,values,mask)
            values=np.concatenate((values,z),axis=-1);mask=np.concatenate((mask,zm),axis=-1)
        if not np.isfinite(values).all() or np.any(values[~mask]!=0):raise ValueError('결측은 0과 마스크 False로 표현')
        if self.shape is not None and self.shape!=values.shape:raise ValueError('드론 수 변경 전 reset 필요')
        if self.last_time is not None and not np.isclose(time_end_s-self.last_time,.5,atol=1e-6):
            raise ValueError('0.5초 간격 필요. 누락 구간은 유효 관측 없음으로 명시')
        self.shape=values.shape;self.last_time=float(time_end_s);self.values.append(values.copy());self.masks.append(mask.copy())
        n=values.shape[0]
        if self.previous_warning is None:self.previous_warning=np.zeros((n,n),bool)
        if len(self.values)<40:return {'time_end_s':time_end_s,'status':'warmup','links':[]}
        # 학습의 1초 평가 주기와 일치합니다. 매 0.5초 점수를 만들지 않습니다.
        if not np.isclose(time_end_s%1.,0.,atol=1e-6):return {'time_end_s':time_end_s,'status':'between_decisions','links':[]}
        history=np.asarray(self.values);m=np.asarray(self.masks);off=~np.eye(n,dtype=bool)
        enough=np.sum(history[...,33]>0,axis=0)>=3
        src,dst=np.where(off&enough);rows=[]
        probability=np.empty(0)
        if len(src):
            x=history[:,src,dst][:,:,self.features].transpose(1,0,2)
            mm=m[:,src,dst][:,:,self.features].transpose(1,0,2)
            x=np.where(mm,np.clip((x-self.mean)/self.std,-8,8),0).astype(np.float32)
            logits=self.model(torch.from_numpy(x),torch.from_numpy(mm)).numpy()
            probability=expit(logits[:,0]/self.cal['temperature'])
        lookup={(int(i),int(j)):float(p) for i,j,p in zip(src,dst,probability)}
        tight=self.cal['threshold_conservative'];balanced=self.cal['threshold_balanced']
        for i,j in zip(*np.where(off)):
            p=lookup.get((int(i),int(j)));old=bool(self.previous_warning[i,j]);fresh=bool(mask[i,j,46] and values[i,j,46]>.5)
            margin=float(values[i,j,41]) if mask[i,j,41] else None
            if p is None:status='insufficient_observations'
            elif not fresh:status='stale_observations'
            elif margin is None:status='insufficient_observations'
            elif margin<=0:status='currently_degraded'
            else:
                if self.alarm_policy is not None:
                    chosen=self.alarm_policy['selected']
                    active=bool(chosen and ((old and p>=chosen['exit_threshold']) or p>=chosen['enter_threshold']))
                else:active=bool(tight is not None and p>=tight)
                status='warning' if active else 'watch' if p>=balanced else 'normal'
            warning=status=='warning';self.previous_warning[i,j]=warning
            # 지원 조건을 벗어난 점수는 확률처럼 노출하지 않습니다. 0으로 치환하지 않습니다.
            reported=p if status in ('normal','watch','warning') else None
            rows.append({'source':int(i),'peer':int(j),'risk_5s':reported,'status':status,
                         'new_warning_episode':bool(warning and not old),
                         'inbound_measurement_age_s':float(values[i,j,30]) if mask[i,j,30] else None,
                         'reverse_measurement_age_s':float(values[i,j,32]) if mask[i,j,32] else None,
                         'observed_margin_db':margin})
        return {'time_end_s':float(time_end_s),'status':'ready','links':rows}
