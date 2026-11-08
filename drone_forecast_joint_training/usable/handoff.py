"""선정 결과에서 스키마·모델·비교 CSV·팀원 연결 설명 생성."""
import csv,json,shutil
from pathlib import Path
from swarm_sim.regions import ROOT,read_json,write_json
from usable.collect import FEATURES as BASIC
from usable.spatial import NAMES
from usable.evaluate import RESULT,RUNS

DESCRIPTIONS=[
 '자기 x방향 속도 · m/s','자기 y방향 속도 · m/s','자기 고도 · m','자기 중계 역할 · 0/1','이웃 중계 역할 · 0/1',
 '직전 0.5초의 송신 시도 수','직전 0.5초의 ACK 확인 성공 수','해당 구간의 ACK 성공 비율',
 '마지막 성공 ACK의 수신 전력 · dBm','마지막 성공 ACK의 SINR · dB','마지막 ACK 이후 시간 · s','자기 큐 패킷 수',
 '자기 x좌표 · m','자기 y좌표 · m','이웃에서 받은 데이터의 성공 수','마지막 수신 데이터 헤더의 상태 시각 나이 · s',
 '데이터 헤더로 추정한 이웃 상대 x · m','데이터 헤더로 추정한 이웃 상대 y · m','이웃 상대 x속도 · m/s','이웃 상대 y속도 · m/s',
 '데이터 헤더 기반 추정 거리 · m','해당 거리의 변화율 · m/s','데이터 헤더 기반 현재 추정 경로 손실 · dB',
 '기존 헤더 기반 5초 손실 추정 · 선정 모델에서 제외','기존 헤더 기반 현재 LoS · 0/1',
 '기존 헤더 기반 5초 LoS · 선정 모델에서 제외','기존 헤더 기반 5초 손실 변화 · 선정 모델에서 제외','자기 위치에서 높이 40m 이상 건물까지 거리 · m',
 '마지막 수신 비콘 전력 · dBm','마지막 수신 비콘 SINR · dB','마지막 비콘 측정 나이 · s',
 '이웃이 보고한 역방향 SINR · dB','역방향 원래 측정 나이 · s','직전 구간 비콘 수신 수',
 '비콘 기반 추정 이웃 상대 x · m','비콘 기반 추정 이웃 상대 y · m','비콘 기반 이웃 상대 x속도 · m/s','비콘 기반 이웃 상대 y속도 · m/s',
 '비콘 기반 추정 거리 · m','비콘 기반 거리 변화율 · m/s','최근 관측 양방향 SINR 중 작은 값 · dB','관측 SINR 여유 · dB',
 '직전 3초 수신 SINR 기울기 · dB/s','직전 3초 역방향 SINR 기울기 · dB/s','직전 5초 비콘 수신 수','직전 10초 비콘 수신 수',
 '양방향 측정 나이 2초 이하 · 0/1','자기 속도 변화 기반 가속도 크기 · m/s²',
 '비콘 상태 + 지도 기반 현재 손실 추정 · dB','등속 외삽 3초 손실 추정 · dB','3초 손실 변화 추정 · dB',
 '등속 외삽 5초 손실 추정 · dB','5초 손실 변화 추정 · dB','비콘 상태 기반 현재 LoS 추정 · 0/1',
 '3초 등속 외삽 LoS 추정 · 0/1','5초 등속 외삽 LoS 추정 · 0/1','자기 3초 외삽선의 비행 제한 영역 교차 · 0/1',
 '이웃 3초 외삽선의 비행 제한 영역 교차 · 0/1','자기 5초 외삽선의 비행 제한 영역 교차 · 0/1','이웃 5초 외삽선의 비행 제한 영역 교차 · 0/1',
]


def main():
    import torch
    selected=read_json(RESULT/'selection.json')['selected_variant']+'_s11';source=RUNS/selected;dest=ROOT/'model';dest.mkdir(exist_ok=True)
    state=torch.load(source/'best.pt',map_location='cpu',weights_only=False);indices=state['feature_indices'];names=BASIC+NAMES
    assert len(names)==60 and len(DESCRIPTIONS)==60
    schema={'raw_features':60,'basic_features':48,'spatial_features':12,'selected_features':len(indices),
            'lookback_steps':40,'radio_interval_s':.5,'decision_interval_s':1,'validated_profile':{'n_drones':8,'n_relays':5,'altitude_m':50},
            'values_shape':'[N,N,48] with map_path or [N,N,60] precomputed',
            'features':[{'index':i,'name':name,'description':DESCRIPTIONS[i],'used_by_model':i in indices} for i,name in enumerate(names)]}
    write_json(dest/'feature_schema.json',schema)
    for name in ('best.pt','calibration.json','run.json','alarm_policy.json'):
        if (source/name).exists():shutil.copy2(source/name,dest/name)
    text='# 예측기 입력·출력 연결 명세\n\n## 데이터 흐름\n\n수신한 데이터·비콘과 자기 상태 → 기본 관측 48항목 → 정적 지도에서 파생한 12항목 → 관측 마스크·학습 전용 전처리 → Transformer → 위험 점수와 운용 상태.\n\n'
    text+='0.5초 간격 관측, 처음 20초 입력 축적, 1초 간격 예측. 입력 시각: 상대 시뮬레이션 시계의 0.5초 정렬 양수. 누락 구간: 결측 프레임의 명시적 제공. 결측 값 0, 마스크 False. 원래 관측값 0과 결측의 구분.\n\n'
    text+='현재 기본 수집기: `usable/collect.py`. 비콘 관측: `usable/beacons.py`. 지도 파생: `usable/spatial.py`. 미래 실제 상태·정답은 모델 입력에서 제외.\n\n'
    text+='원시 입력 60항목 중 선정 모델의 직접 입력 '+str(len(indices))+'항목. 자기 절대 좌표는 지도 계산에 활용하되 신경망의 직접 입력에서 제외.\n\n'
    text+='| 번호 | 이름 | 설명 | 직접 모델 입력 |\n|---|---|---|---|\n'
    for i,name in enumerate(names):text+=f'| {i} | `{name}` | {DESCRIPTIONS[i]} | {"포함" if i in indices else "제외"} |\n'
    text+='''
## 출력 상태

| 상태 | 의미 |
|---|---|
| warmup | 과거 20초 입력 축적 |
| between_decisions | 다음 1초 결정 시각까지 관측만 갱신 |
| insufficient_observations | 이웃별 유효 비콘 관측 부족 |
| stale_observations | 양방향 측정 중 하나의 나이 2초 초과 |
| currently_degraded | 측정상 현재 품질 기준 미달 |
| normal | 현재 관측 유효, 일반 경고 기준 미만 |
| watch | 일반 경고 기준 이상 |
| warning | 높은 알림 진입 문턱 도달 후 낮은 해제 문턱까지 유지 |

`risk_5s`: 향후 5초 구간 내 최소 1초 연속 링크 불가용 위험. 운영 후보 상태에서만 확률 제공. 관측 부족·노후·현재 저하는 None. 위험 0으로 치환 없음.

`new_warning_episode`: 같은 방향 링크의 알림 상태가 꺼짐에서 켜짐으로 바뀐 첫 경고만 True. 현재 배포의 `model/alarm_policy.json` 사용. 진입과 해제 문턱이 다른 히스테리시스. 강제 쿨다운·다중 프레임 대기 없음.

고신뢰 알림은 임박 위험 통보용. 3초 이상 선행 포착률의 큰 감소와 상충. 재배치 정책에는 알림의 0/1만 주기보다 연속 위험 점수와 원래 통신 관측을 함께 제공하는 방향의 별도 검증 필요.

## 재배치와 연결할 때

예측기 반환값을 재배치 정책의 관측으로 추가. 원래 위치·통신 관측은 유지. 관측 부족 플래그도 함께 전달. 위험 점수 자체를 보상으로 최대화하거나 최소화하는 설계보다 실제 연결 유지·전달 성공·이동 비용에 대한 별도 보상 검증 필요.

현재 3초 이동 주기와 예측 경고 선행시간의 정합성 확인 필요. 필요한 이동 시간과 제어 지연을 합친 요구 선행시간의 정의가 우선. 10초 보조 출력은 이번 배포의 검증 완료 예측 출력에 미포함.

현재 파이프라인은 전체 관측 기록을 시간 순서로 재생하는 형태. 폐루프 환경으로 바꿀 때는 환경 step에서 기본 관측을 제공하고 다음 행동을 적용하는 연결부 구현 필요. 이번 결과에 예측 기반 재배치 성능 또는 E2E 제어 성능 주장은 미포함.

재배치로 실제 단절을 예방한 경우의 예측 오경보 판정과 고정 궤적 예측 평가는 구분. 학습에 없던 새 정책의 궤적은 별도 평가 및 필요시 데이터 보강 대상.
'''
    (ROOT/'docs/입력_출력_명세.md').write_text(text,encoding='utf-8')
    comp=read_json(RESULT/'bank_comparison.json');rows=[]
    for name,item in comp['results'].items():
        v=item['balanced']['all'];rows.append({'run':name,'ap':v['average_precision'],'macro_ap':item['balanced']['macro_region_ap'],
             'precision':v['precision'],'recall':v['recall'],'fpr':v['false_positive_rate'],'brier':v['brier'],'samples':v['samples']})
    with (RESULT/'model_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
