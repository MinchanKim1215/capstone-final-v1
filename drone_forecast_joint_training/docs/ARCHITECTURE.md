# v5 시스템 계약과 검증 설계

## 예측기와 제어 흐름

1. RF·링크 관측은 .5초마다 기록합니다. 각 관측자 5기 × 상대 8노드에 대해 공개 측정 48개, 유효 마스크 48개, 알려진 의도 9개를 사용합니다.
2. 현재 시점까지 **40프레임, 20초**를 각 유향 링크별로 분리해 Transformer에 넣습니다. 전체 window는 모두 과거이므로 내부 attention은 window 전체를 볼 수 있습니다. 시간축 이후의 관측은 API에 전달하지 않습니다.
3. 알려진 정적 지도와 수신한 현재 비행 의도에서 계산한 기하 특징 46개를 결합합니다. 이는 실제 미래 위치·미래 재머 상태가 아닌 인과적 외삽입니다.
4. 32차원·4헤드·2층 Transformer가 네 가지 단절 확률을 출력합니다. dropout=0, math SDPA를 사용해 수집과 PPO 재평가의 차이를 줄입니다.
5. 예측 위험을 현재 연결 문맥과 물리 후보 25개의 연결 이득에 결합하고, 공유 actor와 작은 adapter가 5기의 재배치 행동을 고릅니다. 기존 안전·이동·명령 제약을 적용한 뒤 비행을 실행합니다.
6. 현재의 전체 상태만 사용하는 중앙 critic이 학습을 돕습니다. 실제 actor 입력에는 critic의 전체 상태·미래 정답을 전달하지 않습니다.

LightGBM의 점수·latent·residual을 Transformer나 policy에 넣지 않습니다. `World(forecast=False)`가 기존 예측기를 생성하지 않으며 RF recorder만 연결합니다. 기존 모듈 import 때문에 LightGBM Python 의존성은 남아 있습니다. 비교 명령에서만 명시적으로 트리 모델을 로드합니다.

## 확률과 정답

| 출력 | 의미 |
|---|---|
| `risk_5s` | 현재 연결된 링크에서 향후 5초 창에 1초 이상 연속 단절이 존재할 위험 |
| `onset_3to4s` | 약 3~4초 선행 구간에서 지속 단절이 시작할 위험 |
| `onset_3to7s` | 약 3~7초 선행 구간에서 지속 단절이 시작할 위험 |
| `onset_0to7s` | 향후 약 0~7초 구간의 지속 단절 시작 위험 |

정확한 .5초 이산 구간은 `solo/targets.py`에 있고, 기존 `advance.features.targets`의 해당 네 head와 동일한지 검사합니다. 라벨은 RF 로그에서 **오프라인으로만** 만들며, 에피소드 끝의 관측되지 않은 미래는 음성으로 채우지 않고 censor합니다. 현재 단절된 링크와 self 링크는 예측 대상에서 제외합니다. 완전한 40프레임 전에는 지원 마스크가 false입니다.

출력층은 사건 포함 관계에 맞춰 `p34 ≤ p5`, `p34 ≤ p37`, `p5 ≤ p07`, `p37 ≤ p07`를 강제합니다. 단순 독립 sigmoid가 아닙니다. monotonic 구조만으로 확률 보정이나 성능이 보장되지는 않습니다.

RF·의도·기하 입력은 수집 시 float16으로 양자화해 저장하고 네트워크 계산은 float32로 합니다. 실시간 추론도 같은 양자화 계약을 적용합니다. 물리·RF 시뮬레이터의 내부 계산 정밀도를 낮추지는 않습니다. masked 센서 값은 0으로 대체하고 마스크를 별도로 입력합니다.

## 학습과 공동학습

- 지도학습: 링크별 양성/음성 샘플을 균형 추출하되 importance weight로 해당 비행의 원래 유병률을 보정합니다. 매 step에 한 비행을 균등하게 선택합니다. loss head weight는 `[1,2,2,1]`입니다.
- `fixed`: 같은 사전학습 Transformer를 고정하고 actor·adapter·critic을 학습합니다.
- `joint`: actor·adapter·critic에 더해 Transformer 인코더와 확률 출력층도 학습합니다. PPO 보상과 지도 BCE 보조 손실을 함께 사용합니다.
- `physics`: learned forecaster 없이 물리 후보와 기존 actor를 쓰는 기준선입니다.

PPO clip=.15, gamma=.995, GAE lambda=.95, 전역 gradient clip=.5, entropy 계수=.003, physical teacher KL 계수=.003, 지도 손실 계수=.1입니다. 기본 학습률은 actor/adapter 1e-4, Transformer encoder 1e-5, 예측 출력층 5e-5, critic 3e-4입니다. 안전한 출발값이며 모든 맵의 최적값이라고 주장하지 않습니다. KL .02를 넘는 epoch는 중단합니다.

**핵심:** rollout 때 계산한 예측 점수를 고정한 채 PPO를 학습하지 않습니다. 원시 RF·마스크·의도·기하·물리 후보를 저장하고, 업데이트 때 현재 Transformer로 문맥·후보 이득·teacher 보정을 다시 미분 가능하게 계산합니다. 따라서 인코더와 확률 head가 보상을 받을 수 있습니다. PPO 손실만 따로 미분해 두 경로의 gradient norm을 기록합니다. auxiliary BCE가 있다는 이유만으로 공동학습 성공이라고 부르지 않습니다.

각 update 시작에는 저장된 행동 log-probability와 현재 모델의 재계산 값이 맞는지 전 표본 검사합니다. stale neural feature나 잘못된 체크포인트 혼합을 발견하면 중단합니다. dropout=0과 동일 관측 양자화로 이 재평가를 안정화했습니다.

## 교수 피드백 반영

누적 보상 값의 크기는 reward scaling에 따라 달라지므로 성능 증거로 사용하지 않습니다. 통신 단절·패킷 손실의 비용을 이동 패널티보다 크게 유지합니다. 예를 들어 전체 단절 보상 항은 -3, 이동 패널티 상한은 -.003으로 기존 의도를 보존합니다. 실제 선택은 CTR, PDR, 지연 반영 PDR, 최장 단절, 비행 시간, 연속 궤적 안전 감사로 판단합니다. 단지 보상 계수를 키워 성능이 향상됐다고 보고하지 않습니다.

## 데이터 분리

245개 맵을 공간적 중첩 그룹 단위로 분리했습니다. 훈련 166, 새 검증 24, 최종 테스트 32, 과거 검증 이력이 있는 제외 맵 23개입니다. 전체 196개 공간 그룹이 split을 넘지 않는지 검사합니다. 과거 평가에 쓴 지역을 새 테스트라고 재사용하지 않습니다.

검증은 맵별 4조건 96회, 테스트는 128회입니다. 축소 pilot은 서로 다른 검증 8개 맵에서 1조건씩만 사용합니다. 학습은 seeded permutation을 순회하며 6회 순환마다 각 맵에 두 방향 × 세 재머 조건을 배정합니다. 192회 데이터 수집만으로 모든 맵의 모든 조건을 본 것은 아닙니다.

정책이 바뀌면 비행 경로와 단절 정답의 분포도 달라집니다. 따라서 각 정책의 closed-loop 지표와, **동일한 기준선 관측에 예측기만 적용한 AP**를 구분해 저장합니다. 비교 명령은 두 예측기의 유효 표본도 맞춥니다. 프레임·유향 링크를 독립적인 수십만 실험으로 취급하지 않습니다. paired CI는 공간 지역 단위로 bootstrap합니다.

## 경보

.5초 RF를 읽지만 행동과 경보 판단 주기는 3초입니다. 최소 1초 지속 단절 사건의 최초 경보를 평가하며 21초 warmup, 8초 tail censor, 최소 3초 선행 조건을 명시합니다. 단순 분류 recall은 조기 경보 recall이 아닙니다.

검증 데이터에서 quantile·고정 threshold와 hysteresis 비율을 탐색합니다. precision≥.8, 오경보≤5/드론·시간, 최초 경보≥10건을 동시에 만족한 후보 중 사건 recall이 높은 것을 고릅니다. 없으면 disabled입니다. 전체 검증 선택으로 고정한 규칙만 `solo.alarms.AlarmMonitor`가 활성화할 수 있고, checkpoint·source·plan·분석 코드 SHA를 검사합니다. 테스트에서 threshold를 재조정하지 않습니다.

## 재개·파일 무결성

checkpoint는 임시 파일 쓰기→fsync→원자적 replace 후, SHA를 가진 latest 포인터를 원자적으로 갱신합니다. 모델·Adam·난수·구성·누적 update·초기 예측기 SHA·핵심 소스 150개 SHA·맵 계획 SHA를 기록합니다. 원시 비행은 NPZ+메타 JSON 모두 있어야 재사용하며 배열 SHA를 검사합니다. worker 완료 순서가 훈련 표본 순서를 바꾸지 않습니다.

Windows는 `spawn`, `freeze_support`, `__main__` 진입점, 파일 잠금을 사용합니다. 예측기 map cache는 지역별 잠금 하나로 두 방향의 공용 캐시 쓰기를 직렬화합니다. CUDA 체크는 실제 backward를 수행합니다. 이번 전달 환경은 CPU이므로 이 설계가 사용자 Windows GPU에서 실측 통과했다고 표현하지 않습니다.

## 공개 추론 API

`StreamingController(checkpoint)`를 생성하고 `.5, 1.0, ...`초에 `observe(time_s, values, masks, intents)`를 호출합니다. time=0부터 시작해 매 3초 `act(...)`를 호출합니다. 입력 shape는 클래스에 명시되어 있습니다. `physics_score`는 **예측 점수가 들어 있지 않은** `solo.features.observe_physics`의 teacher이고, gain은 물리적 링크 margin 개선량입니다. 자체 통합에서 LightGBM이 섞인 후보를 전달하면 단독 예측기 계약을 어기게 됩니다.

full-validation 선택 후 `AlarmMonitor(checkpoint, frozen_selection)`에 같은 시점의 확률·support를 전달해 경보를 생성합니다. 추론은 미래 배열이나 정답을 받지 않습니다. 유향 링크별 위험은 팀의 end-to-end 단절 확률과 동일하지 않습니다.

이 API는 시뮬레이션의 정책·예측 경계입니다. 실제 장비용 센서 드라이버와 무선 패킷 입력, 명령 전달·비상 정지, RF 실측 보정은 제공되지 않습니다. 연속 궤적 검사는 시뮬레이터의 이동·지도 가정에서 유효하며 현실의 풍속·위치 오차까지 검증하지 않습니다.
