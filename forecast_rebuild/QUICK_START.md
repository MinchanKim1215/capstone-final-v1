# 예측 개선 연구 후보 실행

기존 코드에 덮어쓰지 않는다. 후보와 보정의 정확한 경로는 `artifacts/research_candidate.json`에 저장한다. 선정 결과는 `RESULTS_20261102.md`를 기준으로 확인한다.

## 결과와 상태

프로젝트 루트의 `00_예측개선_현황.bat`를 실행하면 학습·선택·독립 확인 상태를 표시한다. 상태가 완료여도 실사용 인증 완료를 뜻하지 않는다.

기존 가상환경을 그대로 사용한다. 프로젝트 루트의 PowerShell에서 실행한다.

```powershell
$env:PYTHONIOENCODING='utf-8'
.\drone_forecast_joint_training\.venv\Scripts\python.exe -B -m forecast_rebuild.status
.\drone_forecast_joint_training\.venv\Scripts\python.exe -B -m unittest forecast_rebuild.test_core forecast_rebuild.test_codec forecast_rebuild.test_diagnostics -q
.\drone_forecast_joint_training\.venv\Scripts\python.exe -B -m forecast_rebuild.check_selected_runtime
```

마지막 명령은 선택 자료의 짧은 수신 기록으로 배치·실시간 확률, 누락 관측 처리, 센서 자격, 미검증 경보 차단을 확인한다. 장시간 학습이나 PPO 교체를 시작하지 않는다.

## 로컬 예측 API

`runtime.LocalForecast`는 중계 드론 한 대가 실제 보유한 수신 이력만 받는다. 기존 `adaptive_joint_expanded` 관측기와 같은 로컬 48개 값·마스크·9개 의도 표현을 사용한다. 통신 모듈의 원시 로그를 이 표현으로 바꾸는 장비별 어댑터는 별도다. 다른 드론의 수신되지 않은 내부 정보를 채워 넣으면 안 된다.

```python
from forecast_rebuild.contracts import read
from forecast_rebuild import ARTIFACTS
from forecast_rebuild.runtime_stable import StableLocalForecast as LocalForecast

candidate = read(ARTIFACTS / 'research_candidate.json')
predictor = LocalForecast(
    model=candidate['model'],
    policy=candidate['alarm_policy'],
    city=city,                 # 현재 공개 지도. 명시적 _forecast_scenario 필요
    nominal=nominal_route_xy,   # 선정 모델이 요구하는 비행 전 공개 경로
    observer=relay_id,
    n_drones=n_drones,
    n_search=n_search,
    device='cpu',
    confirmation_goal_met=False,
    hardware_validated=False,
)
result = predictor.observe(
    time_end_s=0.5,             # 임무 상대 시각. 0.5초 간격, 순서대로 전달
    values=local_values,        # [N, 48]
    masks=local_masks,          # [N, 48], 실제 관측 여부
    intents=received_intents,   # [N, 9], 자기 명령·실제 받은 이웃 명령
    energy=local_energy,        # 선택 측정 [N, 6], 없으면 None
)
```

`city`와 공개 경로를 준비하는 시뮬레이션 예시는 `planning.public_context`와 `check_selected_runtime.py`에 있다. 실제 지도 경로·좌표계·무선 조건을 준비하지 않고 위 변수명만 복사하여 실행하는 코드는 아니다.

20초 준비 이력 이후 매초 결과를 반환한다. 모든 0.5초 프레임을 제공하며 미수신 프레임도 마스크로 명시한다. 최대 임무 길이는 기본 7,200초이며 새 임무 전 `reset()`한다.

## 출력 의미

| 필드 | 의미 |
|---|---|
| `risk_5s` | 다음 5초 내 지속 단절 위험. 필수 RF 지원이 없으면 `None` |
| `risk_3to7s` | 앞으로 3~7초 구간의 지속 단절 위험 |
| `advisory_risk_5s` | 확대된 관측 조건의 연구용 점수. 검증된 확률이라는 뜻이 아님 |
| `advisory_first_alert` | 기본 RF 자격의 문턱·히스테리시스 최초 경보 |
| `alarm_data_eligible` | 현 보정 정책의 센서 신선도까지 갖췄는지 여부 |
| `qualified_first_alert` | 보정 정책의 자료 자격을 충족한 최초 경보. 실사용 승인과 다름 |
| `trusted_alarm` | 전체 성능 목표·독립 확인·하드웨어 검증까지 충족한 경보. 본 연구 후보는 자동 활성화하지 않음 |

`None`을 0으로 바꾸지 않는다. 센서가 없으면 `energy=None`을 전달한다. 측정값을 0dBm 또는 가짜 정상값으로 채우지 않는다.

에너지 배열 열은 자기 I+N(dBm), 실제 받은 이웃 I+N(dBm), 이웃 측정 나이(s), 자기 측정 유효 여부, 이웃 측정 유효 여부, 자기 측정 나이(s)다. 비콘 전달 지연과 물리 측정 시각을 보존한다. `beacon_codec.py`는 송신·측정 시각과 CRC를 포함하는 128바이트 예산 시제품이며 실제 무선 드라이버가 아니다.

## 지원 가정

선정 모델은 이번 시뮬레이터의 2.4GHz·50m·10dBm·1MHz·1Mbps, TDMA, Bullington, 그림자 감쇠 표준편차 1.5dB/상관 시간 4초 조건에서 검증했다. `city._forecast_scenario`의 명시된 조건이 다르면 API가 거부한다. 같은 숫자를 설정하는 행위가 실제 장비의 보정·검증을 대신하지는 않는다.

기체·통신모듈·GNSS·안테나·실제 채널을 정하고 실측 자료로 별도 검증해야 한다. 기존 PPO에 자동 대체하지 않았다. 연결 시에는 입력·출력 시간 의미를 맞추고 폐루프 비교를 새로 수행한다.

## 재현과 보존

실험 자료와 후보 태그는 불변으로 취급한다. 같은 태그에 다른 가중치를 덮어쓴 뒤 이전 예측 캐시를 재사용하지 않는다. 새 실험은 새 폴더·태그로 만든다. 확인 결과를 보고 같은 확인 자료에 맞춰 반복 조정한 점수를 독립 성적으로 보고하지 않는다.

`artifacts/model_alarm_freeze.json`은 코드·선택 모델·보정·문턱을 묶는다. 고정된 파일이 달라지면 확인 평가가 거부된다. `artifacts/audit.json`은 원시 자료·원본 모델 보존과 일부 캐시 예측 재계산을 확인한다. 기존 최종 평가 32개 지도는 본 연구에서도 열지 않는다.
