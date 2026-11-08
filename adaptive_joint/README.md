# 지도별 배치 선택 + 1초 예측 + 공동 PPO 환경

2026-10-30 구성. 새 장시간 학습을 시작하기 전의 코드·실행 환경이다.
**학습 준비 검증과 성능 검증은 다르다. 기존 best 176의 성능을 유지하거나 향상했다는 결과는 아직 없다.**

## 구성

`지도·출동 대수·최소 수색 대수 → 임무 전 배치 선택 → 1초 단절 예측 → 3초 중계 이동 → 실제 결과로 PPO 갱신`

- 총 8 / 수색 3 / 중계 5, V형 40m.
- 총 6 / 수색 4 / 중계 2, 횡렬 20m.
- 총 10 / 수색 6 / 중계 4, 횡렬 20m.
- 총 6 / 수색 4 / 중계 2, V형 40m.

출동 가능 대수·최소 수색 수를 만족하지 않는 후보는 선택 확률을 0으로 만든다.
드론 수는 **출발 전** 선택한다. 비행 중 드론을 추가하거나 역할 수를 바꾸지 않는다.
선택기는 지도 ID·난수·실제 간섭 상태·미래 채널·실험 결과를 입력받지 않는다.
건물 수·높이 통계뿐 아니라 8×8 건물 점유 격자와 비행 제한 격자를 포함한 140개 공개 입력으로 지도 배치를 구분한다.
관측 부족·노후 링크는 위험도 0으로 표시하지 않고 unknown으로 유지하며, 정책에도 예측 지원 비율을 별도로 제공한다.

## 실제 갱신되는 부분

| 부분 | 학습 신호 |
|---|---|
| Transformer 인코더·새 17분류 시간 출력층 | PPO 행동 보상 + 실제 다음 단절의 지도학습 오차 |
| 중계 이동 정책 | 안전한 25개 이동 후보에 대한 PPO |
| 중앙 가치망 | 현재 상태에서의 누적 보상 추정. 실행 Actor의 입력과 분리 |
| 임무 전 배치 선택기·가치망 | 임무당 한 번의 배치 선택에 대한 PPO |

예측기는 기존 1초 실험 best 5,500의 **시간 구간 가중치**로 초기화한다.
그 모델은 8대 자료에서 개발된 모델이다. 6·10대에 대한 성능 검증 완료를 뜻하지 않는다.
기존 네 확률 출력층은 호환을 위해 보존하되 v6 학습·행동에 사용하지 않는다.
새 가변 대수 정책망과 선택기는 새로 학습해야 한다. 초기 중계 정책은 관측 기반 물리 후보 점수로 시작한다.

시간 정답은 다음 **1초 이상 지속 단절의 첫 시작**이다. `[0,8)`초를 0.5초씩 16개로 나누고 시작 없음 1개를 더한다.
요약 구간은 `[0,5)`, `[3,4)`, `[3,7)`, `[0,7)`로 엄격히 통일했다.
첫 1초 실험의 일부 닫힌 상한 요약과 다르므로 AP를 직접 증감 비교하면 안 된다.
미래 확인 구간이 없는 마지막 8.5초는 정답 유효성에서 제외한다. 실제 행동은 미래 정답 유효성으로 차단하지 않는다.

## 환경과 보존

- Windows / Ryzen 5 7600 / RAM 약 32GB / RTX 4070 SUPER 12GB.
- 이미 검증된 `../drone_forecast_joint_training/.venv` 사용. Python 3.12.14, PyTorch 2.9.1+cu128.
- `requirements-lock.txt`는 이 PC에서 확인한 설치 목록이다. GPU 패키지를 임의로 교체하지 않는다.
- 기존 소스·설정·best 176·physics·새 예측기 best의 SHA256을 `artifacts/legacy_before.json`에 보존한다.
- 새 캐시·Numba 캐시·로그·체크포인트·평가는 모두 **이 폴더의 artifacts**에만 기록한다.
- 기존 v5와 신규 v6의 체크포인트 형식은 다르다. 자동 덮어쓰기·가중치 강제 대입을 하지 않는다.

## 실행 순서

배치 파일을 더블클릭하거나, 작업공간 최상위에서 다음 명령을 실행한다.

```powershell
$py = '.\drone_forecast_joint_training\.venv\Scripts\python.exe'
& $py -B -m adaptive_joint.run doctor
& $py -B -m adaptive_joint.run selftest
& $py -B -m adaptive_joint.run prepare
& $py -B -m adaptive_joint.run assets --workers 4
& $py -B -m adaptive_joint.run preflight
& $py -B -m adaptive_joint.run plan
```

`doctor`, `selftest`, `prepare`, `preflight`, `plan`은 **256회 새 학습을 시작하지 않는다.**
`assets`는 train·개발 190지도 × 2방향 × 4배치의 정적 명목 경로를 준비한다. 실제 채널 평가·학습은 하지 않는다.
지도 등록 정보에 빠진 건물 통계는 원본 정적 지도에서 같은 정의로 직접 계산한다. 빠진 값을 0으로 대체하지 않는다.
preflight만 실행용 소규모 PPO 갱신을 하며, 결과는 성능 비교·best 선택·경보 보정에서 제외한다.
실제 CUDA gradient, 새 예측기까지의 정책 gradient, 가중치 갱신, 가변 대수 입력,
원본 8대 물리 결과 일치, 전체 임무, 저장·복원·Adam 재개를 확인한다.

### 새 학습 시작 명령 — 준비 작업에서는 실행하지 않음

```powershell
& $py -B -m adaptive_joint.run train --run adaptive_v6_20261030
```

기본 목표: 256 update × 8비행. 초기 16 update는 네 배치에 균등한 공통 사례를 배정하고,
그 이후에는 배치 선택기의 실제 확률로 출발 배치를 선택한다.
이 균등 수집 기간에는 선택기 PPO를 갱신하지 않는다. 강제한 선택을 on-policy 행동으로 가장하지 않는다.
아래 명령만 중단된 **동일 설정·동일 소스** 실행을 재개한다.

```powershell
& $py -B -m adaptive_joint.run train --run adaptive_v6_20261030 --resume
```

`artifacts/runs/<run>/status.json`, `progress.json`, `history.json`, `best.json`과 `artifacts/logs/`에서 진행을 확인한다.
저장은 임시 파일·해시·latest 포인터 순서로 확정한다. 중단된 수집은 동일 체크포인트의 완료 사례를 검증한 뒤 재사용한다.
실행 중 설정을 바꾸지 않는다. 다른 설정은 새 연구 폴더/새 protocol이 필요하다.

## 평가 단계

1. 원래 train 166개 지역만 PPO 수집에 사용한다.
2. validation 24개 지역은 인접 공간 그룹을 유지해 **모델 선택 8 / 경보 보정 8 / 확인 8**로 나눴다.
3. 이 24개는 이전 연구에서 사용된 개발 지역이다. 새로운 독립 최종 평가라고 주장하지 않는다.
4. 기존 test 32개 지역의 실제 지형 실행·결과는 사전 검증에서 열지 않는다. 모델·경보를 동결한 `finaltest`에서만 접근한다.

학습 후 체크포인트 경로를 실제 best.json에서 확인하고 실행한다.

```powershell
& $py -B -m adaptive_joint.run evaluate --checkpoint '<best checkpoint>' --paired --cases 0
& $py -B -m adaptive_joint.run replay --checkpoint '<best checkpoint>' --rollouts '<evaluation/.../selection/legacy_v5>'
& $py -B -m adaptive_joint.run calibrate --checkpoint '<best checkpoint>'
& $py -B -m adaptive_joint.run confirm --checkpoint '<best checkpoint>' --alarm '<calibration/alarm.json>'
& $py -B -m adaptive_joint.run freeze --run adaptive_v6_20261030 --alarm '<calibration/alarm.json>'
& $py -B -m adaptive_joint.run finaltest --freeze '.\adaptive_joint\artifacts\frozen_selection.json'
```

`--paired`는 같은 지역·방향·난수·간섭에서 네 고정 배치를 실행하고, 실제 선택기의 선택과 비교한다.
가용 대수와 최소 수색 조건이 허용하면, 원본 best 176도 동일한 새 물리 환경에서 동결 재생해 통신 지표를 비교한다.
이 원본 모델의 any-event 출력은 별도 보존하며, 새 next-event AP로 가장하지 않는다.
`replay`는 고정된 동일 관측·시점·지원 마스크·정답에서 1초 사전학습 모델과 새 공동학습 예측기를 비교한다.
원본 176 점수도 같은 새 정답으로 평가하되, 기존 HTML의 native AP와는 다른 평가임을 명시한다.
사후 정답을 보고 고른 oracle은 **선택 가능성의 오프라인 상한**으로만 표시한다. 온라인 입력이 아니다.
지도별 승자가 거의 같으면 자동 선택의 이득이 작을 수 있다. 학습 전에 향상률을 만들어내지 않는다.

경보 목표 기본값은 정밀도 90%, 3초 선행 사건 재현율 80%, 드론·시간당 오경보/늦은 경보 1회 이하, 경보 20개 이상이다.
이는 **개발 목표**이며 교수님 지정값이나 달성값이 아니다. 재현율을 포함한 조건을 모두 만족하지 못하면 경보는 비활성화한다.
0개 경보의 정밀도는 0%나 100%가 아니라 정의 불가로 기록한다.
`calibrate`는 경보 문턱을 고르는 단계다. 확률 자체의 temperature scaling을 적용했다는 뜻은 아니다.
원시 확률의 신뢰도는 Brier 등으로 별도 평가한다. on-policy 예측은 해당 행동 정책 아래의 실제 미래를 대상으로 한다.
best 승격에는 전 비행 완료·안전 검사·개별 PDR 90% 이상·명목 완료 비행 대비 시간비 1.15 이하를 요구한다.
이 역시 개발 목표다. 명목 비행 자체가 미완료인 사례의 시간비는 정의 불가로 남기고 절대 완료 여부를 평가한다.

## 꼭 구분할 해석

- 새 환경의 준비 완료는 성능 향상 입증이 아니다. 아직 새 정책의 대규모 학습이 필요하다.
- 네 배치는 과거 완화된 탐색 기준의 후보다. 모두 검증된 최적 배치가 아니며 6대 V형은 과거 재검증 PDR이 미달했다.
- 기체 수·수색 수·중계 수·실제 비콘 비용·총 30packet/s의 동일 트래픽 조건을 함께 보고한다.
- **CTR은 수색 드론 모두가 서로 도달 가능한 시간 비율이다. 이 시뮬레이터에는 별도 지상 기지국 노드가 없다.**
  이전 대화에서 이를 ‘기지국 연결률’이라고 표현한 부분은 정정한다.
- 예측 AP, AUROC, 표본 정밀도·재현율, 적시 첫 경보 정밀도·사건 재현율, 통신·임무 지표를 합쳐 한 성공률로 표시하지 않는다.
- `artifacts/readiness.json`과 `artifacts/preflight/.../checks.json`이 실제 사전 검증 근거다.
