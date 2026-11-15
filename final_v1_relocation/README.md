# Final V1 재배치 개선 · 예측 오류와 단절 상황 대응

10주차 역할(예측 오류와 단절 발생 상황에서의 재배치 개선)의 개선안 1~4를 구현한 패키지다. 학습된 V7 PPO와 V21 예측기는 그대로 두고, 그 위의 규칙만 바꾼다.

## 왜 원본 파일을 직접 고치지 않았나

개선안 문서의 "바꿀 코드"는 V21 실행 시 SHA-256으로 검사된다.

* `forecast_learning/control.py`는 `forecast_learning.training.verify_freeze()`가 `load_profile()`마다 해시를 확인한다. 한 글자만 바꿔도 V21이 `Locked predictor/control source changed`로 멈춘다.
* `forecast_avoidance/controller.py`와 `forecast_learning/control.py`는 `final_v1/protocol.py`의 `dependency_sha256`에도 고정돼 있다.
* `RadioState.known`은 V21 LightGBM 입력(`missing_search`)이기도 하다. known 판정을 바꾸면 학습된 예측기의 입력 분포가 달라진다.

그래서 `final_v1/transformer_upgrade/bridge.py`와 같은 방식을 쓴다. 비행 중에만 `final_v1.runner`의 `choose`(재배치 판단)와 `CommandChannel`(명령 채널)을 이 패키지 것으로 바꿔 끼운다. 모든 개선안을 끈 `v21` 설정은 Final V1 V21과 같은 결정을 낸다(단위 검사 300개 상황, 캠페인의 `v21_matches_final_v1` 확인).

| 개선안 | 문서의 "바꿀 코드" | 구현 위치 |
|---|---|---|
| 1. 단절 중 복구 재배치 | `forecast_avoidance/controller.py` choose() 첫 조건, `forecast_prevention/controller.py` Inbox.state()의 known, `final_v1/control.py` 복구 분기 | `recovery.py`, `control.py`의 `Controller.choose` (V18이 `insufficient_observation`일 때만 실행) |
| 2. 명령 미수신 중계기의 자율 재연결 | `final_v1/transport.py` CommandChannel.apply()의 local 분기 | `transport.py`의 `FallbackChannel` |
| 3. 발동 기준 조정 | `final_v1/control.py` 문턱 비교, `forecast_learning/control.py` GUARDS | `control.py` (`threshold_scale`, `ambiguous_small_moves`). GUARDS는 그대로 쓴다 |
| 4. 급락형 대비 우회 경로 (선택) | `forecast_avoidance/controller.py` fragile_margin() 활용, `final_v1/control.py` 후보 점수 | `bypass.py`, `control.py`의 `_bypass_or` |

## 개선안별 동작

**개선안 1 · 단절 중 복구 재배치** (`recovery.py`)
* 조건: V18이 수색기 비콘 누락으로 판단을 건너뛰었다(`insufficient_observation`).
* 끊긴 수색기 위치: 공유 항법 계획의 예정 위치(`planner.path` → `forecast_navigation.preview`)를 쓴다.
* 위치 불확실성: 끊긴 수색기의 링크 여유를 `0.25dB × 끊긴 초 + 0.20dB × 예측 시점(초)`만큼 낮춘다(V17 Forecast와 같은 계수). 잡음은 측정된 잡음 중 최댓값으로 둔다(측정이 없을 때의 -107dBm 낙관을 막는다).
* 이동 폭 제한: 끊긴 지 6초 이하면 80m, 12초 이하면 57m, 그보다 오래면 40m까지만 보정을 바꾼다.
* 후보: PPO 상위 7개와 기본 대형, 끊긴 수색기 쪽 이동 4개. "끊긴 수색기까지 가장 넓은 경로의 여유"가 반초 두 번 연속 양수가 되는 시각이 가장 이른 후보를 고른다. 시각이 같으면 부족분(dB·s)이 1.5 이상 줄어야 한다. 이미 연결된 수색기끼리의 0.5~2.5초 여유는 떨어뜨리지 않는다.
* 대상: 명령을 낼 첫 중계기와 3초 안에 비콘이 들어온 중계기만 움직인다.
* 기록: 사건 `reason="recovery"`, `expert="cut_recovery"`. 기존 복구 명령 집계에 그대로 잡힌다.
* 복구 근거 (`cut_evidence`, 불필요한 복구 이동 줄이기): `none`(기본) · `model`(보정 없이 계산한 모델이 지금 끊겼다고 볼 때만) · `persistent`(같은 수색기 소식이 직전 판단 주기에도 끊겨 있었을 때만) · `both`. 조정할 수치가 없는 조건만 후보로 두었다. 학습 지도 개발 비행으로 고른 결과 `none`이 가장 좋았다(아래 결과).

**개선안 2 · 명령 미수신 대응** (`transport.py`)
* `neutral`: 기본 대형 복귀 (Final V1 채택)
* `hold`: 직전에 받은 명령을 최대 6초 유지, 그 뒤 기본 대형
* `reconnect`: 마지막으로 명령을 받은 위치에 가장 가까운 보정을 고른다. 6초 안에 다시 받지 못하면 기본 대형
* `local`: Final V1 자율 보정 (비교용)
* 수신 여부 난수는 부모 클래스가 같은 순서로 뽑는다. 대응 방식끼리 같은 수신 표본에서 비교된다(단위 검사로 확인).

**개선안 3 · 발동 기준 조정** (`control.py`)
* 발동 문턱 = V21 문턱(0.1775) × 0.9 / 0.8 / 0.7 / 0.6.
* `ambiguous_small_moves`: 점수가 낮춘 문턱 이상, 원래 문턱 미만인 애매한 구간에서는 보정 변화 28.3m(대각 한 칸) 이하 후보만 허용한다.
* 물리 검사(이득, 초기 3초 안전, 단절 수 증가 금지, 단기 점수)는 V21 GUARDS 그대로 쓴다.

**개선안 4 · 우회 경로 확보** (`bypass.py`)
* 조건: V21이 아무것도 바꾸지 않았고, 지금부터 7초까지 연결이 예측되며, 3~7초 구간에서 링크 하나만 끊겨도 수색망이 갈라지고, 그 링크의 건물 감쇠가 6dB 이상이다.
* 후보: V21과 같은 물리 검사를 통과하고, 링크 하나를 지워도 남는 여유를 3dB 이상 올려야 한다. 보정 변화는 40m 이하, 6초에 한 번까지만 낸다.

## 실행

프로젝트 루트에서 기존 학습 가상환경을 사용한다. 필요한 자료는 다음과 같다.

* V7 PPO: `adaptive_joint_expanded/artifacts/protocol.json`, `adaptive_joint_expanded/artifacts/runs/adaptive_v7_expanded100_20261031/checkpoints/s0000064.pt`
* V21: `forecast_learning/artifacts/`의 `freeze.json`, `protocol.json`, `*_manifest.json`, `models/*.txt`
* 지도: `additional_maps_20261030/expanded_plan.json`과 확인 지도 14개의 `scene.local.json`

`final_v1/artifacts/protocol.json`이 없으면 Final V1과 같은 규칙으로 확인 비행 42개와 명령 손실 14개 조건을 다시 만들고, 재생 HTML에 저장된 56개 조건과 같은지 확인한다. `expanded_plan.json`의 일부 지도 경로는 지도를 만든 PC의 절대 경로로 저장돼 있어 원본 코드로는 그 PC에서만 열린다. 이 패키지는 같은 지도를 작업 폴더에서 이름으로 찾아 쓴다(지도 해시 검사는 그대로, `paths.py`). `freeze.json`의 모델 경로가 다른 PC의 절대 경로이면 같은 이름의 파일을 `forecast_learning/artifacts/models/`에서 찾는다(해시 검사는 그대로). 지도 캐시는 첫 실행 때 자동으로 만들어진다.

자료가 다른 컴퓨터에 있으면 `collect_for_claude.py` 하나를 그 프로젝트 폴더에 복사해 `python collect_for_claude.py`로 실행한다. 위 파일만 골라 버전(해시)을 확인하고, 19MB 조각으로 나눠 `claude_upload/`에 압축한다.

```powershell
# 근거 확인 (시뮬레이터 불필요, 1분 이내)
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m unittest final_v1_relocation.tests.test_relocation -v
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1_relocation.replay_audit

# 짧은 동작 확인: 2개 조건, 40 명령 주기
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1_relocation.campaign --limit 2 --steps 40

# 전체 비교: 확인 비행 42 × 12설정 + 명령 손실 14 × 5설정
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1_relocation.campaign

# 개선안 1만 먼저
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1_relocation.campaign --arms r1_cut_recovery --skip-stress

# 복구 근거 선택 (학습 지도 5곳 × 간섭 3 × 난수 3, 확인 비행과 분리)
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1_relocation.campaign --dev

# 비행 1회
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1_relocation.fly --arm r1_cut_recovery --steps 24
```

루트의 `10_재배치개선_근거확인.bat`, `11_재배치개선_비교실행.bat`으로도 실행할 수 있다. `--controller v21_transformer`를 주면 11월 8일 기본 Transformer 업데이트 위에서 같은 비교를 한다.

결과는 `final_v1_relocation/artifacts/<controller>/`에 `results.json`과 `RESULTS.md`로 저장된다. 중간에 멈춰도 다시 실행하면 끝난 비행은 건너뛴다. 소스가 바뀌면 다시 비행한다.

| 설정 | 내용 |
|---|---|
| `v21` | 적용 전 기준 (모든 개선안 끔) |
| `r1_cut_recovery` | 개선안 1 |
| `r3_t090` ~ `r3_t060` | 개선안 3, 문턱 ×0.9 ~ ×0.6 |
| `r3_t070_small` | 개선안 3, 문턱 ×0.7 + 애매 구간 작은 이동 |
| `r4_bypass` | 개선안 4 |
| `r13_combined` | 개선안 1 + 문턱 ×0.8 작은 이동 (사전에 정한 조합) |
| `r1_model` · `r1_persistent` · `r1_both` | 개선안 1 + 복구 근거 조건 (불필요한 복구 이동 줄이기) |
| 명령 손실 | `v21`×{neutral, hold, reconnect}, `r1_cut_recovery`×{neutral, reconnect} |

RESULTS.md에는 개선안 문서의 확인 지표가 모두 들어간다. 확인 비행 표에는 새 단절/시간, 단절 초/시간, 10초 이상 단절, 긴 단절의 시간 비중, 단절 길이 중앙값과 90%, 복구·예방·우회 명령 수, 중계 이동, 기한 내 전달, 지연 p95, 완주, 안전 위반이 있다. 그 밖에 V21 대비 차이와 공간 그룹 95% 구간, V18 내부 판단 이유 분포, 단절 유형별 수, 명령 손실 조건의 10초 안 안정 복구와 복구 중앙값을 담는다. 채택 판단은 Final V1 승격 조건(안전 위반 0, 전원 완주, 전달 손실 0.25%p 이하, 임무 시간 +2% 이하, 주지표 95% 상한 < 0)을 따른다. 개선안 1의 주지표는 10초 이상 단절/시간이다.

## 재생 기록으로 확인한 근거

`replay_audit.py`가 Final V1 재생 HTML에 저장된 실제 확인 비행 42개를 다시 집계한다([REPLAY_AUDIT.md](REPLAY_AUDIT.md)). 디코딩한 연결 기록으로 210개 비행의 단절 구간을 다시 계산하면 저장된 값과 모두 일치한다.

* V21이 실제로 끊긴 상태에서 내린 판단 306번 중 282번(92.2%)에서 V18이 수색 비콘 누락으로 판단을 건너뛰었다. 10초 이상 단절에서는 166번 중 160번(96.4%)이다.
* 판단 시점이 있었던 단절 133건 중 115건은 모든 판단이 막혔다. 10초 이상 단절은 37건 중 33건이다.
* 차단은 단절 시작 뒤 중앙값 1.5초(90% 2.5초)에 시작된다. 비콘 나이가 3초를 넘는 순간은 1초 이상 단절로 집계되는 시점보다 먼저 온다.
* 명령 손실 14조건에서 무작위 미수신 1,098회 중 1,071회(97.5%)가 6초 안에 끝났다. 강제 손실 구간(41회)은 모두 12초 이상이므로 `reconnect`와 `hold`는 이 구간에서 시간 제한 뒤 기본 대형으로 돌아간다.

* 같은 V7 궤적에 기록된 V21 점수로 경보 성능(127/246, 정밀도 65.46%, 3.15회/시간)이 그대로 재현된다. 3~7초 앞 판단 시점에서 점수가 제어 문턱 0.1775를 넘은 단절은 163/246건이다. 문턱을 ×0.6(0.1065)까지 낮춰도 176건으로 13건만 늘고, 문턱을 넘는 판단 시점은 324→391개로 는다. 문턱 아래 81건 중 68건은 최고 점수가 0.1065 미만이다. 개선안 3의 효과 상한은 작고, 놓친 단절을 줄이려면 예측 점수 자체를 개선해야 한다.

이 결과로 약점 2의 원인 가설(insufficient_observation)은 기록으로 확인된다. 개선안 1이 실제로 단절을 줄이는지는 캠페인을 실행해야 알 수 있다.

## 시뮬레이션 결과 (2026-11-11)

요약은 [SUMMARY.md](SUMMARY.md), 전체 표는 [results/RESULTS.md](results/RESULTS.md), 복구 근거 비교는 [results/DEV_RESULTS.md](results/DEV_RESULTS.md) · [results/DEV_RESULTS_round2.md](results/DEV_RESULTS_round2.md) · [results/EVIDENCE_CONFIRM.md](results/EVIDENCE_CONFIRM.md)에 있다. 개선안을 끈 V21은 42개 비행 모두 Final V1 기록과 같게 재현됐다.

| 개선안 | 결과 | 채택 조건 |
|---|---|---|
| 1. 단절 중 복구 | 10초 이상 단절 37 → 20건, −0.80회/시간 [−1.41, −0.11] | 충족 |
| 2. 미수신 대응 | 마지막 수신 위치 복귀: 10초 안 안정 복구 85% → 77% | 미충족 |
| 3. 문턱 조정 | ×0.6에서 새 단절 166 → 160건, −0.29 [−0.63, +0.00] | 미충족 |
| 4. 우회 경로 | 발동 0회 | 미충족 |
| 1-1. 복구 근거 조건 | 개발 비행 1·2차 모두 원래 개선안 1보다 총 단절 시간이 길었음 | 원래 개선안 1 유지 |

## 한계

* 개선안의 매개변수(불확실성 계수, 이동 폭, 문턱 배율, 차폐 6dB 등)는 사전에 정했다. 확인 비행 결과를 보고 고른 값이 아니다. 확인 비행 결과로 값을 다시 고르면 같은 비행으로 재평가할 수 없다.
* 끊긴 수색기의 예정 위치는 공유 항법 계획 가정에 기댄다. 실제 기체에서는 계획과 실제 위치의 차이가 시뮬레이션보다 클 수 있다.
* 명령 채널은 Final V1과 같은 확률 모델이다. 실제 지연과 채널 점유는 포함하지 않는다.
