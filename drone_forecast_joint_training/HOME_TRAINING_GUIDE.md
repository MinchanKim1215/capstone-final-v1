# 단독 Transformer + 공동 PPO 학습 안내

2026-10-29 · v5 · Python 3.12 64-bit · PyTorch 2.9.1

## 먼저 읽기

이 버전의 **유일한 학습형 단절 예측기는 Transformer**입니다. GRU는 사용하지 않습니다. 기존 LightGBM 점수나 잔차를 Transformer 입력으로 넣지 않습니다. 물리·지도 기반 후보 생성, 충돌 회피, 패킷 시뮬레이터는 유지합니다. LightGBM 패키지는 이전 공통 모듈의 import와 선택적 비교 실험 때문에 설치되지만, 기본 학습·추론에서 트리 가중치를 로드하거나 예측하지 않습니다.

공동학습은 예측기만 따로 학습한다는 뜻이 아닙니다. Transformer를 먼저 지도학습하고, 그 예측으로 재배치를 결정하는 PPO 단계에서 **정책 보상과 단절 정답 손실을 함께 사용**합니다. PPO 보상이 Transformer 인코더와 확률 출력층 모두에 전달되는지 코드와 실제 업데이트에서 검사합니다.

제공 체크포인트는 학습·재개·연결 검증을 위한 파일입니다. 장시간 학습 완료, 실제 비행 승인, Windows/CUDA 실행 완료를 뜻하지 않습니다. 이번 환경에서 실행한 범위와 수치는 `TRAINING_REPORT.md`를 보세요.

## 1. Windows 준비

1. ZIP을 완전히 풀고 `C:\capstone_transformer\drone_forecast_joint_training` 같은 짧은 경로에 둡니다. ZIP 내부에서 배치 파일을 실행하지 마세요. 이전 GRU 프로젝트 폴더에 덮어쓰지 마세요.
2. Python **3.12 64-bit**와 Windows Python Launcher를 설치합니다. 터미널에서 `py -3.12 --version`을 확인합니다.
3. NVIDIA 드라이버가 설치된 PC에서 `nvidia-smi`로 GPU를 확인합니다. 여기에 표시되는 CUDA 버전만으로 PyTorch의 GPU 실행 가능 여부가 확인되지는 않습니다.
4. 권장 여유 공간은 20GB 이상, 메모리는 16GB 이상입니다. 32GB RAM이면 worker 4부터 시작하기 편합니다. 자원이 부족하면 worker 2를 사용하세요. 이는 여유를 둔 운영 권장값이며 모든 PC를 실측한 요구 사양은 아닙니다.

프로젝트 폴더에서 다음을 실행합니다.

```bat
01_setup_nvidia.bat
```

이 명령은 전용 `.venv`를 만들고 `torch==2.9.1+cu128` 공식 빌드와 고정 의존성을 설치한 뒤, 데이터 검사·CPU 테스트·**실제 CUDA Transformer/PPO 순전파, 역전파, 가중치 업데이트**를 실행합니다. CUDA 검사가 실패하면 학습을 시작하지 않습니다. CPU 빌드가 먼저 설치돼 있어도 정확한 CUDA 빌드를 선택합니다. 별도 CUDA Toolkit 설치 여부보다 설치된 PyTorch 빌드와 드라이버·GPU의 호환성이 중요합니다.

성공 결과는 `results/transformer/verification/cuda-check.json`에 저장됩니다. 이 파일에서 `passed`, `predictor_updated`, 두 policy gradient 값, `cpu_target_logp_max_error`, `checkpoint_resume_bitwise_equal`, GPU 이름을 확인하세요. `cuda_available: true`만 확인하는 방식이 아닙니다.

CPU만 쓰려면 `00_setup_cpu.bat`을 실행합니다. NVIDIA용 학습 배치 파일은 `--device cuda`를 강제하므로 CPU에서는 아래의 직접 명령을 쓰세요.

## 2. 권장: 데이터 확대 후 본 학습

```bat
02_train_transformer.bat 256 4
```

첫 숫자 **256은 누적 목표 업데이트 수**, 둘째 숫자 4는 시뮬레이션 worker 수입니다. 256번을 추가한다는 뜻이 아닙니다. 아래 순서로 진행합니다.

1. 대상 PC에서 CUDA 역전파 확인.
2. 훈련·검증 190개 맵의 두 방향 기본 경로 검사. 기존 유효 캐시를 재사용합니다.
3. 훈련 데이터 **192회 비행** 수집 및 고정된 검증 8회 확보. 제공된 부트스트랩 데이터는 재사용하고 부족한 데이터만 추가합니다.
4. Transformer 지도학습 **누적 3,000 step**. 동일 배치 설정으로 이어서 학습하며, 검증에서 선택한 best 체크포인트를 사용합니다.
5. 예측기가 검증의 상수 점수 AP 기준을 넘는지 최소 점검. 이 검사는 실사용 품질 인증이 아닙니다.
6. 새 실행 `transformer_main`에서 공동 PPO **누적 256 update**. 기본 한 update는 전체 비행 8회, 최대 PPO 4 epoch입니다. KL 제한 때문에 epoch가 일찍 끝날 수 있습니다. 256 update 전부 실행하면 훈련 비행은 2,048회입니다.
7. 16 update마다 고정 검증 8회. 8번의 연속 검증에서 개선이 없으면 중단하고 best 모델을 남깁니다. 따라서 목표 update 전 조기 종료될 수 있습니다.
8. 종료 후 검증 **24개 맵 × 4조건 = 96회**를 기준선과 선택 모델로 실행하고 최종 검증 선택을 기록합니다. 최종 테스트 32개 맵은 이 단계에서 열지 않습니다.

PC가 버거우면 다음처럼 worker만 줄이세요. 재개할 때 worker나 CPU/CUDA 장치 변경은 허용되지만, 같은 장치에서의 비트 단위 재현성을 다른 장치까지 보장하지는 않습니다.

```bat
02_train_transformer.bat 256 2
```

GPU는 신경망 학습을 맡고, RF·비행 시뮬레이션 worker는 CPU에서 실행됩니다. GPU 사용률이 낮아도 수집 단계가 진행 중일 수 있습니다. 학습 시간은 GPU뿐 아니라 CPU, 비행 길이, worker 수, 검증 빈도에 좌우됩니다. 이 환경의 CPU 소규모 시간은 보고서에 있으며, 사용자 GPU의 총 소요 시간으로 환산하지 않았습니다. 기본 192회 사전 수집은 첫 166개 훈련 지역을 모두 한 번 이상 포함하지만, 각 맵의 여섯 방향·재머 조합을 모두 포함하지는 않습니다.

## 3. 빠르게 제공 모델을 이어 학습하는 방법

```bat
03_continue_verified.bat 16 4
```

이 명령은 제공된 `verified_joint` 모델·Adam 상태·난수 상태에서 이어갑니다. 부트스트랩을 192회로 확대하거나 Transformer를 3,000 step까지 사전학습하는 절차는 실행하지 않습니다. **최종 학습에는 2번 경로를 권장합니다.** 두 배치 파일은 서로 다른 실행 이름을 쓰므로 혼동하지 마세요. 제공 pilot은 한 update당 비행 4회·PPO 최대 2 epoch·매 update 검증 8회·조기 중단 비활성 설정입니다. 03번의 기본 누적 목표는 16이며, 02번의 본 학습 설정과 다릅니다.

`verified_joint`는 최초 생성 때 사용한 사전학습 파일을 SHA로 고정합니다. 이후 별도로 예측기를 더 학습해도 이미 시작한 PPO 실행에 새 가중치를 몰래 주입하지 않습니다. 새 예측기를 이용하려면 `transformer_main` 같은 **새 run 이름**으로 시작해야 합니다.

## 4. 중단과 재개

콘솔에서 `Ctrl+C`를 한 번 누르고 종료를 기다립니다. 실행 중인 worker가 안전하게 멈추는 데 시간이 걸릴 수 있습니다. 전원 차단 시에도 마지막으로 완료한 checkpoint를 기준으로 재개합니다.

**처음 실행한 것과 같은 명령을 다시 실행하세요.** 별도 `--resume` 옵션은 없습니다. 모델, Adam, 난수, 누적 update를 읽습니다. 완료 비행은 요청·체크포인트·코드·데이터 SHA가 모두 맞을 때만 재사용합니다. 중단된 비행은 처음부터 다시 수집합니다. 지도학습은 마지막 저장 간격 이후의 미저장 step을 다시 실행할 수 있습니다. PPO는 한 update가 모두 끝나야 commit합니다.

다음 파일을 사람이 편집하거나 최신 포인터를 임의로 바꾸지 마세요.

- `configs/transformer_plan.json`
- `solo/`의 핵심 코드와 `transformer_run.py`
- `model/`의 원본 가중치 및 참조된 사전학습 checkpoint
- `results/transformer/**/latest.json`, `best.json`, `checkpoints/*.pt`

해시 검사는 수정·누락·다른 데이터 혼합을 발견하면 중단하도록 설계했습니다. 해시를 현재 파일 값으로 억지 수정해 우회하면 실험 재현성이 깨집니다. 소스나 데이터 분할을 바꿔 연구하려면 보관한 복사본에서 새 실험을 시작하세요. 동일 run 폴더에 두 학습 프로세스를 동시에 띄울 수 없습니다.

## 5. 세부 명령

PowerShell에서는 `.venv\Scripts\python.exe` 앞에 `&`를 붙여도 됩니다. 아래는 프로젝트 폴더의 cmd 기준입니다.

```bat
.venv\Scripts\python.exe transformer_run.py doctor
.venv\Scripts\python.exe transformer_run.py selftest
.venv\Scripts\python.exe transformer_run.py cuda-check
.venv\Scripts\python.exe transformer_run.py collect --episodes 192 --workers 4
.venv\Scripts\python.exe transformer_run.py pretrain --steps 3000 --device cuda
.venv\Scripts\python.exe transformer_run.py train --run transformer_main --mode joint --updates 256 --workers 4 --device cuda
.venv\Scripts\python.exe transformer_run.py select --runs transformer_main --workers 4
```

`pretrain --steps`와 `train --updates`는 모두 누적 목표입니다. 예측기 사전학습의 기본 배치는 256개 링크 시퀀스, 검증 간격은 200 step, plateau patience는 8입니다. 재개할 때 저장된 batch/seed/validation/patience를 바꿀 수 없습니다. PPO의 batch·epoch·학습률도 같은 run 안에서는 바꿀 수 없습니다. 필요하면 새 run 이름으로 시작합니다.

CPU 경로는 예를 들어 다음과 같습니다.

```bat
.venv\Scripts\python.exe transformer_run.py pipeline --run transformer_main --episodes 192 --pretrain-steps 3000 --updates 256 --workers 2 --device cpu
```

Linux에서는 `python3.12 setup_environment.py --torch cpu` 후 `.venv/bin/python transformer_run.py ... --device cpu`를 사용합니다. Linux NVIDIA는 `--torch cu128`과 `--device cuda`입니다. Linux CPU 실행만 이번 환경에서 검증했습니다.

### 예측기를 고정하는 대조 실험

```bat
.venv\Scripts\python.exe transformer_run.py train --run fixed_full --mode fixed --updates 256 --workers 4 --device cuda
.venv\Scripts\python.exe transformer_run.py select --runs transformer_main fixed_full --workers 4
```

두 실행을 **동일한 best 사전학습 파일에서 새로** 시작해야 비교가 공정합니다. `fixed`는 Transformer 가중치를 고정하고 정책을 학습합니다. `joint`는 Transformer까지 함께 업데이트합니다. `physics`는 학습형 예측 없이 물리 후보를 쓰는 기준선입니다. 선택은 검증 연결률·패킷 전달률·단절 길이와 안전·비행 시간 조건을 봅니다. 학습 누적 보상 크기만으로 선택하지 않습니다.

LightGBM은 선택적 비교 전용입니다.

```bat
.venv\Scripts\python.exe transformer_run.py benchmark-lightgbm
```

이 명령은 동일한 검증 관측·시점·정답·유효 표본으로 LightGBM과 사전학습 Transformer를 비교합니다. 기본 policy에 LightGBM을 연결하지 않습니다.

## 6. 지표와 파일

| 확인할 내용 | 경로 |
|---|---|
| 설치·장치·데이터 | `results/transformer/verification/doctor.json` |
| 실제 GPU 역전파 | `results/transformer/verification/cuda-check.json` |
| 지도학습 최근 저장 / best | `results/transformer/pretrain/latest.json`, `best.json` |
| PPO 진행 | `results/transformer/train/transformer_main/progress.json`, `history.json` |
| PPO 검증 best / 이력 | 해당 run의 `best.json`, `validation_history.json` |
| 검증 선택과 경보 규칙 | `results/transformer/frozen_selection.json` |
| 같은 관측에서의 예측기 비교 | `results/transformer/common_corpus_validation.json` |
| LightGBM 비교 | `results/transformer/lightgbm_reference.json` |
| 실제 비행 로그와 독립 감사 | 각 rollout의 `.npz`와 `.json` |

- **CTR**: 탐색 드론의 연결 상태 비율. 예측·재배치 정책을 실행한 뒤 실제 시뮬레이션 결과에서 계산합니다.
- **PDR / timely PDR**: 생성 패킷의 전달률 / 지연을 반영한 전달률. 연결률과 같지 않습니다.
- **AP**: 희귀 단절 분류의 순위 성능. 상수 점수의 AP는 양성 비율이며, 다른 맵·표본의 AP를 직접 우열 비교하지 마세요.
- **Brier**: 확률 오차. 낮을수록 좋습니다. AP 상승이 확률 보정 향상을 보장하지 않습니다.
- **최초 경보 recall ≥3s**: 최소 1초 지속되는 단절 사건에 대해, 최초 경보가 단절 3초 이상 전에 발생한 비율. 단순 분류 recall과 다릅니다.

여기서 사건 기반 **5초 경보는 `onset_3to4s` head**, 8초 경보는 `onset_3to7s` head를 사용합니다. `risk_5s` 분류 수치와 같은 지표가 아닙니다.

경보는 검증에서 precision ≥80%, 오경보 ≤5회/드론·시간, 최초 경보 ≥10건의 조건을 만족하는 규칙이 있을 때만 활성화합니다. 조건을 만족하지 못하면 `enabled: false`입니다. 감소한 8회 pilot 검증은 경보 활성화·최종 테스트에 사용할 수 없습니다.

## 7. 최종 테스트는 개발을 마친 뒤 한 번

전체 검증과 모델 선택이 끝난 뒤에만 다음을 실행합니다.

```bat
05_final_holdout_test.bat
```

32개 미사용 맵 × 4조건 = **128회**에서 기준선과 선택 모델을 평가합니다. 테스트 접근 표시가 생기면 추가 학습은 차단됩니다. 중단 시 같은 선택과 같은 명령으로 재개할 수 있지만, 결과를 본 뒤 모델을 바꾸고 같은 맵을 다시 미사용 테스트라고 부를 수 없습니다. 계속 개발하려면 새로운 미사용 지역을 따로 확보해야 합니다.

## 8. 문제 해결

- `py` 또는 Python 3.12를 못 찾음: Python 3.12 64-bit와 Launcher 설치 후 새 터미널에서 재시도.
- CUDA unavailable / CUDA backward 오류: `01_setup_nvidia.bat`의 최초 오류와 `nvidia-smi`를 확인. CPU로 조용히 전환하지 않으므로 CUDA 문제가 드러납니다. CPU로 실행하려면 명시적으로 CPU 경로 선택.
- RAM 부족·PC 멈춤: worker를 2로 줄이고 다른 프로그램 종료. GPU OOM이면 같은 run의 minibatch를 임의 변경하지 말고 새 run을 작은 minibatch로 시작. 기본 minibatch는 16입니다.
- 해시·resume configuration 오류: 다른 버전·코드·가중치를 섞었는지 확인. 해시 우회 금지. 원래 ZIP의 별도 복사본과 로그를 보관.
- 오래 걸림: `stage: train`은 전체 비행 수집, `committed_update`는 실제 PPO 저장 완료. worker 수를 늘리면 RAM/CPU 부담도 증가합니다.
- 저장 공간 부족: 최신 2 update의 원시 rollout만 자동 유지하지만 checkpoint와 요약은 남깁니다. 원시 학습 로그를 직접 지울 때는 진행 중인 update 및 부트스트랩 데이터를 지우지 마세요.

## 9. 실기체 적용 범위

`solo/deploy.py`의 `StreamingController`는 .5초 공개 RF 관측을 누적하고 3초마다 행동과 확률을 반환합니다. 관측 40개 이전 또는 현재 연결이 확인되지 않은 링크는 예측 지원 대상에서 제외합니다. 별도 `AlarmMonitor`가 전체 검증으로 고정된 경보 규칙을 적용합니다. raw 추론 API가 자동으로 경보를 활성화하지는 않습니다.

실제 무선 장비 입력을 이 관측 계약으로 변환하는 드라이버, 실시간 명령 전달, 실측 RF 보정, 위치 오차·풍속·통신 지연 검증은 별도의 통합 단계입니다. 시뮬레이션 통과를 실비행 안전성 보장으로 해석하지 마세요.

공식 참고: [PyTorch 2.9.1 설치](https://pytorch.org/get-started/previous-versions/#v291), [TransformerEncoderLayer](https://docs.pytorch.org/docs/2.9/generated/torch.nn.TransformerEncoderLayer.html).
