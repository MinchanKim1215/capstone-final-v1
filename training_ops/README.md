# 로컬 학습 실행 기록

2026-10-29 한국 시간에 준비하고 실행한 v5 본 학습입니다. 아래는 실행 설정이며, 실시간 진행 상태는 `status.json`과 로그를 확인하세요.

## 실행 환경과 검사

- Windows 11, AMD Ryzen 5 7600 (6코어 / 12스레드), 메모리 약 32GB.
- NVIDIA GeForce RTX 4070 SUPER, VRAM 12GB, 드라이버 610.88.
- 전용 가상환경: `../drone_forecast_joint_training/.venv`.
- Python 3.12.14 (앱에 포함된 런타임 기반), PyTorch 2.9.1+cu128.
- 원본 ZIP SHA256: `59f7432751ed02cae2dd11988626d5cdb6056a6ab7908cebb7e07d9df0d231f9`.
- 압축 해제 직후 전체 배포 무결성 검사 2,381항목 통과.
- doctor 통과, 테스트 24개 통과, pip 의존성 충돌 없음.
- 실제 CUDA 역전파, encoder와 확률 head의 정책 gradient, 가중치 갱신 통과.
- CPU/CUDA log-probability 최대 차이: 2.384185791015625e-7.
- CUDA checkpoint/Adam 복구 후 중단 없는 실행과 비트 단위 동일 확인.
- 학습 실행 환경에 `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` 적용.
- AC 전원의 자동 절전은 확인 당시 0초(사용 안 함). 전원 설정 변경 없음.

검사 JSON은 프로젝트의 `results/transformer/verification/`에 있습니다.
하드웨어와 설치 패키지 기록은 `machine.json`, `installed_packages.txt`에 있습니다.

## 실행 내용

프로젝트 폴더에서 다음 명령을 실행하는 별도 로컬 프로세스를 시작했습니다.

```powershell
.venv\Scripts\python.exe -u transformer_run.py pipeline --run transformer_main --episodes 192 --pretrain-steps 3000 --updates 256 --workers 4 --device cuda
```

순서: 기본 경로 점검 → 훈련 데이터 누적 192회 수집 → Transformer 누적 3,000 step → 공동 PPO 누적 256 update → 전체 검증 선택.
기존 유효 데이터와 체크포인트는 재사용합니다. 조기 종료 조건에 따라 목표 update 전에 끝날 수 있습니다.
최종 holdout 테스트 명령은 실행하지 않습니다.

## 진행 확인과 재개

- `check_status.bat`를 실행하면 프로세스 상태와 최근 로그를 표시합니다.
- `status.json`에는 시작 시각, 프로세스 ID, 실행 명령, 로그 경로, 실행 결과가 기록됩니다.
- `logs/날짜_시각.stdout.log`는 진행 로그, 같은 이름의 `.stderr.log`는 진단 로그입니다.
- 데이터 수집 중에는 CPU가 주로 일하며 GPU 사용률이 낮을 수 있습니다.
- 지도학습 저장 지점: 프로젝트 `results/transformer/pretrain/latest.json`.
- PPO 저장 지점: 프로젝트 `results/transformer/train/transformer_main/latest.json` 및 `progress.json`.
- 중단 또는 재부팅 후 `resume_training.bat`를 실행하면 같은 설정으로 마지막 저장 지점부터 재개합니다. 실행 중에는 다시 시작할 필요가 없습니다.
- 재부팅 후 자동 시작이나 정기 알림은 설정하지 않았습니다. 이 감독 프로세스는 종료 결과를 파일에 남기며 오류 발생 시 무한 재시작하지 않습니다.
- 컴퓨터 전원과 사용자 로그인을 유지하세요. 로그와 체크포인트가 별도 프로세스에 의해 계속 기록됩니다.

원본 학습 소스·보상·데이터 분할·제공 체크포인트는 수정하지 않았습니다. 이 폴더의 실행 관리 도구는 프로젝트 밖에 두었습니다.
