# Capstone v5 — 단독 Transformer 예측 + 공동 PPO 재배치

유일한 학습형 단절 예측기는 40개 RF 관측을 읽는 작은 Transformer입니다. GRU와 LightGBM 확률 입력을 제거했습니다. 알려진 지도·비행 의도를 이용하는 물리 후보 생성과 기존 비행·충돌·RF·패킷 시뮬레이션은 유지합니다.

**Windows NVIDIA 시작:** ZIP을 새 폴더에 풀고 `01_setup_nvidia.bat`, 성공 후 `02_train_transformer.bat 256 4` 실행. 중단 후 같은 명령으로 재개합니다.

- [상세 학습 안내](HOME_TRAINING_GUIDE.md): 설치, CPU/CUDA 검사, 전체 학습과 제공 모델 재개, 지표, 문제 해결.
- [검증 결과](TRAINING_REPORT.md): 이번 환경에서 실제 실행한 양·성능·제한.
- [시스템 계약](docs/ARCHITECTURE.md): 입력, 미래 정보 차단, PPO 결합, 경보와 실기체 연동.
- `solo/`: v5 학습·예측·추론 구현.
- `transformer_run.py`: 통합 CLI.
- `tests/`: 누수·gradient·원자적 저장·재개·경보 검사.

`02_train_transformer.bat`은 데이터 192회·지도학습 3,000 step으로 확대한 뒤 **새 `transformer_main`**을 학습합니다. `03_continue_verified.bat`은 제공된 소규모 검증 run만 이어갑니다. 둘은 목적이 다릅니다.

v4 GRU 체크포인트와 v5 Transformer는 호환되지 않습니다. `docs/v4_results.json`과 `docs/prediction_alarm_metrics.*`는 과거 기록이며 v5 성능이 아닙니다. `cooperative/`, `swarm_sim/`, `usable/`, `advance/`, `upgrade/`는 보존된 기반 코드입니다. 이전 진입점 대신 `transformer_run.py`를 사용하세요.

학습 코드 연결과 시뮬레이션 결과를 검증한 패키지이며, 장시간 학습의 성능 향상이나 실비행을 보증하지 않습니다. 자세한 실측 범위는 보고서에 기록합니다.
