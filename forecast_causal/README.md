# 캡스톤 예측 개선 V16

사용자 요청의 네 단계: 장애 원인 분해 → 실제 관측 보강 → 물리·LightGBM·GRU 대조 → 같은 기준의 새 비행 검증.

진행 기록: 프로젝트 최상위 `PROJECT_FOLLOWUP.md`, `artifacts/status.json`, `artifacts/milestones.json`, `WORKLOG.md`.

## 코드와 데이터

- `recorder.py`, `instrumented_packet.py`, `replay.py`: 원래 궤적·RF·비컨·패킷 결과를 보존하는 계측. 장애 진단 정답과 관측 센서를 별도로 저장.
- `causes.py`: 패킷250ms 마감 이전의 다중 증거 분류. 건물·간섭 제거는 순간 연결 대조이며 배타적 원인 확정이 아님.
- `wire.py`, `state.py`, `features.py`: 로컬 관측과 실제 수신128B 비컨에서만 새 패킷 상태 구성. 예약 슬롯18B 사용; 원본 내용 정확히 복원.
- `path_uncertainty.py`, `physics.py`: 훈련 자료로만 경로 오차 크기를 추정. 실제 수신 경로와 공개 지도에서 미래 위치 시나리오 특징 생성.
- `tree_train.py`, `neural_train.py`: 고정한 예산의 입력 추가 대조. `compare.py`:12후보·보정 문턱 고정 후 새42비행 평가.
- `runtime.py`: 실제 순차 실행 API. `observe(..., local_network=...)`에는 해당 수신기 자신의27개 현재 센서값만 제공. 다른 노드의 완전한 상태·미래 정보·진단 정답을 주지 않음.
- `post.py`: 실제 API, 무결성, 수치 보고, 도표 생성. `check_delivery.py`: 실제 도표 검토까지 마친 뒤 완료 처리.

## 실행

프로젝트 루트의 `.\drone_forecast_joint_training\.venv\Scripts\python.exe -u -B -m forecast_causal.status`로 현재 상태 확인.

전체 작업은 `forecast_causal.campaign`이 단계별 별도 로그를 기록한다. 오류가 있으면 원인을 먼저 확인하고 완료된 단계의 불변 계약을 확인한 뒤 재개한다. 성공한 학습이나 새 확인 결과를 임의로 덮어쓰지 않는다.

납품 봉인 후 `campaign`·`post`를 다시 실행하면 해시를 확인하고 종료한다. 개별 훈련·자료 생성·보고서 생성 모듈을 직접 다시 실행하지 않는다. 후속 실험은 새 폴더와 새 확인 자료로 구성한다.

## 평가 조건

첫 중계드론 서비스 장애,3≤선행시간<7초, 동일 입력 자격·히스테리시스·최초 경보·전체 사건 분모. 사용자 우선순위는 탐지율 증가, 정밀도 유지, 오경보 약4회/드론·시간. 주FA4 및 보조FA3,5의 문턱은 확인 전에 고정한다.

기존V12~V15 납품본·원래 코드와 가중치·현재V13 프로필 보존. 기존PPO는 자료 생성 정책으로 고정한다. 원래 최종32지도는 열지 않는다. 새42비행은 알려진14지도의 새 난수 검증이며 실제 무선장비·미지 지형 성능 인증이 아니다.
