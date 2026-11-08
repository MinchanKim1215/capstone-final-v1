# 캡스톤 디자인 II — 새 공동학습 실행 기록

시작: 2026-10-30T22:41:42+09:00. 아래 상태 확인: 2026-10-30T22:46:55+09:00.

- 실행 이름: `adaptive_v6_20261030`.
- 상태: 1회 갱신·8비행 처리 완료. 현재 2번째 갱신의 `joint_ppo` 단계 진행 중.
- 첫 가중치 갱신과 체크포인트 저장 확인. 예측기 인코더·시간 출력층까지 PPO 기울기 전달 확인.
- 첫 8비행의 안전 검사 통과. 체크포인트 SHA256 대조 완료.
- 기본 상한: 256회 갱신 × 8비행 = 2,048비행. 사전 설정된 조기 종료 기준 포함.
- 초기 16회: 네 대형 균등 수집. 이후: 실제 선택기에 따른 대형 선택 및 선택기 PPO 갱신.
- 학습 중 시스템 자동 절전 방지 적용. 화면 꺼짐 허용. 컴퓨터 전원과 실행 환경 유지 필요.
- 최종 평가 지도 32개 봉인 유지. 아직 성능 향상률을 판단하는 단계는 아님.

## 이후 진행 확인

이 문서는 확인 시점의 기록이다. 현재 상태는 다음 자동 저장 파일에 반영된다.

- `artifacts/runs/adaptive_v6_20261030/status.json`: 현재 수집·학습·종료 상태.
- `artifacts/runs/adaptive_v6_20261030/progress.json`: 저장 완료한 갱신·비행 수.
- `artifacts/runs/adaptive_v6_20261030/history.json`: 갱신별 학습 기록.
- `artifacts/runs/adaptive_v6_20261030/validation_history.json`: 정기 개발 평가 후 생성되는 기록.
- `artifacts/runs/adaptive_v6_20261030/best.json`: 검증 기준을 만족한 best가 있을 때 생성.
- `artifacts/launcher/status.json`: 실행기 상태 및 전체 출력·오류 로그 위치.
- `artifacts/launch_audit.json`: 본학습 시작 확인 근거.

첫 저장 체크포인트: `artifacts/runs/adaptive_v6_20261030/checkpoints/s0000001.pt`.
SHA256: `57e347703342215afce7904ae7ef6ef5c904ca5d744dc6e2d716e52d2e55e076`.
