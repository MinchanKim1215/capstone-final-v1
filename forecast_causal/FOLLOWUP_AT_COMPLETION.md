# 캡스톤 예측 개선 — 지속 작업 기록

갱신: 2026-11-03T19:29:26+09:00

사용자 요청: 네 단계를 빠뜨리지 말고 순서대로 진행. 진행 상황·근거·다음 행동을 계속 기록.

**운영 우선순위: 선행 탐지율 향상, 오경보 약4회/드론·시간, 정밀도 유지. 어려운 사건도 평가 분모에 포함.**

## 네 단계

- [x] 1. 놓친 장애의 원인 분해
  - 완료: 2026-11-03T17:48:13+09:00
  - 근거: STAGE1_CAUSES.md 및 artifacts/selection_event_causes.json; 기존200건·134건 집계 일치
- [x] 2. 원인에 맞는 관측 정보 보강
  - 완료: 2026-11-03T18:33:05+09:00
  - 근거: 335개 입력·정답 일치, 실제 수신 순서·관측 불가/미래값 교란 불변: artifacts/input_preflight.json
- [x] 3. 원인별 예측기 구성·대조 학습
  - 완료: 2026-11-03T18:57:56+09:00
  - 근거: artifacts/trained_models.json·selection_decision.json·freeze.json; 12개 후보·주FA4/보조FA3,5문턱을 확인 전 고정
- [x] 4. 같은 기준의 신규 비행 검증·채택 판단
  - 완료: 2026-11-03T19:29:26+09:00
  - 근거: RESULTS_20261103.md·final_results.json·runtime_checks.json·audit.json·delivery_checks.json; 새42비행·실제API·377비행 보존·도표 확인 완료

## 현재 상태

- 상태: completed
- 단계: all_four_stages_training_new_confirmation_runtime_audit_report_completed

## 다음 행동

후속 우선순위: 물리·발행 경로 예측을 정교화하고 점수 미달123건·이미 켜진 경보13건·간섭 그룹5/29건을 분석한다. 새 큐 입력·GRU의 추가 효과는 확인되지 않았다. 현재 프로필은 보존하며 다음 실험도 새 확인 자료·동일 경보 비용으로 검증한다.

## 유지할 조건

- 원본 코드·모델 및V8~V15 납품본 보존. 현재 탐지 우선V13 프로필 유지.
- 메인PPO 정책은 자료 생성 기준으로 고정. 예측 개선과 재배치 성과를 구분.
- 원래 최종32지도 미사용. 이전 확인 비행을 새 모델 학습·선택에 사용하지 않음.
- 실제 관측값과 실제 수신 패킷만 예측 입력. 전파 정답·간섭 일정·미래패킷은 분석·정답 전용.
- 기존 사건 정의·3≤선행시간<7초·최초 경보·히스테리시스·전체 사건 분모 유지.
- 훈련/모델 선택/확률·문턱 보정/새 확인을 분리. 결과를 보고 유리한 모델로 다시 선택하지 않음.
- 실행 시작은 완료가 아님. 실패하면 오류와 재개 지점을 기록.

## 이번 작업 위치

- `forecast_causal/`: 이번 단계별 코드·분석·학습.
- `forecast_causal/artifacts/protocol.json`: 사용자 요청과 보호 조건.
- `forecast_causal/artifacts/status.json`: 현재 단계·갱신 시각.
- `forecast_causal/artifacts/milestones.json`: 완료 근거와 다음 행동.
- `forecast_causal/WORKLOG.md`: 기술 인계 기록.
- `forecast_peer/RESULTS_20261103.md`: 이전12모델의 비교 결과. 새 개선 결과와 구분.
