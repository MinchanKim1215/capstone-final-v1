# 사건 진단·패킷 서비스 예측 실험 v10

기존 코드·가중치·보고서와 최종 지도32개를 보존하는 별도 실험이다. 사용자 승인에 따라 사건 진단, 정답·입력 구성, 한 변수씩 바꾸는 대조 학습, 새 확인 비행 평가를 수행한다.

**2026-11-02 완료.** 네 단계와 최종 보존·정답·실시간 검사를 통과했다. 개발 경보 목표는 두 과제 모두 미충족이다. [결과 보고서](RESULTS_20261102.md)를 따른다. 완료된 동결 실험의 가중치·코드·결과를 덮어쓰지 않는다.

## 상태와 산출물

프로젝트 루트 `02_사건분석_서비스예측_현황.bat` 또는 `python -B -m forecast_service.status`로 확인한다.

- `DIAGNOSTICS.md`: 선택90비행의 기존 고정 모델 실패 유형.
- `artifacts/diagnostics/timelines/INDEX.md`: 사건별 미탐·오경보 타임라인.
- `artifacts/diagnostics/physics_components.json`: 선택 자료 표집 사건의 전파 손실·섀도잉·간섭 항 분해. 모델·문턱 변경에는 사용하지 않는다.
- `artifacts/experiment_plan.json`: 학습 전에 고정한 비교 계획.
- `artifacts/freeze.json`: 확인 전에 고정한 후보·보정·구현.
- `artifacts/confirmation_dataset.json`: 실제 확인42비행 목록. 중단 당시의 부분 수집 목록과 구분한다.
- `artifacts/recovery/packet_log_recovery.json`: 같은42비행의 누락된 패킷 기록 복원 이력.
- `RESULTS_20261102.md`, `artifacts/final_results.json`: 최종 검사 후 작성되는 과제별 결과.
- `artifacts/audit.json`, `runtime_checks.json`: 보존·정답·추론·실시간 검사.

## 두 과제의 구분

**기존 링크 과제:** 기존1,123,200표본·정답·경보 정의를 유지한다. 동일 기본 모델과 사건 중심 가중치를 두 난수로 비교한다.

**새 서비스 과제:** 수색군 패킷의250ms 내 전달률이 최근1초에서90% 미만인 상태가1초 이상 지속되는 시작을 예측한다. 지정 중계드론1대의 수신한 과거 관측만 입력한다. 연구 가정이며 실제 장비의 서비스 요구사항은 아니다. 기존 링크 과제와 AP·재현율을 직접 차감하지 않는다.

정답 계산의 지연 경계에는 처음부터1e-10초 수치 허용오차가 있다. 기존 엔진의 적시 전달 누계는 엄격한 `delay<=0.25`이므로 별도로 검사한다. 이 조건을 혼용한 기록 검사 오류는 원본 비행·모델·정답 계산식·문턱을 유지한 정확 재생으로 복구한다.

## 연구용 로더

최종 검사 후 `artifacts/research_candidates.json`의 정확한 후보를 사용한다.

```python
from forecast_service.runtime_aligned import load_link, load_service

link = load_link(city, relay_id, n_drones, n_search, nominal=nominal_xy)
service = load_service(city, n_search, n_drones, n_search, nominal=nominal_xy)
```

서비스 API는 지정된 첫 중계드론만 평가했다. 입력은 매0.5초의 로컬 `[N,48]`값·마스크, `[N,9]`수신/자기 명령, 선택적 `[N,6]`에너지다.20초 이력 뒤1초마다 확률을 출력하고, 원래 평가와 맞춘 경보 시작은21초다. 이 시작 시각·공개 무선 설정 어댑터는 동결 원본을 변경하지 않는다.

`None`을0위험으로 바꾸지 않는다. 새 서비스 출력은 향후 새로운 서비스 악화의 시작 위험이다. 현재 서비스가 정상이라는 판정이 아니다. 실제 장비 검증 전 자동 경보와 PPO 교체는 비활성이다.

모든 후보는 기존14개 지도에서 새 난수의42비행으로 확인한다. 새로운 지형 일반화 성능이나 실제 통신모듈 성능으로 표현하지 않는다. 원래 최종 지도32개는 계속 봉인한다.
