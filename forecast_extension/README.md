# 예측 학습 연장·새 경험 추가 실험 v9

2026-11-02. 사용자 승인에 따른 세 단계의 학습·확인·감사를 모두 완료했다. 기존 v8 코드·가중치·보고서와 봉인된 최종 평가 지도32개를 보존했다. 개발 경보 목표에는 미달했다. 결과와 범위는 [최종 보고서](RESULTS_20261102.md)를 따른다.

1. 같은5,000회 Transformer 체크포인트에서20,000회까지 연장한다.
2. 새 비행377개와 일관된 위치·센서 오차 관측을 추가한다. 새 자료 Transformer는 같은 시작점·같은 추가15,000회 업데이트로 비교한다. 새 자료 트리도 에너지 사용 여부를 나누어 학습한다.
3. 정밀도·선행 사건 재현율·오경보를 함께 비교한다. 새 확인 비행은 후보·문턱·코드를 고정한 뒤 평가한다.

## 현재 상태 확인

프로젝트 루트의 `01_예측연장_현황.bat`를 실행한다. 기존 `00_예측개선_현황.bat`는 이전 v8 결과다.

```powershell
$env:PYTHONIOENCODING='utf-8'
.\drone_forecast_joint_training\.venv\Scripts\python.exe -B -m forecast_extension.status
```

정확한 상태 파일은 `artifacts/status.json`, `collection_status.json`, `stream_preparation_status.json`, 각모델 `status.json`, `post_status.json`이다.

## 완료되는 결과

- [최종 비교 보고서](RESULTS_20261102.md): 세 단계와 새 비행 성적·한계. 실험 종료 후 생성한다.
- `artifacts/final_results.json`: 원시 AP·경보·보정·같은FA예산·공간 그룹 재표집.
- `artifacts/freeze.json`: 확인 전에 고정한 모든 후보·보정·자료·구현 해시.
- `artifacts/audit.json`: 원본 보존·표본 역할·증강 정답 보존·캐시 재추론 감사.
- `artifacts/runtime_checks.json`: 전체 한 비행의 배치/실시간 출력 대조.
- `artifacts/research_candidate.json`: 확인 전에 선택한 연구 후보. 실제 운용 자동 경보 비활성.
- `artifacts/bundle_checks.json`: 실제 관측80프레임으로 내보낸 로더와 세 운영점을 검사한 결과.
- `artifacts/delivery_manifest.json`: 검수한 보고서·그림·모델·검사 기록의 해시.

## 입력과 실행 API

기존 로컬 관측 계약을 사용한다. 수신한 과거48개 값·마스크, 자기/수신 명령9개, 선택적 에너지6개를0.5초마다 전달한다. 20초 이력 후1초마다 예측한다. 상세 열 의미는 이전 `forecast_rebuild/QUICK_START.md`와 같다.

실험 종료 후 새 후보 로더를 사용한다. 이전 로더는 이전 v8 후보를 읽으므로 구분한다.

```python
from forecast_extension.runtime import load_candidate

predictor = load_candidate(
    city=city, observer=relay_id, n_drones=n_drones, n_search=n_search,
    nominal=preflight_nominal_xy, device='cpu', operating_point='advisory',
)
```

지원 운영점은 `advisory`, `common_FA_budget1`, `precision90_FA1_partial`이다. 보정 자료에서 해당 조건을 만족하는 지점이 없으면 대안 운영점 요청을 거부한다. 확률은 같고 경보 문턱만 달라진다. 부분 목표 운영점은 전체 신뢰 경보를 활성화하지 않는다.

`risk_5s=None`을0으로 바꾸지 않는다. 실제로 수신하지 않은 이웃의 상태·명령을 채워 넣지 않는다. 센서가 없으면 명시적으로 미측정 상태를 전달한다. 실제 장비의 원시 로그를48개 관측으로 바꾸는 어댑터와 하드웨어 검증은 별도다.

## 해석 범위

- 훈련 지도245개는 동일하다. 새 난수·대형·방향·간섭 실현과 관측 오차를 추가한다.
- 마지막42비행은 **기존 확인 지도14개에서 새 비행**이다. 새 지도 일반화 또는 실측 성능으로 표현하지 않는다.
- 원래 최종 지도32개는 계속 봉인한다.
- 물리적 RF 세계는 고정된2.4GHz·50m 조건이다. 위치·센서 오차는 측정 가정의 증강이다. 실제 모듈 특성·이동 간섭·지형·기체 자세를 실측한 것은 아니다.
- 기존 PPO의 비행을 고정하여 예측만 비교한다. 이번 결과를 새 연결 유지율·완주 시간 개선으로 해석하지 않는다.
- 모든 목표를 못 맞추면 연구용 후보로 남긴다. 실제 사용 가능 여부는 선택한 기체·통신모듈의 실측으로 확인한다.

## 개발·재현

```powershell
.\drone_forecast_joint_training\.venv\Scripts\python.exe -B -m unittest forecast_extension.test_belief forecast_extension.test_metrics -q
```

상세 실행 기록은 `WORKLOG.md`다. 캐시·모델 태그는 불변으로 취급한다. 확인 후 같은 확인 결과에 맞춰 모델을 재선택하지 않는다. `runtime.py`와 그에 대응하는 연구 후보의 해시가 일치해야 로드된다.
