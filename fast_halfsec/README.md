# Final V1 후속 · 0.5초 선제 재배치 학습

2026-11-07 15:40 KST에 학습과 확인 180비행을 마쳤다. [결과 보고서](RESULTS.md) · [완주 조건 민감도](artifacts/result_review.json) · [예측 점수 추가 후 궤적 대조](artifacts/identical_trajectory_check.json)

Drive 지도 6개 묶음을 검증하고 기존 345개에 새 장면 300개를 등록했다. 공간 간격 100m, 공통 건물 ID, 기존 공간 그룹, 보호 구역과의 겹침을 검사했다. 새 지도 206개는 학습, 48개는 보정, 39개는 확인, 7개는 격리로 배정했다. 실제 시험에 사용한 지도와 난수는 `artifacts/protocol.json`에서 확인한다.

## 제어와 학습

1. 기존 V7 대형 선택기와 무선에 독립적인 수색 계획을 불러온다. 수색기가 실제로 도착하면 다음 지점으로 진행한다. 수색 계획과 안전한 비행 간선은 지도에 기반한다.
2. 1Hz 비컨에서 수신에 성공한 정보와 현재 공유 항법을 사용한다. 기존 통신 여유와 최근 30초 수신 변화를 별도로 기록한다. 같은 보고를 여러 번 읽어도 새 급락으로 중복 집계하지 않는다.
3. 0.5초마다 계획을 검사한다. 수신 처리 0.1초, 계산 0.2초, 전달 0.1초를 반영한다. 명령 보정은 수신한 뒤 적용한다. 이동·패킷·전달 계산은 0.1초, 전파 판정은 0.5초 간격이다.
4. 학습용 60비행의 기록으로 3~7초 선행 위험 모델과 0.5~3초 임박 위험 모델을 학습한다. 최근 30초 이력을 제외한 모델도 같은 자료로 학습한다. 문턱은 별도 보정 지도 15개에서 정한다.
5. 안전 검사를 통과한 한 대의 보정 후보와 유지 선택 중 행동을 고르도록 PPO를 학습한다. 새로 실행한 16비행으로 두 번 갱신한다. 3초 단위의 할인율을 실제 경과 시간에 맞춰 적용한다.
6. 새 지도 12개·세 간섭 조건의 36조건에서 다섯 방식을 고정 비교한다. 동일 환경의 3초·0.5초 물리 제어, 0.5초 학습 예측, 0.5초 학습 예측과 PPO, 원래 Final V1 3초 제어를 포함한다.

같은 0.5초 물리 제어 궤적에서 예측 점수를 비교한다. 제어 지표는 각각 다른 명령으로 끝까지 실행한 결과이다. 새 환경의 3초와 0.5초 비교는 판단 주기를 분리한다. 원래 Final V1과의 비교에는 운동·통신 모형 차이도 포함된다. 수색 경로 미완료는 완주율과 커버리지에 포함한다.

## 실행과 재개

작업 폴더에서 기존 Python 환경을 사용한다. 지도 등록은 검증된 ZIP과 등록부의 지문을 보관한다. 학습·평가 결과는 비행마다 저장한다. 중단되면 같은 명령으로 검증된 결과부터 이어서 진행한다.

```powershell
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m fast_halfsec.download_maps
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m fast_halfsec.import_maps
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m fast_halfsec.collect_supervision --workers 6
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m fast_halfsec.campaign --workers 6
```

- `artifacts/status.json`에서 현재 단계와 완료 건수를 확인한다.
- `artifacts/map_import_audit.json`에 다운로드·건물·비행 제한 구역 검증 결과를 보관한다.
- `artifacts/models/cutoff_v2/forecast_profile.json`에 LightGBM 가중치와 상위 점수대를 세분화한 보정 문턱을 연결한다.
- `artifacts/models/cutoff_v2/actor_2.pt`에 두 번째 PPO 갱신의 결과를 저장한다.
- `artifacts/freeze.json`에 확인 비행 실행 전 고정한 코드·모델·자료 지문을 기록한다.
- `RESULTS.md`와 `artifacts/final_results.json`에 확인 결과와 채택 기준을 기록한다.
