**예측기 동일 관측 예비 검증**

2026-10-29T14:01:02+09:00

기존 CUDA 본 학습을 변경하지 않고 고정된 검증 관측 8회를 CPU 1개 스레드로 재생했다. 새 시뮬레이션·학습·최종 테스트는 수행하지 않았다.

이 자료는 이미 PPO checkpoint 선택에 사용한 맵이다. 공간 그룹 교차 적합은 혼합 비율과 경보 문턱의 직접 과적합을 줄일 뿐, 과거 모델 선택 편향까지 제거하지 않는다.

**동일 링크·시점·정답에서의 AP**

| 모델 | 5초 위험 AP | 3–4초 시작 AP | 3–7초 시작 AP |
| --- | --- | --- | --- |
| pretrain1200 | 0.2961 | 0.0800 | 0.1314 |
| pretrain2800 | 0.3658 | 0.1046 | 0.1532 |
| joint176 | 0.3835 | 0.1170 | 0.1537 |
| joint190 | 0.3825 | 0.1151 | 0.1568 |
| lightgbm | 0.4736 | 0.1214 | 0.1875 |
| blend_crossfit | 0.4654 | 0.1194 | 0.1817 |

LightGBM이 확률을 낼 수 있는 공통 지원 표본으로 비교했다. Transformer 원래 지원 범위의 결과와 제외 표본 수는 classification.json/results.json에 별도 저장했다.

**공간 그룹 교차 적합 사건 경보**

| 모델 | 경보 | 조건 | 경보 정밀도 | 오경보/드론·시간 | 3초 선행 사건 재현율 | 탐지/사건 | 목표 전부 충족 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pretrain2800 | 5s | legacy | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| pretrain2800 | 5s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| pretrain2800 | 8s | legacy | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| pretrain2800 | 8s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| joint176 | 5s | legacy | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| joint176 | 5s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| joint176 | 8s | legacy | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| joint176 | 8s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| joint190 | 5s | legacy | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| joint190 | 5s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| joint190 | 8s | legacy | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| joint190 | 8s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| lightgbm | 5s | legacy | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| lightgbm | 5s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| lightgbm | 8s | legacy | 57.14% | 0.151 | 0.17% | 3/1779 | False |
| lightgbm | 8s | trust_goal | 50.00% | 0.101 | 0.11% | 2/1779 | False |
| blend_crossfit | 5s | legacy | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| blend_crossfit | 5s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |
| blend_crossfit | 8s | legacy | 66.67% | 0.050 | 0.06% | 1/1779 | False |
| blend_crossfit | 8s | trust_goal | 0.00% | 0.000 | 0.00% | 0/1779 | False |

legacy는 문턱 선택 시 정밀도 80%·오경보 5회/h·경보 10건, trust_goal은 90%·1회/h·10건을 요구한다. 목표 전부 충족 여부에는 최소 3초 선행 사건 재현율 80%도 포함했다. 숫자는 개발 목표이며 운용 승인 기준이나 이미 달성한 성능으로 정하지 않았다.

튜닝 맵에서 허용 가능한 규칙이 없으면 그 fold의 경보를 끄고, 평가 맵의 놓친 사건은 분모에 그대로 남겼다. 최초 경보 정밀도는 horizon 내 사건 유무, 사건 재현율은 3초 이상 선행한 별도 정의다.

**혼합 비율**

- fold 1: LightGBM 비중 0.75; 평가 지역 gyeonggi_reserve_20261027_11, seoul_20261026_10
- fold 2: LightGBM 비중 0.75; 평가 지역 gyeonggi_20261026_batch2_15, gyeonggi_fast_20261027_09
- fold 3: LightGBM 비중 1.00; 평가 지역 gyeonggi_20261025_04, gyeonggi_extra_20261025_02
- fold 4: LightGBM 비중 0.75; 평가 지역 seoul_20261026_batch2_extra_05, gyeonggi_extra_20261027_05

혼합은 176 Transformer와 LightGBM의 확률 가중 평균이다. 비율은 각 fold의 나머지 6개 맵에서 AP 점수로 선택했다. 복잡한 stacking이나 새 확률 보정기는 학습하지 않았다.

**확인할 다음 항목**

1. 모델별 AP 개선과 사건 경보 개선이 같은지 확인한다. AP만 오른 혼합을 신뢰 가능한 경보로 채택하지 않는다.
2. calibration_bins, recall_at_precision, error_complementarity와 지역별 결과에서 누락 사건/오경보 집중 조건을 확인한다.
3. 이 결과로 최종 테스트를 열거나 기존 학습에 혼합을 주입하지 않는다. 전체 검증과 독립된 튜닝/평가 설계가 다음 단계다.
4. 별도 모델 재학습, 손실 계수 조정, 모델 크기 변경은 아직 수행하지 않았다.

**제한**

- These eight maps were already used to select checkpoints; cross-fitting blend weights and alarm thresholds does not remove earlier checkpoint-selection bias.
- Diagnostic only: not full 96-case validation, final test, operational alarm authorization, or a controller rollout comparison.
- Historical LightGBM training bank is not included; training-only checkpoint metadata and fresh-map protocol are checked, but raw bank provenance cannot be independently reconstructed.
- No probability calibrator is fitted in this first audit; existing probabilities and calibration diagnostics are reported.
