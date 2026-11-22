# 캡스톤 디자인 II · Final V1

2026-11-22 기준 소스와 완료된 연구 결과를 모은 저장소다. 드론 편대 통신 단절을 예측하고 중계기를 선제 재배치하는 시뮬레이션 연구를 담았다.

## 기본 0.5초 제어

- 기본 실행은 [0.5초 Transformer–PPO](final_v1/halfsec_main/README.md)다. 비콘 관측 공백과 예측 통신 여유를 확인해 복구 후보를 걸러내고, 정체가 생기면 수색 통로를 비운다.
- 별도 지도 8곳에서 기존 3초 대비 새 단절은 26→20건, 시간당 6.25→3.35회, 평균 임무 시간은 31.2→44.8분이었다. 양쪽 모두 8/8을 완주했다. [검증 수치](final_v1/halfsec_main/validation_summary.json)

## 팀 실행과 이전 결과

- [기존 팀 ZIP](https://github.com/MinchanKim1215/capstone-final-v1/releases/tag/final-v1-20261113)을 푼다. 이 저장소에서 `final_v1/halfsec_main/`, `final_v1/fly.py`, `final_v1/relocation.py`를 같은 위치의 `code/`에 복사한다.
- 기존 Python 환경으로 `code/`에서 `python -B run_latest_final_v1.py --steps 48`을 실행하면 **0.5초 판단 24초**를 확인한다. `--steps`를 생략하면 전체 임무를 실행한다.
- [이전 3초·0.5초 공동학습 결과와 부산 교차 검증](reports/joint-training-results-20261110.md) · [11월 8일 주간보고서 HTML](reports/Final_V1_Weekly_Report_20261108.html)

## 보고서

- [통합 HTML](reports/Final_V1_통합보고서.html)의 단절 유형·원인 탭에서 345개 지도와 2,572개 단절 사건을 확인한다. 56조건 비행 재생도 내장돼 있다.
- [단절 유형 상세](reports/단절유형/급락형_점진형_혼합형_상세분석.md)는 급락형·점진형·혼합형과 그래프를 담았다.
- [단절 전 30초 이력](reports/단절_30초_이력.html), [연결 유지 급락 분석](reports/급락과_단절_위험.html), [빠른 명령 주기](reports/빠른_명령_주기.html), [Final V1 재배치 재생](final_v1/Final%20V1%20재배치%20재생.html)을 별도로 열 수 있다.
- HTML 파일은 내려받아 Chrome, Edge 또는 Safari에서 연다. 기존 HTML의 평가 시점은 각 보고서에 표시했다.

## 최신 버전

- [Final V1 실행 구조](final_v1/README.md)와 [Transformer 업데이트 수치](reports/transformer-upgrade-summary.json)를 참조한다. 동일한 36개 시뮬레이션 조건에서 새 단절 115→61건, 단절 시간 764.5→452.0초, 기한 내 전달률 98.24→98.89%였다.
- [0.5초 공동학습 소스](final_v1/halfsec_joint/)와 [부산 재현 코드](tmp/analyze_halfsec_joint_20261110.py)를 제공한다.
- [장기 공동학습 소스](final_v1/long_training/)와 [학습 계획](reports/long-training-plan.json)에 Transformer·PPO·대형 선택과 후속 전체망 학습을 담았다.
- [0.5초 제어 연구](fast_halfsec/RESULTS.md)와 [후보 선택 코드](halfsec_choice/)는 별도 실험이다.
- 기본 시뮬레이션·통신 엔진은 [원본 프로젝트](drone_forecast_joint_training/)에 있다.

이 저장소는 소스, 기본 0.5초 추론 가중치, 설정과 요약 결과를 제공한다. 팀 ZIP의 지도·기초 모델과 기존 Python 환경이 있어야 비행을 실행할 수 있다. 학습 재현에는 별도의 학습 원자료와 저장본이 필요하다.
