# 캡스톤 디자인 II · Final V1

2026-11-24 기준 제출용 소스와 연구 결과를 모은 저장소다. 드론 편대 통신 단절을 예측하고 중계기를 선제 재배치하는 시뮬레이션 연구를 담았다.

## 제출용 기본 실행

- 기본 실행은 수색기 3대와 중계기 5대로 구성한 `halfsec_planned_eight`다. 비행 전에 건물 높이와 전체 수색 경로의 전파 연결을 평가하고, 기존 중계기 안에서 고정 위치와 구간별 고도를 배정한다. 비행 중에는 통신을 확인한 뒤 수색기가 위험 구간을 통과한다. [배치와 고도 조정](final_v1/halfsec_main/README.md)
- 수평 재배치의 판단 주기는 0.5초다. 실행 결과에 선택한 배치와 고도 구간, 완주 여부, 단절 시간을 표시한다. [실행 인자](final_v1/README.md#제출용-시뮬레이션-실행)

## 팀 실행과 이전 결과

- 기존 팀 ZIP을 새 폴더에 푼다. 최신 소스 ZIP을 따로 풀고, 최상위 폴더 안의 파일과 폴더를 팀 ZIP의 `code/`에 같은 경로로 합친다. [팀 ZIP](https://github.com/MinchanKim1215/capstone-final-v1/releases/tag/final-v1-20261113) · [최신 소스 ZIP](https://github.com/MinchanKim1215/capstone-final-v1/archive/refs/heads/main.zip)
- 기존 Python 환경으로 `code/`에서 `python -B run_latest_final_v1.py --steps 48`을 실행하면 기본 광주 지도의 사전 배치와 24초 비행을 확인한다. `--steps`를 생략하면 전체 임무를 실행한다. `--controller v21_transformer_halfsec`는 이전 0.5초 방식이다.
- 11월 22일 비교에서는 별도 지도 8곳에서 3초 제어 대비 0.5초 제어의 새 단절이 26→20건, 시간당 새 단절이 6.25→3.35회였다. 양쪽 모두 8/8을 완주했다. [당시 검증 수치](final_v1/halfsec_main/validation_summary.json)
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
