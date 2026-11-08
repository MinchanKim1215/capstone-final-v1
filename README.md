# 캡스톤 디자인 II · Final V1

2026-11-08 기준 최신 소스와 연구 결과를 모은 저장소다. 드론 편대 통신 단절을 예측하고 중계기를 선제 재배치하는 시뮬레이션 연구를 담았다.

## 보고서

- [통합 HTML](reports/Final_V1_통합보고서.html)의 단절 유형·원인 탭에서 345개 지도와 2,572개 단절 사건을 확인한다. 56조건 비행 재생도 내장돼 있다.
- [단절 유형 상세](reports/단절유형/급락형_점진형_혼합형_상세분석.md)는 급락형·점진형·혼합형과 그래프를 담았다.
- [단절 전 30초 이력](reports/단절_30초_이력.html), [연결 유지 급락 분석](reports/급락과_단절_위험.html), [빠른 명령 주기](reports/빠른_명령_주기.html), [Final V1 재배치 재생](final_v1/Final%20V1%20재배치%20재생.html)을 별도로 열 수 있다.
- HTML 파일은 내려받아 Chrome, Edge 또는 Safari에서 연다. 기존 HTML의 평가 시점은 각 보고서에 표시했다.

## 최신 버전

- [Final V1 실행 구조](final_v1/README.md)와 [Transformer 업데이트 수치](reports/transformer-upgrade-summary.json)를 참조한다. 동일한 36개 시뮬레이션 조건에서 새 단절 115→61건, 단절 시간 764.5→452.0초, 기한 내 전달률 98.24→98.89%였다.
- [장기 공동학습 소스](final_v1/long_training/)와 [학습 계획](reports/long-training-plan.json)에 Transformer·PPO·대형 선택과 후속 전체망 학습을 담았다.
- [0.5초 제어 연구](fast_halfsec/RESULTS.md)와 [후보 선택 코드](halfsec_choice/)는 별도 실험이다.
- 기본 시뮬레이션·통신 엔진은 [원본 프로젝트](drone_forecast_joint_training/)에 있다.

이 저장소의 전달 범위는 소스, 설정, HTML과 요약 지표다. 시뮬레이션과 학습을 재현하려면 팀이 보유한 지도 데이터, 가중치, 해당 연구의 저장본과 기존 Python 환경이 필요하다.
