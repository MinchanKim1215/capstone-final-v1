# 캡스톤 디자인 II — 개발 자료 미탐·오경보 진단
선택90비행. 고정 보정 문턱0.7700. 기존 경보 정의·사건 분모 유지. 확인 자료 미사용.

| 사건 분류 | 사건 수 |
|---|---:|
| risk_below_entry_threshold | 30,641 |
| no_qualified_input | 3,696 |
| timely_first_alert | 1,431 |
| late_alert_only | 29 |
| prior_alarm_still_active | 18 |
| high_risk_without_new_first_alert | 16 |

| 조건 | 사건 수 | 최초 경보 재현율 | 높은 위험 점수 사전 포착률 |
|---|---:|---:|---:|
| all | 35,831 | 3.99% | 4.04% |
| duration_1to2s | 4,398 | 0.52% | 0.52% |
| no_recent_traffic | 32,570 | 3.43% | 3.46% |
| continuous | 14,742 | 2.90% | 2.92% |
| duration_ge5s | 23,577 | 5.48% | 5.54% |
| recent_traffic | 3,261 | 9.63% | 9.84% |
| duration_2to5s | 7,856 | 1.49% | 1.50% |
| clear | 9,108 | 5.35% | 5.42% |
| intermittent | 11,981 | 4.31% | 4.37% |

높은 위험 점수 사전 포착률은 진단값이며 최초 경보 재현율을 대체하지 않는다. 단절 지속시간·통신 관측·명령/간섭 인접성은 분해 기준이며 원인이 확정된 것은 아니다.

전체 사건: events.csv. 사건별 타임라인 원본: results.json의examples.
