# Final V1

캡스톤 디자인 II의 시뮬레이션 연구 버전이다. 단절 원인과 선행 예측, PPO 행동 선택, 통신을 고려한 재배치, 명령 미수신 시 대응을 네 단계로 확인한다.

## 결과 보기

- [결과 보고서](RESULTS.md)
- [동일 조건 비행 재생](Final%20V1%20재배치%20재생.html)
- [검증된 버전과 파일 지문](artifacts/release.json)
- [검증 진행 상태](artifacts/status.json)

재생 HTML을 브라우저에서 연다. 위 조건 목록에서 지도와 간섭을 선택한다. 왼쪽과 오른쪽에서 각 제어 방식을 바꿀 수 있다. **다음 사건** 버튼으로 단절, 재배치, 명령 손실 시점에 이동한다. 두 지도는 같은 비행 시각을 표시한다. S는 수색기, R은 중계기다.

## 제출용 시뮬레이션 실행

기본 실행은 수색기 3대와 중계기 5대로 구성한 `halfsec_planned_eight`다. 비행 전에 건물과 전체 수색 경로의 전파 연결을 점검하고, 기존 중계기 중 고정 위치를 맡을 기체와 구간별로 고도를 바꿀 기체를 배정한다. 임무 중 수색 구간 진입 전에 통신을 확인한다. Transformer와 PPO는 0.5초마다 수평 재배치를 판단한다.

```powershell
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -B -m final_v1.fly
```

`--steps 48`로 기본 광주 지도의 24초 비행을 확인할 수 있다. `--controller v21_transformer_halfsec`로 이전 0.5초 제어를 실행한다. [실행 인자와 배치 방식](halfsec_main/README.md)

## 재현

프로젝트 루트에서 기존 학습 가상환경을 사용한다. 345개 지도 원인 조사, 기존 예측 특징 파일과 고정 V7 체크포인트가 필요하다.

```powershell
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1.causes
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1.training
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1.campaign
```

## 기존 3초 Transformer 평가

11월 8일 검증된 3초 비교 방식은 `v21_transformer`다. 링크 단절 시점 출력층을 갱신하고, 전체망 위험과 V21 점수를 결합한다. 새 지도 12곳의 36개 조건에서 당시 V21 대비 새 단절 115→61건, 단절 시간 764.5→452.0초, 기한 내 패킷 전달 98.24→98.89%였다. 양쪽 모두 36개 조건을 완주했다.

같은 지역과 난수에서 기준 설정을 비교하려면 `--controller v21`을 사용한다. `--controller v21_transformer`는 업데이트 모델을 명시적으로 선택한다. 보정과 채택 근거는 [업데이트 평가 수치](transformer_upgrade/artifacts/results.json), 두 모델과 실행 설정은 `transformer_upgrade/artifacts/`에 있다.

## 지속 비콘 누락 시 재배치

`--controller v21_transformer_recovery`는 같은 수색기의 비콘 누락이 두 판단 주기 연속 확인될 때 복구 후보를 평가한다. 기존 제어가 개입한 경우 그 결정을 사용한다. 수색기의 다음 위치를 바꾸거나 정상 연결을 끊을 것으로 예측되는 후보를 제외한다. 재연결이 빨라지거나 통신 여유 부족분이 줄어드는 중계 보정을 선택한다.

`python -m final_v1.relocation_evaluation`은 새 지도 15곳의 45조건을 기존 3초 모델과 짝지어 비교한다. 새 단절 감소, 단절 시간, 임무 완주·시간, 기한 내 패킷 전달과 안전을 검증한다.

`python -m final_v1.relocation_replication`은 첫 비교에 쓰이지 않은 지도 8곳의 24조건에서 같은 코드를 재검증한다. [첫 비교](artifacts/relocation/results.json) · [별도 지도 재검증](artifacts/relocation/replication/results.json)

11월 20일 결과에서 첫 45조건의 새 단절은 257→230건, 별도 24조건은 81→98건이었다. 총 69조건을 양쪽 모두 완주했다. 이 규칙은 3초 비교 옵션으로 제공한다.

## 단계와 평가 단위

1. **원인과 패턴** · 345개 지도 기록에서 실제 RF 판정을 재현했다. 급락·점진·혼합은 단절 전후 통신 여유의 형태를 뜻한다. 원인 제거 분석과 분리한다.
2. **PPO 연동** · 245개 훈련 지도에서 만든 입력으로 예측기를 학습했다. 고정 V7 PPO의 후보 점수에 작은 학습 모듈을 더했다. 새 연속 입력을 쓰는 정책과 제외한 정책을 같은 훈련 난수로 갱신했다. 기존 V7 예측을 두 정책에 공통으로 유지했다.
3. **동일 조건 비교** · 14개 검증 지도에서 세 간섭 조건의 새 난수를 사용했다. 예측 비교는 같은 V7 궤적에서 수행했다. 제어 비교는 각 방식의 명령을 비행과 패킷 전달에 반영했다. 단절은 수색망 연결이 1초 이상 끊긴 구간이다.
4. **명령 손실과 복구** · 첫 중계기를 명령 발생 지점으로 둔다. 실제 채널 수식의 직전 RF 프레임으로부터 최대 3홉 명령 수신을 표본 추출한다. 수신 실패 시 기본 대형 또는 직전 명령 한 주기 유지와 로컬 관측 기반 보정을 비교했다. 안정 복구는 연결이 연속 1초 이상 유지되기 시작한 시점으로 측정한다.

명령 수신은 별도의 확률 모델이다. 공동 항법 계획이 공유된다고 가정한다. 실제 대역폭 경쟁, 명령 지연, 기체 안전 제어기, 실기체 비행 검증은 포함하지 않는다. 네 단계에 사용하는 이전 모델과 자료는 원본 경로에 보관된다.
