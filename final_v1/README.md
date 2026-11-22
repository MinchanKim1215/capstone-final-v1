# Final V1

캡스톤 디자인 II의 시뮬레이션 연구 버전이다. 단절 원인과 선행 예측, PPO 행동 선택, 통신을 고려한 재배치, 명령 미수신 시 대응을 네 단계로 확인한다.

## 결과 보기

- [결과 보고서](RESULTS.md)
- [동일 조건 비행 재생](Final%20V1%20재배치%20재생.html)
- [검증된 버전과 파일 지문](artifacts/release.json)
- [검증 진행 상태](artifacts/status.json)

재생 HTML을 브라우저에서 연다. 위 조건 목록에서 지도와 간섭을 선택한다. 왼쪽과 오른쪽에서 각 제어 방식을 바꿀 수 있다. **다음 사건** 버튼으로 단절, 재배치, 명령 손실 시점에 이동한다. 두 지도는 같은 비행 시각을 표시한다. S는 수색기, R은 중계기다.

## 재현

프로젝트 루트에서 기존 학습 가상환경을 사용한다. 345개 지도 원인 조사, 기존 예측 특징 파일과 고정 V7 체크포인트가 필요하다.

```powershell
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1.causes
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1.training
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1.campaign
```

현재 기본값은 0.5초 Transformer–PPO 제어다. 팀 ZIP에 포함된 광주 지도를 대표 설정으로 사용한다. `--command-loss`는 명령 전달 손실 시험을 추가한다. `--steps 48`을 추가하면 기본 제어의 24초 실행을 확인한다. [0.5초 제어와 사용법](halfsec_main/README.md)

```powershell
& .\drone_forecast_joint_training\.venv\Scripts\python.exe -m final_v1.fly
```

## 기존 3초 Transformer 평가

11월 8일 검증된 3초 비교 방식은 `v21_transformer`다. 링크 단절 시점 출력층을 갱신하고, 전체망 위험과 V21 점수를 결합한다. 새 지도 12곳의 36개 조건에서 당시 V21 대비 새 단절 115→61건, 단절 시간 764.5→452.0초, 기한 내 패킷 전달 98.24→98.89%였다. 양쪽 모두 36개 조건을 완주했다.

같은 지역과 난수에서 기준 설정을 비교하려면 `--controller v21`을 사용한다. `--controller v21_transformer`는 업데이트 모델을 명시적으로 선택한다. 보정과 채택 근거는 [업데이트 평가 수치](transformer_upgrade/artifacts/results.json), 두 모델과 실행 설정은 `transformer_upgrade/artifacts/`에 있다.

## 3초 재배치 비교

`--controller v21_transformer_recovery`로 3초 비콘 누락 복구를 비교한다. 초기 재배치 실험의 방법과 수치는 [첨부 연구 요약](../final_v1_relocation/SUMMARY.md)에 있다.

## 단계와 평가 단위

1. **원인과 패턴** · 345개 지도 기록에서 실제 RF 판정을 재현했다. 급락·점진·혼합은 단절 전후 통신 여유의 형태를 뜻한다. 원인 제거 분석과 분리한다.
2. **PPO 연동** · 245개 훈련 지도에서 만든 입력으로 예측기를 학습했다. 고정 V7 PPO의 후보 점수에 작은 학습 모듈을 더했다. 새 연속 입력을 쓰는 정책과 제외한 정책을 같은 훈련 난수로 갱신했다. 기존 V7 예측을 두 정책에 공통으로 유지했다.
3. **동일 조건 비교** · 14개 검증 지도에서 세 간섭 조건의 새 난수를 사용했다. 예측 비교는 같은 V7 궤적에서 수행했다. 제어 비교는 각 방식의 명령을 비행과 패킷 전달에 반영했다. 단절은 수색망 연결이 1초 이상 끊긴 구간이다.
4. **명령 손실과 복구** · 첫 중계기를 명령 발생 지점으로 둔다. 실제 채널 수식의 직전 RF 프레임으로부터 최대 3홉 명령 수신을 표본 추출한다. 수신 실패 시 기본 대형 또는 직전 명령 한 주기 유지와 로컬 관측 기반 보정을 비교했다. 안정 복구는 연결이 연속 1초 이상 유지되기 시작한 시점으로 측정한다.

명령 수신은 별도의 확률 모델이다. 공동 항법 계획이 공유된다고 가정한다. 실제 대역폭 경쟁, 명령 지연, 기체 안전 제어기, 실기체 비행 검증은 포함하지 않는다. 네 단계에 사용하는 이전 모델과 자료는 원본 경로에 보관된다.
