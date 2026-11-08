# 추가 원시 실행 기록

본 학습과 추론에는 **본 ZIP만 있으면 됩니다**. 부트스트랩 훈련·검증 32회와 검증 best 모델 두 개·physics 기준선의 비교 비행 24회 원시 NPZ를 본 ZIP에 넣었습니다. 모든 실행의 checkpoint·메타데이터·SHA·안전 감사·요약은 본 ZIP에 있습니다.

추가 `capstone_transformer_raw_evidence_20261029.zip`에는 나머지 초기 smoke, 중단 재개 probe, 두 모델의 PPO 훈련, 비선택 checkpoint 검증 비행 NPZ를 보관했습니다. 같은 상위 폴더에 풀면 기존 프로젝트 경로에 합쳐집니다. 제공 PPO는 모델·Adam·난수 상태에서 이어가므로 과거 훈련 NPZ 없이 재개할 수 있습니다. 매 update 검증 설정 때문에 최신 checkpoint의 검증 원시 파일이 없으면 요청 조건을 확인한 뒤 해당 검증 비행을 재생성합니다. 추가 ZIP을 같이 풀면 이를 재사용할 수 있습니다.

본 ZIP을 처음 푼 직후 `python release_verify.py --full`로 전달된 바이트를 검사하세요. 추가 기록도 풀었다면 `python release_verify.py --full --optional-evidence`로 전체를 검사합니다. 이후 학습은 progress·latest·best·요약을 갱신하므로 `--full`은 원본 압축 해제 검증용입니다. 기본 `python release_verify.py`는 변경하지 않아야 하는 코드·지도·모델·문서만 확인합니다.

원시 데이터 원본을 지워 과거 결과가 사라지는 일을 막기 위해 별도 보관했으며, 원래 100회 실행 중 누락된 실험을 완료했다고 간주한 것이 아닙니다. 전체 수치와 실행 범위는 `TRAINING_REPORT.md`에 기록했습니다.
