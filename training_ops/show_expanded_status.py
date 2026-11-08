from pathlib import Path
import json

root=Path(__file__).resolve().parent.parent
def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

old=root/'adaptive_joint/artifacts/runs/adaptive_v6_20261030'
p=read(old/'progress.json');s=read(old/'status.json')
print('캡스톤 디자인 II — 현재·후속 학습 상태')
if p:print(f"현재 실험: {p['committed_updates']}회 갱신, {p['training_episodes']}비행 저장 완료")
if s:
    states={'running':'진행 중','completed':'완료','early_stopped':'조기 종료','failed':'실패','interrupted':'중단'}
    stages={'collect':'비행 자료 수집','joint_ppo':'예측·재배치 공동학습','finished':'종료'}
    print('실행 상태:',states.get(s['state'],s['state']),'/',stages.get(s['stage'],s['stage']))
q=read(root/'additional_maps_20261030/queue/status.json')
if q:
    labels={'waiting_for_current_training':'현재 학습 종료 대기','waiting_for_parent_exit':'현재 학습 종료 처리 대기',
            'prepare':'후속 초기 모델 연결·설정 확인','assets':'기준 경로 검사','preflight':'후속 전체 사전 검증',
            'train':'추가 지도 학습 진행','completed':'후속 학습 종료','failed':'후속 작업 실패 — 로그 확인 필요',
            'blocked_parent_interrupted_or_failed':'현재 학습 중단으로 후속 시작 보류','cancelled':'후속 예약 취소'}
    print('후속 실험:',labels.get(q['stage'],q['stage']))
    print('최근 확인:',q['updated_at'])
else:print('후속 실험: 지도·코드 준비 중')
new=root/'adaptive_joint_expanded/artifacts/runs/adaptive_v7_expanded100_20261031'
p=read(new/'progress.json')
if p:print(f"후속 저장: {p['committed_updates']}회 갱신, {p['training_episodes']}비행")
