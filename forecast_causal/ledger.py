"""Human-readable handoff always says completed, current and next separately."""
from . import ARTIFACTS,WORKSPACE
from .contracts import read,write,now
STAGES=['놓친 장애의 원인 분해','원인에 맞는 관측 정보 보강','원인별 예측기 구성·대조 학습','같은 기준의 신규 비행 검증·채택 판단']

def refresh(status):
    path=ARTIFACTS/'milestones.json';m=read(path) if path.exists() else {}
    lines=['# 캡스톤 예측 개선 — 지속 작업 기록','',f'갱신: {now()}','',
        '사용자 요청: 네 단계를 빠뜨리지 말고 순서대로 진행. 진행 상황·근거·다음 행동을 계속 기록.','',
        '**운영 우선순위: 선행 탐지율 향상, 오경보 약4회/드론·시간, 정밀도 유지. 어려운 사건도 평가 분모에 포함.**','',
        '## 네 단계','']
    for i,label in enumerate(STAGES,1):
        entry=m.get(str(i));lines.append(f'- [{"x" if entry else " "}] {i}. {label}')
        if entry:lines.extend([f'  - 완료: {entry["completed_at"]}',f'  - 근거: {entry["evidence"]}'])
    lines+=['','## 현재 상태','',f'- 상태: {status["state"]}',f'- 단계: {status["stage"]}']
    for key in ('complete','total','variant','step','error','next_action'):
        if key in status:lines.append(f'- {key}: {status[key]}')
    done=max([int(k) for k in m] or [0]);next_action=status.get('next_action') or m.get(str(done),{}).get('next_action','계측 코드를 원본과 대조하고 선택45비행의 장애 원인을 재구성한다.')
    if status['stage']=='stage1_selection_replay':next_action='선택45비행 재생을 끝낸 뒤 기존200개 사건·134개 점수 미달 사건의 원인 증거표를 생성한다.'
    elif status['stage']=='stage1_event_cause_analysis':next_action='사건별 원인 증거와 실제 수신 정보의 전조를 대조하고 1단계 완료 근거를 저장한다.'
    elif status['stage']=='stage2_development_replay':next_action='훈련·선택·보정335비행 재생을 끝내고 관측 가능한 새 입력과 원래 입력 대조군을 구성한다.'
    elif status['stage']=='stage3_tree_training':next_action='기존 입력·새 패킷 관측·경로 불확실성·둘의 결합을 같은 예산으로 학습하고 두 GRU 대조군을 이어서 학습한다.'
    elif status['stage'] in ('stage3_neural_training','event_recall_training'):next_action='기존 이력 GRU와 새 패킷 관측 GRU의 대조 학습 후 후보와 보정 문턱을 고정한다.'
    elif 'confirmation_collection' in status['stage']:next_action='고정된 기존PPO로 새42비행을 수집하고 같은 전체 사건 분모에서12개 사전 고정 후보를 평가한다.'
    elif 'confirmation_features' in status['stage']:next_action='새42비행을 원래 패킷 정답과 정확히 대조해 입력으로 구성하고 고정된 후보를 평가한다.'
    lines+=['','## 다음 행동','',next_action,'',
        '## 유지할 조건','',
        '- 원본 코드·모델 및V8~V15 납품본 보존. 현재 탐지 우선V13 프로필 유지.',
        '- 메인PPO 정책은 자료 생성 기준으로 고정. 예측 개선과 재배치 성과를 구분.',
        '- 원래 최종32지도 미사용. 이전 확인 비행을 새 모델 학습·선택에 사용하지 않음.',
        '- 실제 관측값과 실제 수신 패킷만 예측 입력. 전파 정답·간섭 일정·미래패킷은 분석·정답 전용.',
        '- 기존 사건 정의·3≤선행시간<7초·최초 경보·히스테리시스·전체 사건 분모 유지.',
        '- 훈련/모델 선택/확률·문턱 보정/새 확인을 분리. 결과를 보고 유리한 모델로 다시 선택하지 않음.',
        '- 실행 시작은 완료가 아님. 실패하면 오류와 재개 지점을 기록.',
        '', '## 이번 작업 위치','',
        '- `forecast_causal/`: 이번 단계별 코드·분석·학습.',
        '- `forecast_causal/artifacts/protocol.json`: 사용자 요청과 보호 조건.',
        '- `forecast_causal/artifacts/status.json`: 현재 단계·갱신 시각.',
        '- `forecast_causal/artifacts/milestones.json`: 완료 근거와 다음 행동.',
        '- `forecast_causal/WORKLOG.md`: 기술 인계 기록.',
        '- `forecast_peer/RESULTS_20261103.md`: 이전12모델의 비교 결과. 새 개선 결과와 구분.','']
    target=WORKSPACE/'PROJECT_FOLLOWUP.md';tmp=target.with_suffix('.working.md');tmp.write_text('\n'.join(lines),encoding='utf-8');tmp.replace(target)

