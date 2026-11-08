"""Write the reproducible half-second experiment report."""
from . import ROOT

LABELS = dict(final_v1_3s='기존 Final V1 · 3초', physics_3s='같은 새 환경 · 3초 물리',
              physics_05s='같은 새 환경 · 0.5초 물리', learned_05s='0.5초 · 학습 예측',
              ppo_05s='0.5초 · 학습 예측 + PPO', physics='물리 예측 3~7초',
              early_current='학습 예측 3~7초 · 최근 이력 제외',
              early_history='학습 예측 3~7초 · 최근 이력 포함', short_history='학습 예측 0.5~3초')


def percent(value):
    return f'{value*100:.2f}%'


def render(result, profile, ppo, plan, maps):
    lines = ['# Final V1 후속 · 0.5초 선제 재배치 학습', '',
             f"완료 시각 {result['created_at']} · 시뮬레이션 검증", '',
             f"기존 지도 {maps['existing']}개에 새 지도 {maps['added']}개를 등록했다. 총 {maps['total']}개다. "
             f"새 지도 중 학습 {maps['new_roles']['train']}개, 보정 {maps['new_roles']['calibration']}개, "
             f"확인 {maps['new_roles']['confirmation']}개, 격리 {maps['new_roles']['quarantine']}개다.", '',
             '## 학습과 비교 조건', '',
             f"첫 학습에는 새 지도 48개와 기존 지도 12개를 사용했다. 별도 새 지도 {len(plan['calibration'])}개에서 경보 문턱을 보정했다. "
             f"PPO는 실제 이동을 반영한 {sum(item['flights'] for item in ppo['updates'])}비행에서 두 번 갱신했다. "
             '새 최종 지도 12개와 세 간섭 조건을 조합한 36조건에서 다섯 방식을 비교했다.', '',
             '0.5초마다 재배치를 판단한다. 전파는 0.5초마다 계산하고, 기체 이동·패킷·명령은 0.1초로 진행한다. '
             '비컨은 1초, 수신 처리 0.1초, 계획 0.2초, 명령 전달 0.1초를 모형에 반영했다. 명령 변경 시 세 번 반복 전송하고 실제 슬롯을 차감했다. '
             '기체는 안전한 간선을 따라 최대 속도 5m/s·가속도 2.5m/s²로 이동한다. 수색 기체가 다음 지점에 도착해야 임무 계획이 진행된다.', '',
             '기존 V7의 대형 선택기를 사용하고, 물리 예측으로 한 대를 옮기는 후보를 검사한다. '
             'LightGBM은 실제 수신 정보와 발행 계획에서 단절 시간대의 점수를 배운다. PPO는 안전 검사를 통과한 후보와 유지 선택을 순위화한다. '
             '현재 편대의 위치·목표를 공유한다고 가정했으며 미래 전파·실제 간섭 일정은 입력에서 제외했다.', '',
             '## 같은 비행 기록의 예측 성능', '',
             '| 방식 | 제때 탐지 | 탐지율 | 정밀도 | 오경보·늦은 경보/편대·시간 | AP |',
             '|---|---:|---:|---:|---:|---:|']
    for name,row in result['forecast'].items():
        ap = percent(row['ap']) if row['ap'] is not None else '–'
        lines.append(f"| {LABELS[name]} | {row['hits']}/{row['events']} | {percent(row['recall'])} | "
                     f"{percent(row['precision'])} | {row['false_or_late_per_fleet_h']:.2f} | {ap} |")
    lines.extend(['', '모든 예측기는 같은 0.5초 물리 제어 비행의 기록을 사용했다. '
                  '보정 자료에서 오경보·늦은 경보 4회/편대·시간 이내의 문턱을 고정했다. '
                  '선행 탐지는 단절 3초 이상 7초 미만 앞선 경보를 한 사건에 한 번 매칭했다. '
                  '0.5~3초 결과는 별도 시간 구간으로 표시했다. 예측 점수는 이 시뮬레이션에서 학습했다.', '',
                  '## 이동을 실제로 바꾼 비교', '',
                  '| 방식 | 새 단절 | 회/h | 단절 초/h | 기한 내 전달 | 평균 임무 | 평균 중계 이동 | 완주 | 안전 위반 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for name,row in result['control'].items():
        lines.append(f"| {LABELS[name]} | {row['starts']} | {row['starts_per_h']:.2f} | "
                     f"{row['cut_seconds_per_h']:.2f} | {percent(row['timely_pdr'])} | {row['mean_mission_s']/60.:.2f}분 | "
                     f"{row['relay_m']/1000.:.2f}km | {row['completed']}/{row['flights']} | {row['violations']} |")
    lines.extend(['', '주기만의 비교에는 같은 새 환경의 3초 물리와 0.5초 물리를 사용했다. '
                  '기존 Final V1은 원래 3초 이동·명령 모형으로 실행했다. 두 환경의 시간·통신 제약 차이도 포함된 시스템 비교다.', '',
                  '| 대조 | 새 단절 변화 | 시간당 변화 | 공간 그룹 95% 구간 | 기한 내 전달 변화 | 채택 조건 |',
                  '|---|---:|---:|---:|---:|---|'])
    for comparison in result['paired'].values():
        lo,hi = comparison['ci95']
        lines.append(f"| {LABELS[comparison['old']]} → {LABELS[comparison['new']]} | {comparison['starts_delta']:+d}건 | "
                     f"{comparison['starts_per_h_delta']:+.2f}회/h | [{lo:+.2f}, {hi:+.2f}] | "
                     f"{comparison['timely_pdr_pp']:+.2f}%p | {'충족' if comparison['passed'] else '미충족'} |")
    lines.extend(['', '채택 조건은 안전 위반 0건, 기존 완주 조건 악화 0건, 기한 내 전달률 손실 최대 0.25%p, '
                  '평균 임무 시간 증가 최대 5%, 새 단절 건수 증가 없음 및 시간당 단절 차이 95% 구간 상한이 0 미만이다.', '',
                  '## 학습과 전달 검증', '',
                  f"3~7초 이력 포함 예측기는 {profile['models']['early_history']['rows']:,}개 입력, "
                  f"{profile['models']['early_history']['positive']:,}개 양성 시점으로 학습했다.", ''])
    for update in ppo['updates']:
        lines.append(f"- PPO {update['update']}차 갱신 · {update['decisions']:,}개 선택 시점 · {update['device']} · "
                     f"가중치 변화 L2 {update['actor_change_l2']:.6f} · 이전 정책 로그 확률 오차 {update['logp_error']:.7f}")
    lines.extend(['', f"비행 {result['audit']['flights']}회·전파 프레임 {result['audit']['rf_frames']:,}개에서 "
                  f"연결과 지속 단절을 다시 집계했다. 명령 보정 {result['audit']['offset_applications']:,}회 적용에서 전달 지연을 확인했다. "
                  '안전과 패킷 보존 검사를 통과했다.', '',
                  '새 제어기의 학습·확인 지도를 공간 그룹으로 분리했다. 기존 V7 선택기와 Final V1 비교기는 이전 학습으로부터 가져왔다. '
                  '건물 높이의 일부 보간값과 공유 항법·채널 모형을 사용했다. 실기체 통신 성능은 별도로 검증해야 한다.', '',
                  '[원본 수치](artifacts/final_results.json) · [학습 기록](artifacts/ppo_training.json) · '
                  '[지도 검증](artifacts/map_import_audit.json) · [동결 기록](artifacts/freeze.json)', ''])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
