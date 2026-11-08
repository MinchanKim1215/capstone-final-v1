"""Read-only selected research profile and its existing benchmark evidence."""
from .profile import load_profile


def main():
    p=load_profile();a=p['benchmark']['selected'];b=p['benchmark']['matched_previous']
    print('캡스톤 디자인 II — 사용자 선택: 탐지율 우선')
    print('예측 모델:',p['model_tag'],'경보 문턱:',p['threshold'])
    print(f"선행 탐지율: {b['event_recall_3s']*100:.2f}% → {a['event_recall_3s']*100:.2f}%")
    print(f"경보 정밀도: {a['precision']*100:.2f}% / 실제 오경보: {a['false_or_late_per_drone_hour']:.2f}회/h")
    print('탐지 사건:',a['event_hits_3s'],'/',a['eligible_directed_link_events'])
    print('기존 동일42비행 검증 수치입니다. 실제 오경보4회/h의 고정 상한을 보장하지 않습니다.')
    print('연구용 설정입니다. 실제 장비 검증·신뢰 경보·자동PPO 교체는 비활성입니다.')


if __name__=='__main__':main()
