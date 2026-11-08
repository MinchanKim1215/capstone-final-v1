from . import ARTIFACTS
from .contracts import read


def main():
    print('예측 원인 분석·패킷 서비스 예측 실험 v10\n')
    for label,name in [('전체 진행','status.json'),('사건 진단','diagnostics/status.json'),('패킷 복원','replay_status.json'),('현재 평가','evaluation_status.json')]:
        path=ARTIFACTS/name
        if not path.exists():continue
        data=read(path);parts=[str(data[key]) for key in ('state','stage','task','model','scope') if key in data]
        if 'complete' in data:parts.append(str(data['complete'])+'/'+str(data['total']))
        print(label+': '+' / '.join(parts))
    print('\n기존 모델·결과와 최종 지도32개 보존. 실제 장비 검증 미실시. 자동 PPO 교체 없음.')


if __name__=='__main__':main()
