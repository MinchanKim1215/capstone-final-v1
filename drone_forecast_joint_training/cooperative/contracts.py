"""예측 가중치와 후처리 설정이 같은 동결 모델을 구성하는지 검사."""

import json
from swarm_sim.regions import ROOT,sha256_file


def verify_forecaster_files():
    contract=json.loads((ROOT/'configs/refinement_forecaster_integrity.json').read_text(encoding='utf-8'))
    expected=contract['files']
    if set(expected)!={'model/advance/selected/'+name for name in ['best.pt','calibration.json','alarm_policy.json','selection.json']}:
        raise ValueError('예측기 가중치·확률 보정·경보 설정 전체의 지문 필요')
    for name,digest in expected.items():
        if sha256_file(ROOT/name)!=digest:
            raise ValueError('동결 예측기 파일 변경: '+name)
    return expected
