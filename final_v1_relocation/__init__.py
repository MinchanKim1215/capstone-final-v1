"""Final V1 재배치 개선: 예측 오류와 단절 상황 대응 (개선안 1~4).

원본 Final V1 소스(final_v1, forecast_avoidance, forecast_prevention, forecast_learning)는
SHA-256으로 잠겨 있으므로 수정하지 않는다. 이 패키지는 transformer_upgrade/bridge.py와 같은
방식으로 final_v1.runner의 재배치 판단(choose)과 명령 채널(CommandChannel)을 실행 중에만
교체한다. 모든 개선안을 끈 설정(arm "v21")은 Final V1의 V21과 같은 결정을 낸다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
VERSION = "Final V1 relocation upgrade (cut recovery, reconnect fallback, trigger, bypass)"
