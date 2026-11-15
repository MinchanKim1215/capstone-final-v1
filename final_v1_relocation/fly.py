"""개선안 설정 하나로 비행 1회를 실행한다.

  python -m final_v1_relocation.fly --arm r1_cut_recovery
  python -m final_v1_relocation.fly --arm v21 --command-loss reconnect --steps 24
"""
import argparse
import json

from .campaign import specs
from .evaluation import commands, outage_profile
from .options import ARMS, FALLBACKS
from .runner import CONTROLLERS, initialize, run


def main():
    parser = argparse.ArgumentParser(description="Final V1 재배치 개선안 비행")
    parser.add_argument("--arm", choices=tuple(ARMS), default="r1_cut_recovery")
    parser.add_argument("--command-loss", choices=FALLBACKS, help="명령 손실 모델과 미수신 대응")
    parser.add_argument("--controller", choices=CONTROLLERS, default="v21")
    parser.add_argument("--region")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--kind", choices=("clear", "intermittent", "continuous"))
    parser.add_argument("--direction", choices=("east_west", "north_south"))
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    spec = specs()[0][0].copy()
    for key in spec:
        value = getattr(args, key)
        if value is not None:
            spec[key] = value
    initialize(args.controller)
    row, _ = run(spec, args.arm, transport=args.command_loss, max_steps=args.steps, controller=args.controller)
    summary = {k: row[k] for k in ("spec", "relocation_arm", "transport", "duration_s", "mission_completed",
                                   "persistent_onsets", "persistent_cut_seconds", "timely_pdr",
                                   "decision_counts", "v18_reasons", "relocation_counts")}
    summary.update(outage_profile([row]), **{k: v for k, v in commands([row]).items()
                                             if k.endswith("_commands") or k == "recovery_while_actually_down"})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
