"""Run the selected Final V1 research controller."""
import argparse
import json

from . import ARTIFACTS
from .protocol import KINDS, read, sha
from .runner import initialize, run
from .training import read_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Final V1 research simulator")
    parser.add_argument("--region")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--direction", choices=("east_west","north_south"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--command-loss", action="store_true")
    parser.add_argument("--controller", choices=("v21", "v21_transformer"))
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    release_path = ARTIFACTS/"release.json"
    verified = read(ARTIFACTS/"completed.json")
    if sha(release_path) != verified["release_sha256"]:
        raise ValueError("Final V1 release has not passed delivery verification")
    release = read(release_path)
    spec = read(ARTIFACTS/"protocol.json")["confirmation"][0].copy()
    for name in spec:
        value = getattr(args, name)
        if value is not None:
            spec[name] = value
    transport = release["transport_fallback"] if args.command_loss else None
    extension = ARTIFACTS.parent / "transformer_upgrade/artifacts/integration.json"
    selected = args.controller
    if selected is None and extension.is_file() and read(extension)["default"]:
        selected = "v21_transformer"
    if selected == "v21_transformer":
        if not extension.is_file():
            parser.error("The Transformer upgrade needs a completed evaluation")
        from .transformer_upgrade.bridge import run as run_upgrade
        from .transformer_upgrade.release import installed

        payload, profile = installed()
        fleet = profile["thresholds"]["fleet"]
        row, _ = run_upgrade(spec, selected, payload, fleet["threshold"],
                             transport=transport, max_steps=args.steps, network_weight=fleet["network_weight"])
    elif selected == "v21" and extension.is_file():
        from .transformer_upgrade.bridge import run as run_upgrade

        row, _ = run_upgrade(spec, selected, transport=transport, max_steps=args.steps)
    else:
        state = None
        model = release["model"] if selected is None else None
        if model is not None:
            if sha(model["path"]) != model["sha256"]:
                raise ValueError("Released PPO adapter changed")
            state = read_checkpoint(model["path"])["state"]
        initialize()
        row, _ = run(spec, selected or release["controller"], state_dict=state,
                     transport=transport, max_steps=args.steps)
    result = {k:row[k] for k in ("spec","arm","transport","deployment","duration_s","mission_completed",
                                  "persistent_onsets","persistent_cut_seconds","timely_pdr","audited")}
    if "upgrade_arm" in row:
        result["controller"] = row["upgrade_arm"]
    print(json.dumps(dict(version=release["version"],**result),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
