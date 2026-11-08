"""Read saved histories; write a self-contained visualization outside the experiment.

No training imports, model execution, CUDA calls, source edits, or locks on training files.
The pretraining checkpoint is read once on CPU solely to retrieve its validation history.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import os
import time

OPS = Path(__file__).resolve().parent
PROJECT = OPS.parent / "drone_forecast_joint_training"
RESULTS = PROJECT / "results/transformer"
OUTPUT = Path(r"C:\Users\mckim\.codex\visualizations\2026\10\28\01a183d6-b526-7313-baff-735240a78985\capstone-training-history.html")


def read(path):
    for attempt in range(4):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (PermissionError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(.05)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    state = read(OPS / "status.json")
    run = RESULTS / "train/transformer_main"
    pointer = read(run / "latest.json")
    committed = pointer["step"]
    history = [r for r in read(run / "history.json") if r["update"] <= committed]
    assert history and [r["update"] for r in history] == list(range(1, committed + 1))
    validations = [r for r in read(run / "validation_history.json") if r["update"] <= committed]
    pre_pointer = read(RESULTS / "pretrain/latest.json")
    # CPU deserialization only, and no dependency on the experiment's Python modules.
    import torch
    torch.set_num_threads(1)
    checkpoint = torch.load(PROJECT / pre_pointer["path"], map_location="cpu", weights_only=False)
    pre_validation = checkpoint["validations"]
    losses = read(RESULTS / "pretrain/history.json")
    bootstrap = read(RESULTS / "bootstrap/complete.json")
    ppo_keys = ["update", "CTR", "PDR", "longest_outage_s", "safety_passed", "episodes",
                "wall_s", "encoder_change_l2", "predictor_change_l2", "behavior_logp_max_error"]
    ppo = [{k: r[k] for k in ppo_keys} for r in history]
    supervised = [dict(step=r["step"], risk5=r["metrics"]["risk_5s"]["average_precision"],
                       onset37=r["metrics"]["onset_3to7s"]["average_precision"],
                       improved=r["improved"]) for r in pre_validation]
    # Contiguous 25-step means retain coverage of the full loss history without visual noise.
    grouped_losses = []
    for first in range(0, len(losses), 25):
        group = losses[first:first + 25]
        grouped_losses.append(dict(step=group[-1]["step"], loss=sum(r["loss"] for r in group) / len(group)))
    validation_rows = [dict(update=r["update"], CTR=r["score"][0], PDR=r["score"][1],
                            longest_outage_s=-r["score"][2], improved=r["improved"], feasible=r["feasible"])
                       for r in validations]
    data = dict(as_of=datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST"),
                recorded_state=state["state"], target=state["targets"], collected=bootstrap["train_episodes"],
                validation_episodes=bootstrap["validation_episodes"], pretrain_step=pre_pointer["step"],
                ppo=ppo, validation=validation_rows, pretrain=supervised, loss=grouped_losses,
                best_update=read(run / "best.json")["step"], training_episodes=sum(r["episodes"] for r in history),
                all_safety_passed=all(r["safety_passed"] for r in history),
                test_opened=(RESULTS / "test_access.json").exists())
    # Only replace a data marker in literal markup; no remote calls or background poller.
    template = (OPS / "training-history.template.html").read_text(encoding="utf-8")
    packed = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    assert template.count("__TRAINING_SNAPSHOT__") == 1
    output = template.replace("__TRAINING_SNAPSHOT__", packed)
    assert len(output.encode("utf-8")) < 1_000_000
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(output, encoding="utf-8", newline="\n")
    os.replace(temporary, OUTPUT)
    summary = dict(path=str(OUTPUT), as_of=data["as_of"], committed_updates=committed,
                   training_episodes=data["training_episodes"], supervised_validation_points=len(supervised),
                   ppo_validation_points=len(validation_rows), bytes=OUTPUT.stat().st_size,
                   experiment_files_written=0, cuda_used=False)
    (OPS / "visualization_snapshot.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
