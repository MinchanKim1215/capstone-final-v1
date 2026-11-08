"""Local process supervisor; does not change experiment code or configuration."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import msvcrt
import os
import subprocess
import sys

OPS = Path(__file__).resolve().parent
PROJECT = OPS.parent / "drone_forecast_joint_training"
PYTHON = PROJECT / ".venv/Scripts/python.exe"
ARGS = [str(PYTHON), "-u", "transformer_run.py", "pipeline", "--run", "transformer_main",
        "--episodes", "192", "--pretrain-steps", "3000", "--updates", "256",
        "--workers", "4", "--device", "cuda"]
KST = timezone(timedelta(hours=9))


def now():
    return datetime.now(KST).isoformat(timespec="seconds")


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    OPS.mkdir(exist_ok=True)
    with (OPS / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("A local training supervisor is already running.")
        logs = OPS / "logs"
        logs.mkdir(exist_ok=True)
        stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        stdout_path = logs / (stamp + ".stdout.log")
        stderr_path = logs / (stamp + ".stderr.log")
        record_path = logs / (stamp + ".run.json")
        state = dict(state="starting", started_at=now(), supervisor_pid=os.getpid(),
                     project=str(PROJECT), command=ARGS, stdout=str(stdout_path), stderr=str(stderr_path),
                     targets=dict(episodes=192, pretrain_steps=3000, updates=256, workers=4, device="cuda"),
                     holdout_test_requested=False)
        env = os.environ.copy()
        env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            with stdout_path.open("ab", buffering=0) as out, stderr_path.open("ab", buffering=0) as err:
                child = subprocess.Popen(ARGS, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=out, stderr=err)
                state.update(state="running", training_pid=child.pid)
                write_json(OPS / "status.json", state)
                write_json(record_path, state)
                code = child.wait()
                state.update(state="completed" if code == 0 else "failed", exit_code=code, ended_at=now())
        except Exception as exc:
            state.update(state="failed", error=repr(exc), ended_at=now())
            write_json(OPS / "status.json", state)
            write_json(record_path, state)
            raise
        write_json(OPS / "status.json", state)
        write_json(record_path, state)
        return state["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
