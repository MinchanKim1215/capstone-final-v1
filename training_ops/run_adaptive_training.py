"""Detached local launcher. Does not modify frozen v6 sources or settings."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import ctypes
import argparse
import json
import msvcrt
import os
import subprocess
import sys
import tempfile

WORKSPACE = Path(__file__).resolve().parent.parent
ARTIFACTS = WORKSPACE / 'adaptive_joint' / 'artifacts'
LAUNCH = ARTIFACTS / 'launcher'
RUN = 'adaptive_v6_20261030'
PYTHON = WORKSPACE / 'drone_forecast_joint_training/.venv/Scripts/python.exe'


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.stem + '_', suffix='.tmp', delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
        temporary = f.name
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    options = parser.parse_args()
    LAUNCH.mkdir(parents=True, exist_ok=True)
    with (LAUNCH / 'supervisor.lock').open('a+b') as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        run_folder = ARTIFACTS / 'runs' / RUN
        if run_folder.exists() and not options.resume:
            raise RuntimeError('Run already exists. Inspect it and use the documented resume command.')
        if options.resume and not (run_folder / 'latest.json').exists():
            raise RuntimeError('No committed checkpoint exists for resume.')
        stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
        stdout = LAUNCH / (stamp + '.stdout.log')
        stderr = LAUNCH / (stamp + '.stderr.log')
        record = LAUNCH / (stamp + '.launch.json')
        args = [str(PYTHON), '-u', '-B', '-m', 'adaptive_joint.run', 'train', '--run', RUN]
        if options.resume:
            args.append('--resume')
        state = dict(state='starting', run=RUN, started_at=now(), supervisor_pid=os.getpid(),
                     command=args, stdout=str(stdout), stderr=str(stderr),
                     automatic_retries=False, holdout_test_requested=False, resumed=options.resume)
        if options.resume:
            state['resumed_checkpoint'] = json.loads((run_folder / 'latest.json').read_text(encoding='utf-8'))
            previous_path = LAUNCH / 'status.json'
            if previous_path.exists():
                previous = json.loads(previous_path.read_text(encoding='utf-8'))
                state['original_started_at'] = previous.get('original_started_at', previous['started_at'])

        def save():
            write_json(record, state)
            write_json(LAUNCH / 'status.json', state)

        api = ctypes.WinDLL('kernel32', use_last_error=True).SetThreadExecutionState
        api.argtypes = [ctypes.c_uint32]
        api.restype = ctypes.c_uint32
        state['system_sleep_inhibited'] = bool(api(0x80000001))
        state['display_sleep_allowed'] = True
        try:
            save()
            env = os.environ.copy()
            env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                       PYTHONDONTWRITEBYTECODE='1')
            with stdout.open('ab', buffering=0) as out, stderr.open('ab', buffering=0) as err:
                child = subprocess.Popen(args, cwd=WORKSPACE, env=env, stdin=subprocess.DEVNULL,
                                         stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
                state.update(state='running', training_pid=child.pid)
                save()
                code = child.wait()
                state.update(state='completed' if code == 0 else 'failed', exit_code=code, ended_at=now())
                save()
                return code
        except BaseException as error:
            state.update(state='failed', error=repr(error), ended_at=now())
            save()
            raise
        finally:
            api(0x80000000)


if __name__ == '__main__':
    sys.exit(main())
