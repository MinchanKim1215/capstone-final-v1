"""Process-scoped sleep inhibition, released when this experiment completes."""
import ctypes
import os
import time
from . import ARTIFACTS
from .contracts import read,write,now


def main():
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001));start=time.monotonic()
    write(ARTIFACTS/'power_guard.json',dict(pid=os.getpid(),started_at=now(),system_sleep_inhibited=held,display_sleep_allowed=True))
    try:
        while time.monotonic()-start<12*3600:
            file=ARTIFACTS/'status.json'
            if file.exists() and read(file)['state'] in ('completed','failed','stopped'):break
            time.sleep(5)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        write(ARTIFACTS/'power_guard.json',dict(state='released',ended_at=now(),system_sleep_inhibited=False))


if __name__=='__main__':main()
