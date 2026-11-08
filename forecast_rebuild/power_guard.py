"""Temporary process-scoped sleep inhibition; no persistent power settings."""
import ctypes
import os
import time
from . import ARTIFACTS
from .contracts import read,write,now


def main():
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    started=time.monotonic()
    write(ARTIFACTS/'power_guard.json',dict(pid=os.getpid(),started_at=now(),system_sleep_inhibited=held,display_sleep_allowed=True))
    try:
        while time.monotonic()-started<12*3600:
            state=ARTIFACTS/'research_status.json'
            if state.exists() and read(state).get('state') in ('completed','stopped','failed'):break
            time.sleep(5)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        write(ARTIFACTS/'power_guard.json',dict(state='released',ended_at=now(),system_sleep_inhibited=False,display_sleep_allowed=True))


if __name__=='__main__':main()
