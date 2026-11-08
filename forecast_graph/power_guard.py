"""Release temporary sleep inhibition when this finite study finishes."""
import ctypes
import time
from . import ARTIFACTS
from .contracts import read,write,now


def main():
    held=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    write(ARTIFACTS/'power_guard.json',dict(state='holding',system_sleep_inhibited=held,started_at=now()))
    try:
        for k in range(12*3600//5):
            p=ARTIFACTS/'status.json'
            if p.exists() and read(p).get('state') in ('completed','failed','stopped'):break
            time.sleep(5)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        write(ARTIFACTS/'power_guard.json',dict(state='released',system_sleep_inhibited=False,ended_at=now()))


if __name__=='__main__':main()
