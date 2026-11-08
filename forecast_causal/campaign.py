"""Four user-requested stages with saved evidence, errors and resume boundaries."""
import os
import subprocess
import sys
import time
import traceback
from . import ROOT,ARTIFACTS
from .contracts import read,write,now,status

def run(module,label,*extra):
    logs=ARTIFACTS/'logs';logs.mkdir(exist_ok=True)
    with (logs/(label+'.stdout.log')).open('w',encoding='utf-8') as out,(logs/(label+'.stderr.log')).open('w',encoding='utf-8') as err:
        subprocess.run([sys.executable,'-u','-B','-m',module,*extra],cwd=ROOT.parent,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=out,stderr=err,check=True)

def main():
    from .delivery import already_sealed
    if already_sealed():return
    try:
        began=time.monotonic()
        while not (ARTIFACTS/'selection_replay_status.json').exists():
            p=ARTIFACTS/'selection_replay.stderr.log'
            if p.exists() and 'Traceback (most recent call last)' in p.read_text(encoding='utf-8',errors='replace'):raise RuntimeError('Selection reconstruction failed')
            if time.monotonic()-began>2*3600:raise TimeoutError('Selection reconstruction timeout')
            time.sleep(5)
        run('forecast_causal.causes','selection_causes')
        run('forecast_causal.replay','development_replay','--scope','all')
        run('forecast_causal.data','data')
        run('forecast_causal.input_preflight','input_preflight')
        run('forecast_causal.train','training')
        run('forecast_causal.compare','comparison')
        run('forecast_causal.post','post')
    except BaseException as error:
        value=dict(state='failed',stage='campaign',error=repr(error),updated_at=now(),production_ready=False);write(ARTIFACTS/'status.json',value)
        from .ledger import refresh
        refresh(value);traceback.print_exc();raise

if __name__=='__main__':main()
