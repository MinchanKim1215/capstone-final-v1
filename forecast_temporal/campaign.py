"""Finite native campaign with explicit failure and completion evidence."""
import os
import subprocess
import sys
import time
import traceback
from . import ROOT,ARTIFACTS
from .contracts import read,write,now

def run(module,label):
    logs=ARTIFACTS/'logs';logs.mkdir(exist_ok=True)
    with (logs/(label+'.stdout.log')).open('w',encoding='utf-8') as out,(logs/(label+'.stderr.log')).open('w',encoding='utf-8') as err:
        subprocess.run([sys.executable,'-u','-B','-m',module],cwd=ROOT.parent,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=out,stderr=err,check=True)

def main():
    try:
        began=time.monotonic()
        while not (ARTIFACTS/'data_status.json').exists():
            p=ARTIFACTS/'data.stderr.log'
            if p.exists() and 'Traceback (most recent call last)' in p.read_text(encoding='utf-8',errors='replace'):raise RuntimeError('Data construction failed')
            if time.monotonic()-began>3*3600:raise TimeoutError('Finite data timeout')
            time.sleep(5)
        if not read(ARTIFACTS/'preflight.json')['state']=='passed':raise ValueError('Preflight incomplete')
        run('forecast_temporal.train','training');run('forecast_temporal.compare','comparison');run('forecast_temporal.post','post')
    except BaseException as e:
        write(ARTIFACTS/'status.json',dict(state='failed',stage='campaign',error=repr(e),updated_at=now(),production_ready=False));traceback.print_exc();raise

if __name__=='__main__':main()
