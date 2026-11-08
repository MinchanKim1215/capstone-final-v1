"""Finite native campaign; do not terminate at job launch."""
import os
import subprocess
import sys
import time
import traceback
from . import ROOT,ARTIFACTS
from .contracts import read,write,now,status


def run(module,label):
    logs=ARTIFACTS/'logs';logs.mkdir(exist_ok=True)
    with (logs/(label+'.stdout.log')).open('w',encoding='utf-8') as out,(logs/(label+'.stderr.log')).open('w',encoding='utf-8') as err:
        subprocess.run([sys.executable,'-u','-B','-m',module],cwd=ROOT.parent,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=out,stderr=err,check=True)


def main():
    try:
        began=time.monotonic()
        while not (ARTIFACTS/'data_status.json').exists():
            error=ARTIFACTS/'data.stderr.log'
            if error.exists() and 'Traceback (most recent call last)' in error.read_text(encoding='utf-8',errors='replace'):raise RuntimeError('Data construction failed')
            if time.monotonic()-began>3*3600:raise TimeoutError('Data preparation finite timeout')
            time.sleep(5)
        run('forecast_recall.preflight','preflight');run('forecast_recall.train','training');run('forecast_recall.compare','comparison');run('forecast_recall.post','post')
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',stage='campaign',error=repr(error),updated_at=now(),production_ready=False));traceback.print_exc();raise


if __name__=='__main__':main()
