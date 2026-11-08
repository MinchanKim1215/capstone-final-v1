"""Finite follow-up runner; guard releases on recorded completion or failure."""
import os
import subprocess
import sys
import time
import traceback
from . import ARTIFACTS,ROOT
from .contracts import read,write,now,status


def run(module,label):
    logs=ARTIFACTS/'logs';logs.mkdir(exist_ok=True)
    with (logs/(label+'.stdout.log')).open('w',encoding='utf-8') as a,(logs/(label+'.stderr.log')).open('w',encoding='utf-8') as b:
        subprocess.run([sys.executable,'-u','-B','-m',module],cwd=ROOT.parent,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=a,stderr=b,check=True)


def main():
    try:
        from .compare import specification
        specification();started=time.monotonic()
        while not (ARTIFACTS/'data_status.json').exists():
            for name in ('collect.stderr.log','data.stderr.log'):
                path=ARTIFACTS/name
                if path.exists() and 'Traceback (most recent call last)' in path.read_text(encoding='utf-8',errors='replace'):raise RuntimeError('Preparation failed: '+name)
            if time.monotonic()-started>3*3600:raise TimeoutError('Finite preparation wait exceeded')
            time.sleep(5)
        run('forecast_navigation.train','training');run('forecast_navigation.compare','comparison');run('forecast_navigation.post','post')
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',stage='campaign',error=repr(error),updated_at=now(),production_ready=False));traceback.print_exc();raise


if __name__=='__main__':main()
