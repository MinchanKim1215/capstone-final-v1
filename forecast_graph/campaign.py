"""Finite native pipeline; a failure is recorded and never treated as completion."""
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from . import ARTIFACTS,WORKSPACE
from .contracts import read,write,now,status


def run(module,label):
    folder=ARTIFACTS/'logs';folder.mkdir(exist_ok=True)
    with (folder/(label+'.stdout.log')).open('w',encoding='utf-8') as out,(folder/(label+'.stderr.log')).open('w',encoding='utf-8') as err:
        subprocess.run([sys.executable,'-u','-B','-m',module],cwd=WORKSPACE,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=out,stderr=err,check=True)


def main():
    try:
        started=time.monotonic()
        while not (ARTIFACTS/'data_status.json').exists():
            for filename in ('trace.stderr.log','data.stderr.log'):
                path=ARTIFACTS/filename
                if path.exists() and 'Traceback (most recent call last)' in path.read_text(encoding='utf-8',errors='replace'):
                    raise RuntimeError('Preparation failed: '+filename)
            if time.monotonic()-started>3*3600:raise TimeoutError('Data preparation exceeded finite3h wait')
            time.sleep(5)
        run('forecast_graph.train','neural_training')
        status('tree_control_training');run('forecast_graph.tree','tree_training')
        run('forecast_graph.compare','comparison')
        run('forecast_graph.post','post')
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',stage='campaign',error=repr(error),updated_at=now(),production_ready=False))
        traceback.print_exc();raise


if __name__=='__main__':main()
