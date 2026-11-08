"""Finite staged execution; no training/confirmation overlap or reselection."""
import subprocess
import sys
import time
from . import ARTIFACTS,ROOT
from .contracts import read,write,now,status


def run(module,label):
    folder=ARTIFACTS/'logs';folder.mkdir(exist_ok=True)
    with (folder/(label+'.stdout.log')).open('a',encoding='utf-8') as out,(folder/(label+'.stderr.log')).open('a',encoding='utf-8') as err:
        subprocess.run([sys.executable,'-u','-B','-m',module],cwd=ROOT.parent,stdout=out,stderr=err,
                       creationflags=subprocess.CREATE_NO_WINDOW,check=True)


def main():
    while not (ARTIFACTS/'samples/manifest.json').exists():
        if read(ARTIFACTS/'status.json').get('state')=='failed':raise RuntimeError('Preparation failed; inspect replay/data logs')
        time.sleep(5)
    run('forecast_service.train','training')
    # Fresh children import comparison/audit code only after training ends.
    run('forecast_service.compare','comparison')
    run('forecast_service.post','post')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ARTIFACTS/'status.json',dict(state='failed',stage='campaign',error=repr(error),updated_at=now()));raise
