"""Strict isolated Python 3.12 installer. Select exact CPU/CUDA wheels even after CPU installs."""
import argparse,os,platform,subprocess,sys,venv
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--torch',choices=['cpu','cu128'],default='cu128');a=p.parse_args()
    if sys.version_info[:2]!=(3,12) or sys.maxsize<=2**32:raise SystemExit('Use Python 3.12 64-bit: py -3.12 setup_environment.py --torch cu128')
    root=Path(__file__).resolve().parent;env=root/'.venv';venv.EnvBuilder(with_pip=True).create(env)
    python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    def run(*args):subprocess.run([str(python),*args],check=True,cwd=root)
    run('-m','pip','install','--upgrade','pip')
    if platform.system()=='Darwin':
        if a.torch!='cpu':raise SystemExit('Use --torch cpu on macOS; CUDA requires NVIDIA hardware')
        run('-m','pip','install','torch==2.9.1')
    else:run('-m','pip','install','torch==2.9.1+'+a.torch,'--index-url','https://download.pytorch.org/whl/'+a.torch)
    run('-m','pip','install','--only-binary=:all:','-r','requirements-validated.txt')
    run('transformer_run.py','doctor');run('transformer_run.py','selftest')
    if a.torch!='cpu':run('transformer_run.py','cuda-check','--device','cuda')
    print('INSTALLATION VERIFIED. Run 02_train_transformer.bat or see HOME_TRAINING_GUIDE.md.')


if __name__=='__main__':main()
