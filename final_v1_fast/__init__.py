"""Final V1 isolated command-cadence and 10 Hz observation engineering study."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent
ARTIFACTS=ROOT/'artifacts'
VERSION='Final V1 fast control study 20261105'
sys.dont_write_bytecode=True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')
from adaptive_joint_expanded import LEGACY

