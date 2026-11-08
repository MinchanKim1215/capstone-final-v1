"""Full-mission study of half-second preventative relay control."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
LEGACY = WORKSPACE / 'drone_forecast_joint_training'
ARTIFACTS = ROOT / 'artifacts'
VERSION = 'Final V1 half-second study'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
