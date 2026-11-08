"""Isolated v6 study: pre-mission deployment selection + 1 s forecasting + PPO.

The v5 project is a read-only dependency. No legacy file, checkpoint, split, or
integrity contract is modified. All writes belong to adaptive_joint/artifacts.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'drone_forecast_joint_training'
ARTIFACTS = ROOT / 'artifacts'
sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ['NUMBA_CACHE_DIR'] = str(ARTIFACTS / 'numba_cache')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

VERSION = 'adaptive-joint-next-event-v6'
