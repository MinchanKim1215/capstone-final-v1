"""Independent prediction-first study; earlier campaigns are read-only inputs."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ARTIFACTS = ROOT / 'artifacts'
LEGACY = WORKSPACE / 'drone_forecast_joint_training'
VERSION = 'forecast-rebuild-v8-20261102'
sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['NUMBA_CACHE_DIR'] = str(ARTIFACTS / 'numba_cache')
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
