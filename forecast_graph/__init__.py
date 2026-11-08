"""V11 isolated, causal shared-plan temporal graph and hybrid forecast study."""
from pathlib import Path
import os
import forecast_extension

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ARTIFACTS = ROOT / 'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
VERSION = 'forecast-graph-v11-20261102'
os.environ['NUMBA_CACHE_DIR'] = str(ARTIFACTS / 'numba_cache')

