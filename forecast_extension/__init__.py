"""V9: separate continuation, new-experience training and fresh-flight audit."""
from pathlib import Path
import os
import forecast_rebuild

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent
ARTIFACTS=ROOT/'artifacts'
ARTIFACTS.mkdir(parents=True,exist_ok=True)
VERSION='forecast-extension-v9-20261102'
os.environ['NUMBA_CACHE_DIR']=str(ARTIFACTS/'numba_cache')

