"""V10: event diagnostics, packet-service labels and isolated ablations."""
from pathlib import Path
import os
import forecast_extension

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent
ARTIFACTS=ROOT/'artifacts'
ARTIFACTS.mkdir(parents=True,exist_ok=True)
VERSION='forecast-service-v10-20261102'
os.environ['NUMBA_CACHE_DIR']=str(ARTIFACTS/'numba_cache')
