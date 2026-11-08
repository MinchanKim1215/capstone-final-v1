"""Event recall improvement; preserve all previous deliveries."""
from pathlib import Path
import os
import forecast_navigation
ROOT=Path(__file__).resolve().parent
ARTIFACTS=ROOT/'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
os.environ['NUMBA_CACHE_DIR']=str(ARTIFACTS/'numba_cache')
PARENT=forecast_navigation.ARTIFACTS
