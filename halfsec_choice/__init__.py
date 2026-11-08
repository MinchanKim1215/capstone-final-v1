"""Supervised relay choice for the half-second mission simulator."""
from pathlib import Path

from fast_halfsec import ARTIFACTS as PRIOR

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'
FORECAST = PRIOR / 'models/cutoff_v2/forecast_profile.json'
