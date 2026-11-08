"""V16: measured packet-state precursors and cause-grounded forecasting."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ARTIFACTS=ROOT/'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
WORKSPACE=ROOT.parent
PARENT=WORKSPACE/'forecast_recall/artifacts'
NAV=WORKSPACE/'forecast_navigation/artifacts'
