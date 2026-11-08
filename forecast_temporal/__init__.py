"""V14: end-to-end node/edge memory; immutable predecessors."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ARTIFACTS=ROOT/'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
PARENT=ROOT.parent/'forecast_recall/artifacts'
NAV=ROOT.parent/'forecast_navigation/artifacts'
