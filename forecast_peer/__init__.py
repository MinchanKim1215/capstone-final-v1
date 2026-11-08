"""V15: direct observed per-peer histories; prospectively shared blind holdout."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ARTIFACTS=ROOT/'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
PARENT=ROOT.parent/'forecast_recall/artifacts'
GRAPH=ROOT.parent/'forecast_temporal/artifacts'
