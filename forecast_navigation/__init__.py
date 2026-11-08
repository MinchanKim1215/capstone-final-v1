"""Navigation-plan quality probe; V11 stays frozen and continues independently."""
from pathlib import Path
import forecast_graph
ROOT=Path(__file__).resolve().parent
ARTIFACTS=ROOT/'artifacts'
ARTIFACTS.mkdir(parents=True,exist_ok=True)

