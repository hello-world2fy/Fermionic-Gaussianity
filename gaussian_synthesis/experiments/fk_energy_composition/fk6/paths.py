from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT.parent/"fk_optimal_geometry"
sys.path.insert(0,str(PREVIOUS))
from fk5 import paths as previous_paths
