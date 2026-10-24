from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT.parent/"fk_replica_hierarchy"
sys.path.insert(0,str(PREVIOUS))
from fk4 import paths as previous_paths
