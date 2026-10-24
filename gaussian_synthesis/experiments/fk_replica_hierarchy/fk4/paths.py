"""Explicit paths for the read-only predecessor experiment packages."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
for folder in ("fk_two_cluster", "fk_two_cluster_followup", "fk_canonical_structure"):
    path = str(EXPERIMENTS / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

