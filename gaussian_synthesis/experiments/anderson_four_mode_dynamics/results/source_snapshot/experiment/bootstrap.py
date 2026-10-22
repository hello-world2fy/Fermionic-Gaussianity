"""Use the existing repository implementation without modifying it."""
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
sys.path.insert(0, str(REPOSITORY/'gaussian_synthesis'/'src'))
sys.path.insert(0, str(REPOSITORY/'gaussian_synthesis'/'experiments'/'four_mode_landscape'))
