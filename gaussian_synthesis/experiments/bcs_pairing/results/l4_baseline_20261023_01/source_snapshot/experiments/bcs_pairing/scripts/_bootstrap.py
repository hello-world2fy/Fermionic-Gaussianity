"""Set bounded numerical threads and locate this experiment, not library sources."""
import os
from pathlib import Path
import sys

for name in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
