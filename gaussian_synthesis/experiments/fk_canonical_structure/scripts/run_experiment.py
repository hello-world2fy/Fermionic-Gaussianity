"""Run one new, non-overwriting canonical-structure experiment."""
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT.parent/'fk_two_cluster',ROOT.parent/'fk_two_cluster_followup'):
    sys.path.insert(0,str(path))
from fk3.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
