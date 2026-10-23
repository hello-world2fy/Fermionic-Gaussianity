"""Run the explicitly requested, bounded FK pilot."""
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fk.runner import main
if __name__ == '__main__':
    raise SystemExit(main())
