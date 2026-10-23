import os
from pathlib import Path
import sys
import unittest

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
for path in (ROOT,ROOT.parent/'fk_two_cluster',ROOT.parent/'fk_two_cluster_followup'):
    sys.path.insert(0,str(path))
suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
