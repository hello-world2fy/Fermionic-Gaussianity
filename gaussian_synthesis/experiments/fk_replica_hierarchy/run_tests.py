from pathlib import Path
import os
import sys
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent/"tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
