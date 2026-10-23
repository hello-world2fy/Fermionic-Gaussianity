#!/usr/bin/env python3
"""Serial, resource-bounded unit tests. Does not run the full experiment."""

import os
for _name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import json
from pathlib import Path
import signal
import sys
import time
import unittest

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"examples")]
from gaussian_synthesis.solver import ResourceBudget, ResourceLimitError, peak_memory_mib

BUDGET = ResourceBudget(30, 256, STARTED)


class BoundedTestResult(unittest.TextTestResult):
    def startTest(self, test):
        BUDGET.check()
        super().startTest(test)

    def stopTest(self, test):
        super().stopTest(test)
        BUDGET.check()


def timeout(signum, frame):
    raise ResourceLimitError("30-second unit-test watchdog fired")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 30-BUDGET.elapsed))
    success = False
    try:
        suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"))
        result = unittest.TextTestRunner(verbosity=2, resultclass=BoundedTestResult).run(suite)
        success = result.wasSuccessful()
        metadata = {"tests_run": result.testsRun, "failures": len(result.failures),
                    "errors": len(result.errors), "success": success}
    except ResourceLimitError as error:
        metadata = {"success": False, "resource_error": str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    metadata.update(elapsed_seconds=BUDGET.elapsed, peak_memory_mib=peak_memory_mib())
    print(json.dumps(metadata, indent=2))
    raise SystemExit(0 if success else 1)
