#!/usr/bin/env python3
"""Run experiment-specific unit tests; never run the full scans."""
import _bootstrap
from _bootstrap import EXPERIMENT_ROOT
import json
import signal
import time
import unittest
from gaussian_synthesis.solver import ResourceBudget, ResourceLimitError, peak_memory_mib

if __name__ == "__main__":
    budget = ResourceBudget(30, 256)
    def timeout(signum, frame):
        raise ResourceLimitError("30-second experiment-test watchdog fired")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        suite = unittest.defaultTestLoader.discover(str(EXPERIMENT_ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        budget.check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"tests_run": result.testsRun, "success": result.wasSuccessful(),
                      "elapsed_seconds": budget.elapsed, "peak_memory_mib": peak_memory_mib()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
