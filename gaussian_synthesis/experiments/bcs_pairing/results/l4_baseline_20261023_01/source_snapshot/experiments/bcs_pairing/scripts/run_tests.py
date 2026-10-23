#!/usr/bin/env python3
"""Run only structural BCS tests, with a 60 second watchdog."""
import _bootstrap
from _bootstrap import EXPERIMENT_ROOT
import argparse
import json
import signal
import unittest
from gaussian_synthesis.solver import ResourceBudget, ResourceLimitError, peak_memory_mib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    budget = ResourceBudget(60, 512)
    def timeout(*_):
        raise ResourceLimitError('BCS unit-test time budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        suite = unittest.defaultTestLoader.discover(str(EXPERIMENT_ROOT/'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        budget.check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    summary = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'success': result.wasSuccessful(), 'elapsed_seconds': budget.elapsed,
               'peak_memory_mib': peak_memory_mib()}
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
