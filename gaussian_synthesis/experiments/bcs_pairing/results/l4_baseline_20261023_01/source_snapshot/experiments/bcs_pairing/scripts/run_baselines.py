#!/usr/bin/env python3
"""Explicit small pilot; saves exact states, fit histories and frozen acceptance."""
import _bootstrap
from _bootstrap import EXPERIMENT_ROOT, PROJECT_ROOT
import argparse
import json
from pathlib import Path
import signal
import traceback
import numpy as np
from gaussian_synthesis.solver import ResourceBudget, ResourceLimitError, peak_memory_mib
from bcs_pairing.baselines import run_baselines
from bcs_pairing.provenance import write_json, write_csv, snapshot, digest
from bcs_pairing.verification import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=EXPERIMENT_ROOT/'configs/l4_pilot.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full', action='store_true', help='Explicitly run the bounded L=4 pilot')
    args = parser.parse_args()
    if not args.full:
        parser.error('The experiment requires an explicit --full')
    config = json.loads(args.config.read_text())
    if not 0 < config['max_seconds'] <= 180 or not 0 < config['max_memory_mib'] <= 512:
        parser.error('Pilot hard limits are 180 seconds and 512 MiB')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', config)
    environment = snapshot(args.output, EXPERIMENT_ROOT, PROJECT_ROOT)
    budget = ResourceBudget(config['max_seconds'], config['max_memory_mib'])
    def timeout(*_):
        raise ResourceLimitError('L=4 pilot time budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, config['max_seconds'])
    try:
        arrays, quench, asym, records = run_baselines(config, budget, lambda s: print(s, flush=True))
        np.savez_compressed(args.output/'states.npz', **arrays)
        write_csv(args.output/'baselines.csv', quench)
        write_csv(args.output/'asymptotic.csv', asym)
        write_json(args.output/'optimization_starts.json', records)
        result = verify(args.output, check_hashes=False)
        write_json(args.output/'checks.json', result)
        write_json(args.output/'timings.json', {'elapsed_seconds': budget.elapsed, 'peak_memory_mib': peak_memory_mib()})
        artifacts = ['config.json', 'states.npz', 'baselines.csv', 'asymptotic.csv', 'optimization_starts.json',
                     'checks.json', 'timings.json', 'requirements-freeze.txt']
        write_json(args.output/'manifest.json', {'schema_version': 1, 'status': result['status'], 'environment': environment,
                                                'artifact_sha256': {name: digest(args.output/name) for name in artifacts},
                                                'scope': result['scope']})
        print(json.dumps({'status': result['status'], 'checks': len(result['checks']), 'elapsed_seconds': budget.elapsed}))
        return 0 if result['status'] == 'passed' else 1
    except Exception as error:
        write_json(args.output/'failure.json', {'status': 'resource_limit' if isinstance(error, ResourceLimitError) else 'error',
                                               'error': str(error), 'traceback': traceback.format_exc(), 'elapsed_seconds': budget.elapsed})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    raise SystemExit(main())
