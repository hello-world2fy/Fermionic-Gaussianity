#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.coupled_cluster import fit_cc


def job(task):
    t, rank, output = task
    model = picket_fence(12, 2.)
    state = model.evolve([t])[0]
    report, T, fitted = fit_cc(model, state, rank, starts=6, seed=20261025+rank)
    report.update({'L': 12, 'G': 2., 'time': t, 'fit_observables': model.observe(fitted),
                   'exact_observables': model.observe(state)})
    (output/f't{t:g}_rank{rank}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/f't{t:g}_rank{rank}.npz', T=T, state=fitted)
    return {k: report[k] for k in ('time', 'rank', 'best_found_fidelity')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(t, rank, args.output) for t in (.5, 1., 2.15) for rank in (1, 2, 6)]
    with ProcessPoolExecutor(max_workers=9) as pool:
        for result in pool.map(job, tasks):
            print(json.dumps(result), flush=True)
