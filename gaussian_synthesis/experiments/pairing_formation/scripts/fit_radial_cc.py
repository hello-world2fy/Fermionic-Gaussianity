#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.coupled_cluster import fit_cc
from pairing_formation.conditional import energy_resolved_observables


def job(task):
    t, rank, seed, output = task
    begin = time.monotonic()
    model = picket_fence(12, 2.)
    target = model.evolve([t])[0]
    report, T, state = fit_cc(model, target, rank, radial=True, starts=8, seed=seed, maxiter=1000)
    report.update(time=t, seed=seed, elapsed_seconds=time.monotonic()-begin,
                  observables=energy_resolved_observables(model, state))
    name = f't{t:g}_rank{rank}_seed{seed}'
    (output/(name+'.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/(name+'.npz'), target=target, T=T, state=state)
    return {k: report[k] for k in ('time', 'rank', 'seed', 'best_found_fidelity', 'elapsed_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(t, rank, seed, args.output) for t in (.5, 1., 2.15) for rank in (1, 2, 3, 6)
             for seed in (20261104, 20261105)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(job, tasks):
            print(json.dumps(result), flush=True)
