#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.multiorbit import MultiOrbitDynamics, fit_orbits


def job(task):
    t, source, seed, output, maxiter = task
    begin = time.monotonic()
    model = picket_fence(12, 2.)
    root = Path(__file__).resolve().parents[1]/'results'
    saved = np.load(root/f'replica2_L12_G2_01/t{t:g}.npz')
    report, coordinates = fit_orbits(model, saved[source], rank=2, starts=6, seed=seed, maxiter=maxiter)
    dynamics = MultiOrbitDynamics(model, 2)
    report.update(time=t, source=source, seed=seed, maxiter=maxiter, elapsed_seconds=time.monotonic()-begin)
    name = f'{source}_t{t:g}_seed{seed}'
    np.savez_compressed(output/(name+'.npz'), coordinates=coordinates, approximation=dynamics.state(coordinates),
                        target=saved[source]/np.linalg.norm(saved[source]))
    (output/(name+'.json')).write_text(json.dumps(report, indent=2))
    return {k: report[k] for k in ('time', 'source', 'seed', 'fidelity', 'elapsed_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--times', type=float, nargs='+', default=[.5, 1., 2.15])
    parser.add_argument('--maxiter', type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(t, source, seed, args.output, args.maxiter) for t in args.times for source in ('target', 'witness') for seed in (20261111, 20261112)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job, task) for task in tasks]):
            print(json.dumps(future.result()), flush=True)
