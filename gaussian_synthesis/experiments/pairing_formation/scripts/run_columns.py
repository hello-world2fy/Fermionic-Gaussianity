#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import json
import argparse
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.columns import adaptive_columns


def job(task):
    t, seed, device, count, full_starts, output = task
    model = picket_fence(12, 2.)
    target = model.evolve([t])[0]
    name = f't{t:g}_seed{seed}'
    def checkpoint(history, atoms, parameters, matrices, coefficients):
        np.savez_compressed(output/f'{name}.npz', target=target, atoms=atoms, parameters=parameters,
                            qp_Z=matrices, coefficients=coefficients)
        (output/f'{name}.json').write_text(json.dumps({'L': 12, 'G': 2., 'time': t, 'seed': seed,
               'full_starts': full_starts, 'device': device, 'history': history,
               'global_optimum_certified': False, 'atom_normalization': 'full Fock space before symmetry projection'}, indent=2))
        if len(history) % 4 == 0:
            print(json.dumps({'time': t, 'seed': seed, **{k: history[-1][k] for k in
                 ('columns', 'fidelity', 'coefficient_mass', 'elapsed_seconds')}}), flush=True)
    adaptive_columns(model, target, count=count, seed=seed, device=f'cuda:{device}',
                     full_starts=full_starts, checkpoint=checkpoint)
    return name


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--times', type=float, nargs='+', default=[.5, 1., 2.15])
    p.add_argument('--seeds', type=int, nargs='+', default=[20261027, 20261028])
    p.add_argument('--columns', type=int, default=32)
    p.add_argument('--full-starts', type=int, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(t, seed, device, args.columns, args.full_starts, args.output)
             for device, (t, seed) in enumerate((t, seed) for t in args.times for seed in args.seeds)]
    if len(tasks) > 8:
        raise ValueError('This runner schedules at most one task per GPU')
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context('spawn')) as pool:
        for name in pool.map(job, tasks):
            print('finished '+name, flush=True)
