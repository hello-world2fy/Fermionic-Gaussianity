#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.stochastic import torch_run


def job(args):
    L, G, kind, seed, dt, device, samples, output = args
    start = time.monotonic()
    model = picket_fence(L, G)
    report, blocks = torch_run(model, count=samples, kind=kind, seed=seed,
                               times=np.linspace(0, 3., 61), dt=dt, device=device)
    report.update({'L': L, 'G': G, 'wall_seconds': time.monotonic()-start})
    name = f'{kind}_seed{seed}_dt{dt:g}'
    (output/f'{name}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/f'{name}.npz', blocks=blocks)
    return {k: report[k] for k in ('method', 'seed', 'dt', 'device', 'samples', 'wall_seconds',
                                  'max_final_spin_length_squared_drift', 'max_final_number_drift')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--levels', type=int, default=12)
    p.add_argument('--G', type=float, default=2.)
    p.add_argument('--samples', type=int, default=65536)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(args.levels, args.G, kind, 20261025+seed, dt, device, args.samples, args.output)
             for device, (kind, seed, dt) in enumerate((kind, seed, dt)
             for kind in ('gaussian', 'discrete') for seed in range(2) for dt in (.01, .005))]
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(job, tasks):
            print(json.dumps(result), flush=True)
