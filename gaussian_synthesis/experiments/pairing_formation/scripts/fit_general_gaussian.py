#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import time
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.projected_gaussian import fit_general


def job(task):
    t, device, seed, fits, output = task
    start = time.monotonic()
    model = picket_fence(12, 2.)
    target = model.evolve([t])[0]
    onsite = json.loads((fits/f'pbcs_t{t:g}.json').read_text())['fit']['parameters']
    report, Z, projection = fit_general(model, target, onsite, starts=4, seed=seed, device=f'cuda:{device}')
    report.update({'L': 12, 'G': 2., 'time': t, 'seed': seed, 'device': device,
                   'wall_seconds': time.monotonic()-start})
    name = f't{t:g}_seed{seed}'
    (output/f'{name}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/f'{name}.npz', Z=Z, paired_projection=projection)
    return {k: report[k] for k in ('time', 'seed', 'best_found_fidelity', 'seniority_zero_probability', 'wall_seconds')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--onsite-fits', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(t, device, 20261025+rep, args.onsite_fits, args.output)
             for device, (t, rep) in enumerate((t, rep) for t in (.5, 1., 2.15) for rep in range(2))]
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(job, tasks):
            print(json.dumps(result), flush=True)
