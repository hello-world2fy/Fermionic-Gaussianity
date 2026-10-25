#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import json
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.costs import repair_minimum_mass, matched_magic_witness


def job(task):
    t, seed, output = task
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    data = np.load(root/f'adaptive_full_L12_G2_01/t{t:g}_seed{seed}.npz')
    upper, c, residual, dual = repair_minimum_mass(data['atoms'], data['target'])
    lower, W = matched_magic_witness(model, data['target'])
    report = {'time': t, 'seed': seed, 'upper': upper, 'lower': lower}
    name = f't{t:g}_seed{seed}'
    (output/f'{name}.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/f'{name}.npz', coefficients=c, fock_repair=residual, restricted_dual=dual, analytic_support_witness=W)
    return {'time': t, 'seed': seed, 'mass_upper': upper['mass'], 'extent_upper': upper['extent_upper_numerical'],
            'extent_lower': lower['extent_lower_numerical']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(job, [(t, seed, args.output) for t in (.5, 1., 2.15) for seed in (20261027, 20261028)]):
            print(json.dumps(result), flush=True)
