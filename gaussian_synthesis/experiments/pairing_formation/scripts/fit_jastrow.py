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
from pairing_formation.fitting import excitation_features
from pairing_formation.jastrow import JastrowDynamics, fit_jastrow


def job(task):
    t, family, seed, output = task
    begin = time.monotonic()
    model = picket_fence(12, 2.)
    root = Path(__file__).resolve().parents[1]/'results'
    target = np.load(root/f'replica2_L12_G2_01/t{t:g}.npz')['target']
    saved = np.load(root/f'conditional_orbitals_01/target_t{t:g}_seed20261031.npz')
    parameters = saved['parameters']
    z = parameters[:, :model.L]+1j*parameters[:, model.L:]
    raw = np.sum(excitation_features(model)*z[model.m], axis=1)
    dynamics = JastrowDynamics(model, family)
    report, coordinates, approximation = fit_jastrow(dynamics, target, raw, starts=8, seed=seed)
    report.update(time=t, L=12, G=2., seed=seed, elapsed_seconds=time.monotonic()-begin)
    name = f'{family}_t{t:g}_seed{seed}'
    np.savez_compressed(output/(name+'.npz'), coordinates=coordinates, features=dynamics.features,
                        approximation=approximation, target=target)
    (output/(name+'.json')).write_text(json.dumps(report, indent=2))
    return {k: report[k] for k in ('time', 'family', 'seed', 'fidelity', 'complex_cone_dimension', 'elapsed_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--families', nargs='+', default=['collective_0', 'collective_1', 'collective_2', 'two_body'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(t, family, seed, args.output) for t in (.5, 1., 2.15)
             for family in args.families
             for seed in (20261107, 20261108)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job, task) for task in tasks]):
            print(json.dumps(future.result()), flush=True)
