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
from pairing_formation.conditional import fit_conditional_products, energy_resolved_observables


def job(task):
    t, source, seed, output = task
    begin = time.monotonic()
    model = picket_fence(12, 2.)
    root = Path(__file__).resolve().parents[1]/'results'
    saved = np.load(root/f'replica2_L12_G2_01/t{t:g}.npz')
    target = saved[source]
    report, basis, parameters, approximation, preserved = fit_conditional_products(model, target, starts=12, seed=seed)
    report.update(time=t, L=12, G=2., source=source, seed=seed, elapsed_seconds=time.monotonic()-begin)
    normalized = target/np.linalg.norm(target)
    report['observables'] = {name: {'counting': model.observe(state), 'resolved': energy_resolved_observables(model, state)}
                             for name, state in [('target', normalized), ('fit', approximation), ('count_preserved_fit', preserved)]}
    name = f'{source}_t{t:g}_seed{seed}'
    (output/(name+'.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(output/(name+'.npz'), target=target, basis=basis, parameters=parameters,
                        approximation=approximation, count_preserved=preserved)
    return {k: report[k] for k in ('time', 'source', 'seed', 'fidelity', 'elapsed_seconds')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    tasks = [(t, source, seed, args.output) for t in (.5, 1., 2.15) for source in ('target', 'witness') for seed in (20261031, 20261101)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(job, tasks):
            print(json.dumps(result), flush=True)
