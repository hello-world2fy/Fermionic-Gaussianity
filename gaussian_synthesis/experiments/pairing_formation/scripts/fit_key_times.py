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
from pairing_formation.fitting import fit_product_family


def job(task):
    L, G, t, family, starts = task
    model = picket_fence(L, G)
    state = model.evolve([t])[0]
    result, fit = fit_product_family(model, state, radial=family != 'pbcs', slope=family == 'radial_slope',
                                    starts=starts, seed=20261025+round(t*1000))
    return {'L': L, 'G': G, 'time': t, 'family': family, 'fit': result,
            'exact_observables': model.observe(state), 'fit_observables': model.observe(fit)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--levels', type=int, default=12)
    p.add_argument('--G', type=float, default=2.)
    p.add_argument('--times', type=float, nargs='+', default=[.5, 1., 2.15])
    p.add_argument('--starts', type=int, default=12)
    p.add_argument('--workers', type=int, default=9)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    tasks = [(args.levels, args.G, t, f, args.starts) for t in args.times
             for f in ('pbcs', 'radial', 'radial_slope')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(job, tasks):
            name = f"{result['family']}_t{result['time']:g}.json"
            (args.output/name).write_text(json.dumps(result, indent=2))
            print(json.dumps({k: result[k] for k in ('L', 'G', 'time', 'family')} |
                             {'fidelity': result['fit']['fidelity']}), flush=True)
