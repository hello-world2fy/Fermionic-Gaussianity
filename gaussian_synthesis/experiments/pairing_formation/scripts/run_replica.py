#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import sys
import json
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.replica2 import Replica2
from pairing_formation.structure import analyze


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--levels', type=int, default=12)
    parser.add_argument('--G', type=float, default=2.)
    parser.add_argument('--times', type=float, nargs='+', default=[.5, 1., 2.15])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = picket_fence(args.levels, args.G)
    replica = Replica2(model)
    for t in args.times:
        start = time.monotonic()
        state = model.evolve([t])[0]
        u, report = replica.solve(state)
        report.update({'L': args.levels, 'G': args.G, 'time': t,
                       'state_structure': analyze(model, state)[0],
                       'witness_structure': analyze(model, u/np.linalg.norm(u))[0],
                       'wall_seconds': time.monotonic()-start})
        np.savez_compressed(args.output/f't{t:g}.npz', target=state, witness=u,
                            reconstruction=replica.evaluate(u)[1])
        (args.output/f't{t:g}.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ('L', 'G', 'time', 'residual',
                     'moment_dual_extent_lower', 'wall_seconds')}), flush=True)
