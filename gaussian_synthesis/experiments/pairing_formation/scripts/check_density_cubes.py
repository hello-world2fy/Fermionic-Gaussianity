#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.density_cubes import mixed_swap_cubes, disjoint_cube_bound, cube_diagnostics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    cubes = mixed_swap_cubes(model)
    np.savez_compressed(args.output/'cubes.npz', rows=cubes)
    reports = []
    for t in (.5, 1., 2.15):
        target = np.load(root/f'replica2_L12_G2_01/t{t:g}.npz')['target']
        report = disjoint_cube_bound(abs(target)**2, cubes)
        report['time'] = t
        report['best_cube']['m'] = model.m[report['best_cube']['rows']].tolist()
        report['best_cube']['occupied_levels'] = [np.flatnonzero(model.occ[i]).tolist() for i in report['best_cube']['rows']]
        controls = {}
        for family, folder in [('two_body', 'jastrow_fits_01'), ('collective_3', 'jastrow_fits_extended_01'), ('conditional_two_body', 'jastrow_fits_extended_01')]:
            state = np.load(root/folder/f'{family}_t{t:g}_seed20261107.npz')['approximation']
            mass, invariant, contribution = cube_diagnostics(abs(state)**2, cubes)
            best = report['best_cube']['index']
            controls[family] = {'max_invariant': float(np.max(abs(invariant))),
                                'exact_selected_cube_invariant': float(invariant[best]),
                                'exact_selected_cube_mass': float(mass[best])}
        report['controls'] = controls
        reports.append(report)
        print(json.dumps({k: report[k] for k in ('time', 'candidate_cubes', 'disjoint_cubes', 'covered_probability', 'fidelity_upper_bound_float64')}), flush=True)
    (args.output/'diagnostics.json').write_text(json.dumps(reports, indent=2))
