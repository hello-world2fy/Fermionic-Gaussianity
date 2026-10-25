#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.column_prediction import orthonormal_span, pool_pursuit, coherent_kernels


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    target = np.load(results/'replica2_L12_G2_01/t1.npz')['target']
    manifest = {'train_time': 1., 'train_G': 2., 'L': 12,
                'holdout_G': [1.8, 2.4], 'phase_kicks': [0., float(np.pi/3), float(-np.pi/3)],
                'holdout_elapsed': [0., .25, .5, 1., 1.5, 2.],
                'transports': ['static', 'free'],
                'protocol': 'freeze Gaussian symmetry-orbit atoms at t=1; optional common free one-body transport; all cases declared before evaluation',
                'records': []}
    np.savez_compressed(args.output/'training.npz', state=target)
    families = []
    for seed in (20261027, 20261028):
        path = results/f'adaptive_full_L12_G2_01/t1_seed{seed}.npz'
        data = np.load(path)
        if data['atoms'].shape[1] != 32:
            raise RuntimeError('Adaptive training is incomplete; do not freeze a partial checkpoint')
        for rank in (8, 16, 32):
            families.append((f'adaptive_seed{seed}_R{rank}', data['atoms'][:, :rank]))
    for family in ('canonical', 'husimi', 'haar'):
        pool = [np.load(results/f'full_haar_L12_t1_pilot_01/device{device}_{family}.npz') for device in range(8)]
        atoms = np.concatenate([x['atoms'] for x in pool], axis=0).T
        keys = np.concatenate([x['keys'] for x in pool])
        selected, history = pool_pursuit(atoms, target, 64)
        (args.output/f'{family}_pool_selection.json').write_text(json.dumps(history, indent=2))
        for rank in (8, 16, 32, 64):
            families.append((f'{family}_weighted_R{rank}', atoms[:, np.argsort(keys)[-rank:]]))
            families.append((f'{family}_poolgreedy_R{rank}', atoms[:, selected[:rank]]))
    # Include the previous strong physical baseline at its actual dimension.
    for q in (1, 2, 3):
        Q = np.load(results/f'frozen_L12_G2_01/linear_rpa_q{q}.npz')['basis']
        families.append((f'linear_rpa_q{q}', Q))
    for name, atoms in families:
        Q, singular = orthonormal_span(atoms)
        initial = Q.conj().T@target
        fidelity = float(np.vdot(initial, initial).real)
        initial /= np.sqrt(fidelity)
        coef = np.linalg.lstsq(atoms, target, rcond=1e-11)[0]
        residual = target-atoms@coef
        filename = name+'.npz'
        np.savez_compressed(args.output/filename, Q=Q, initial=initial, atoms=atoms, coefficients=coef)
        record = {'name': name, 'dimension': Q.shape[1], 'initial_fidelity': fidelity,
                  'file': filename, 'sha256': hashlib.sha256((args.output/filename).read_bytes()).hexdigest(),
                  'coefficient_mass': float(np.sum(abs(coef))),
                  'fock_residual_repair_l1': float(np.sum(abs(residual))),
                  'cost_scope': 'Gaussian symmetry-orbit cost except linear_rpa basis, whose columns are not single atoms',
                  'kernels': coherent_kernels(model, atoms, coef)}
        manifest['records'].append(record)
        print(json.dumps({k: record[k] for k in ('name', 'dimension', 'initial_fidelity', 'coefficient_mass')}), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
