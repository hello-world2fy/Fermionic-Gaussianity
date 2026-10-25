#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import json
import argparse
import hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.column_prediction import orthonormal_span


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    target = np.load(root/'replica2_L12_G2_01/t1.npz')['target']
    model = picket_fence(12, 2.)
    derivative = model.H@target-np.vdot(target, model.H@target).real*target
    manifest = {'L': 12, 'train_G': 2., 'train_time': 1.,
                'holdout_G': [1.6, 2.6], 'phase_kicks': [0., float(np.pi/5), float(-np.pi/5)],
                'holdout_elapsed': [0., .25, .5, 1., 1.5, 2.], 'transports': ['static', 'free'],
                'protocol': 'training-time Hamiltonian response versus canonical witness response; new parameter/kick holdouts; all families evaluated',
                'actual_canonical_measure_sampled_for_response_training': False, 'records': []}
    families = []
    for family in ('hamiltonian', 'canonical_response'):
        for seed in (20261029, 20261030):
            path = root/f'response_columns_01/{family}_seed{seed}.npz'
            data = np.load(path)
            if data['atoms'].shape[1] != 32:
                raise RuntimeError('Response training is not complete')
            for rank in (8, 16, 32):
                families.append((f'{family}_seed{seed}_R{rank}', data['atoms'][:, :rank]))
    for seed in (20261027, 20261028):
        data = np.load(root/f'adaptive_full_L12_G2_01/t1_seed{seed}.npz')
        for rank in (8, 16, 32):
            families.append((f'state_only_seed{seed}_R{rank}', data['atoms'][:, :rank]))
    for q in (1, 2, 3):
        families.append((f'linear_rpa_q{q}', np.load(root/f'frozen_L12_G2_01/linear_rpa_q{q}.npz')['basis']))
    np.savez_compressed(args.output/'training.npz', state=target)
    for name, atoms in families:
        Q, singular = orthonormal_span(atoms)
        coefficients = Q.conj().T@target
        fidelity = float(np.vdot(coefficients, coefficients).real)
        coefficients /= np.sqrt(fidelity)
        file = args.output/(name+'.npz')
        np.savez_compressed(file, Q=Q, initial=coefficients, atoms=atoms)
        leakage = derivative-Q@(Q.conj().T@derivative)
        manifest['records'].append({'name': name, 'file': file.name,
              'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'dimension': Q.shape[1],
              'initial_fidelity': fidelity, 'hamiltonian_leakage_relative': float(np.linalg.norm(leakage)/np.linalg.norm(derivative))})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'frozen_families': len(families), 'holdout_G': manifest['holdout_G']}))
