#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.conditional_dynamics import ConditionalDynamics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    target = np.load(root/'replica2_L12_G2_01/t1.npz')['target']
    manifest = {'L': 12, 'train_time': 1., 'train_G': 2., 'holdout_G': [1.5, 2.5, -1.8],
                'kicks': {'none': 0., 'spectral_plus': .3, 'spectral_minus': -.3},
                'kick_definition': 'exp[-i eta (H0-H0(F)-2m)]',
                'elapsed': [0., .25, .5, 1., 1.5, 2.],
                'protocol': 'New parameter/kick holdouts after static-basis failure. Family and initial coordinates frozen at t=1; subsequent orbital motion comes only from Hamiltonian TDVP. All predictions saved before exact references.',
                'records': []}
    np.savez_compressed(args.output/'training.npz', state=target)
    families = []
    for source in ('target', 'witness'):
        x = np.load(root/f'conditional_orbitals_01/{source}_t1_seed20261031.npz')['parameters']
        families.append((source+'_conditioned', True, x[:, :model.L]+1j*x[:, model.L:]))
        x = np.array(json.loads((root/f'conditional_freeze_01/{source}_common_fit.json').read_text())['parameters'])
        families.append((source+'_common', False, (x[:model.L]+1j*x[model.L:])[None]))
    radial = np.load(root/'frozen_L12_G2_01/linear_rpa_q1.npz')['basis']
    _, indices = model.blocks[1]
    u, _, vh = np.linalg.svd(radial[indices, 1])
    z = np.log(np.maximum(abs(np.r_[u[:, 0], vh[0]]), 1e-14))+1j*np.angle(np.r_[u[:, 0], vh[0]])
    families.append(('linear_common', False, z[None]))
    for name, conditioned, logarithms in families:
        dynamics = ConditionalDynamics(model, conditioned)
        coordinates, fidelity = dynamics.encode(logarithms, target)
        path = args.output/(name+'.npz')
        np.savez_compressed(path, coordinates=coordinates)
        dimension = model.r+1+(model.r-1 if conditioned else 1)*(model.L-2)
        manifest['records'].append({'name': name, 'file': path.name, 'type': 'tdvp', 'conditioned': conditioned,
                'complex_manifold_dimension': dimension, 'initial_fidelity': fidelity,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for name, source in [('linear_rpa_q2', root/'frozen_L12_G2_01/linear_rpa_q2.npz'),
                         ('linear_rpa_q3', root/'frozen_L12_G2_01/linear_rpa_q3.npz'),
                         ('hamiltonian_R32', root/'response_freeze_01/hamiltonian_seed20261029_R32.npz')]:
        data = np.load(source)
        Q = data['basis'] if 'basis' in data else data['Q']
        c = Q.conj().T@target
        fidelity = float(np.linalg.norm(c)**2)
        c /= np.sqrt(fidelity)
        path = args.output/(name+'.npz')
        np.savez_compressed(path, Q=Q, initial=c)
        manifest['records'].append({'name': name, 'file': path.name, 'type': 'static', 'dimension': Q.shape[1],
                    'initial_fidelity': fidelity, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'families': len(manifest['records']), 'holdout_G': manifest['holdout_G']}))
