#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.fitting import fit_product_family, excitation_features
from pairing_formation.conditional import conditioned_tangent_span


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    saved = np.load(root/'replica2_L12_G2_01/t1.npz')
    target = saved['target']
    first = model.H@target-np.vdot(target, model.H@target).real*target
    manifest = {'L': 12, 'train_G': 2., 'train_time': 1., 'holdout_G': [1.7, 2.3, -2.],
                'kicks': {'none': 0., 'spectral_plus': .4, 'spectral_minus': -.4},
                'kick_definition': 'exp[-i eta (H0-H0(F)-2m)]',
                'elapsed': [0., .25, .5, 1., 1.5, 2.], 'transport': 'static',
                'protocol': 'Freeze all orbital profiles and density-tangent spaces using t=1 only; G=-2 is interaction sign reversal; spectral phase kicks are one-body unitaries',
                'canonical_basis_scope': 'witness-derived coordinates, not canonical measure sampling', 'records': []}
    np.savez_compressed(args.output/'training.npz', state=target)
    bases = {}
    F = excitation_features(model)
    for source in ('target', 'witness'):
        bases[source+'_conditioned'] = np.load(root/f'conditional_orbitals_01/{source}_t1_seed20261031.npz')['basis']
        report, _ = fit_product_family(model, saved[source]/np.linalg.norm(saved[source]), radial=True, starts=12, seed=20261102)
        (args.output/(source+'_common_fit.json')).write_text(json.dumps(report, indent=2))
        x = np.array(report['parameters'])
        raw = F@(x[:model.L]+1j*x[model.L:])
        basis = np.zeros((model.dim, model.r+1), complex)
        for m in range(model.r+1):
            rows = np.flatnonzero(model.m == m)
            v = np.exp(raw[rows]-max(raw[rows].real))
            basis[rows, m] = v/np.linalg.norm(v)
        bases[source+'_common'] = basis
    bases['linear_rpa_common'] = np.load(root/'frozen_L12_G2_01/linear_rpa_q1.npz')['basis']
    families = []
    for name, basis in bases.items():
        for order in (0, 1, 2):
            Q, dimensions = conditioned_tangent_span(model, basis, order=order)
            families.append((name+f'_order{order}', Q, {'density_polynomial_order': order, 'sector_dimensions': dimensions}))
    for q in (2, 3):
        Q = np.load(root/f'frozen_L12_G2_01/linear_rpa_q{q}.npz')['basis']
        families.append((f'linear_rpa_q{q}', Q, {}))
    for family, seed in [('hamiltonian', 20261029), ('canonical_response', 20261029), ('state_only', 20261027)]:
        Q = np.load(root/f'response_freeze_01/{family}_seed{seed}_R32.npz')['Q']
        families.append((family+'_R32', Q, {}))
    for name, Q, extra in families:
        c = Q.conj().T@target
        fidelity = float(np.linalg.norm(c)**2)
        c /= np.sqrt(fidelity)
        path = args.output/(name+'.npz')
        np.savez_compressed(path, Q=Q, initial=c)
        leakage = first-Q@(Q.conj().T@first)
        manifest['records'].append({'name': name, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'dimension': Q.shape[1], 'initial_fidelity': fidelity,
                   'hamiltonian_leakage_relative': float(np.linalg.norm(leakage)/np.linalg.norm(first)), **extra})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'families': len(families), 'dimensions': sorted(set(x[1].shape[1] for x in families))}))
