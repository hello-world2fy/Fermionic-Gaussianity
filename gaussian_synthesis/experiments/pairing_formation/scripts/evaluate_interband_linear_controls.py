#!/usr/bin/env python3
"""Unfitted scattering-time and finite-size linear-quantum controls.

These are mechanistic checks added after the pilot. They do not constitute a
new prospectively selected primary endpoint. Born weights can exceed one before
normalization: no perturbative accuracy claim is made at these strong quenches.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.linear import matrices, squeezed_pair_matrix, coupled_cluster_state
from pairing_formation.interband import filtered_density, tomography
from evaluate_conditional import errors


def propagate(model, initial, elapsed):
    particle, hole, source, _ = matrices(model)
    def rhs(t, flat):
        T = flat.reshape(model.r, model.r)
        return (-1j*(source+particle@T+T@hole.T+T@source.conj().T@T)).ravel()
    sol = solve_ivp(rhs, (0., elapsed[-1]), initial.ravel(), t_eval=elapsed, method='DOP853', rtol=2e-11, atol=2e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(-1, model.r, model.r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.frozen/'manifest.json').read_text())
    initial = np.load(args.frozen/'training.npz')['state']
    training_model = picket_fence(manifest['L'], manifest['train_G'])
    T_linear = squeezed_pair_matrix(training_model, [manifest['train_time']])[0]
    T_matched = initial[training_model.blocks[1][1]]/initial[training_model.fermi_index]
    detuning = 2*(training_model.energies[training_model.r:, None]-training_model.energies[None, :training_model.r])
    def integral(t):
        return -np.expm1(-1j*detuning*t)/(1j*detuning)
    predictions = []
    for G in manifest['holdout_G']:
        model = picket_fence(manifest['L'], G)
        for kick, alpha in manifest['kicks'].items():
            for family, T in [('RPA_from_normal', T_linear), ('RPA_m1_matched', T_matched), ('Born_from_normal', None)]:
                if T is not None:
                    matrices_t = propagate(model, T*np.exp(-1j*alpha), manifest['elapsed'])
                    states = np.array([coupled_cluster_state(model, A) for A in matrices_t])
                    raw_weights = []
                else:
                    states, raw_weights = [], []
                    for elapsed in manifest['elapsed']:
                        A = 1j/model.L*(manifest['train_G']*np.exp(-1j*alpha-1j*detuning*elapsed)*integral(manifest['train_time'])+G*integral(elapsed))
                        state = np.zeros(model.dim, complex); state[model.fermi_index] = 1.
                        state[model.blocks[1][1].ravel()] = A.ravel()
                        raw_weights.append(float(np.linalg.norm(A)**2))
                        states.append(state/np.linalg.norm(state))
                    states = np.array(states)
                name = f'{family}_G{G:g}_{kick}'
                np.savez_compressed(args.output/(name+'.npz'), states=states)
                predictions.append({'name': name, 'family': family, 'G': G, 'kick': kick, 'alpha': alpha,
                                    'raw_Born_one_pair_weights': raw_weights})
    (args.output/'predictions_complete.json').write_text(json.dumps(predictions, indent=2))
    references = {}
    for G in manifest['holdout_G']:
        model = picket_fence(manifest['L'], G)
        for kick, alpha in manifest['kicks'].items():
            references[G, kick] = model.evolve(manifest['elapsed'], initial*np.exp(-1j*alpha*model.m))
    chosen = manifest['filters']['target']
    evaluated = []
    for record in predictions:
        model = picket_fence(manifest['L'], record['G'])
        predicted = np.load(args.output/(record['name']+'.npz'))['states']
        rows = []
        for elapsed, a, b in zip(manifest['elapsed'], predicted, references[record['G'], record['kick']]):
            ra, pa = filtered_density(model, a, chosen['particle'], chosen['hole'])
            rb, pb = filtered_density(model, b, chosen['particle'], chosen['hole'])
            oa, ob = tomography(ra, pa), tomography(rb, pb)
            rows.append({'elapsed': elapsed, **errors(model, a, b), 'filter_prediction': oa, 'filter_reference': ob,
                         'negativity_error': oa['negativity']-ob['negativity'],
                         'unnormalized_filtered_trace_distance': float(np.linalg.svd(pa*ra-pb*rb, compute_uv=False).sum()/2)})
        evaluated.append({**record, 'rows': rows})
    (args.output/'evaluation.json').write_text(json.dumps(evaluated, indent=2))
    print(json.dumps({'predictions': len(predictions), 'scope': 'Post-pilot mechanistic baseline, no future-state fitting'}))
