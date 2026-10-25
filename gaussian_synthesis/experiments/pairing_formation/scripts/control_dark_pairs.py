#!/usr/bin/env python3
"""Same-Fermi-state deletion of operator-defined dark band subspaces."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse, hashlib, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import factorial
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.dark_pairs import PairDarkSectors


def job(task):
    L, out = task
    root = out.parent
    model, forms = angular_model(L)
    dark = PairDarkSectors(model, forms)
    Q = load_npz(root / 'rotation_geometry_01' / f'L{L}_symmetry.npz')
    H = (Q.T @ model.H @ Q).toarray()
    F = Q.T @ model.initial
    counts = np.rint(np.asarray(Q.power(2).T @ model.m).ravel()).astype(int)
    blocks = []
    checks = []
    for m, (_, indices) in enumerate(model.blocks):
        cols = np.flatnonzero(counts == m)
        W = Q[indices.ravel()][:, cols].toarray()
        n = indices.shape[0]
        C = W.reshape(n, n, len(cols))
        D = dark.projectors[m]
        left = np.einsum('ab,bck->ack', D, C)
        right = np.einsum('abk,cb->ack', C, D)
        both = np.einsum('abk,cb->ack', left, D)
        matrices = {kind: W.T @ P.reshape(n*n, len(cols))
                    for kind, P in (('both', both), ('either', left + right - both))}
        bases = {}
        for kind, P in matrices.items():
            values, U = np.linalg.eigh((P + P.conj().T) / 2)
            error = float(np.max(np.minimum(abs(values), abs(values-1))))
            assert error < 3e-11, (m, kind, error)
            bases[kind] = U[:, values < .5]
            checks.append({'m': m, 'kind': kind, 'dimension': len(cols),
                           'dark_dimension_in_symmetry': int(sum(values > .5)),
                           'projector_eigenvalue_error': error})
        blocks.append((cols, bases))
    spaces = {}
    for kind, selection in [('either', 'm1'), ('either', 'multiple'),
                            ('either', 'all'), ('both', 'multiple'), ('both', 'all')]:
        vectors = []
        for m, (cols, bases) in enumerate(blocks):
            remove = (m == 1 if selection == 'm1' else m >= 2 if selection == 'multiple' else m >= 1)
            local = bases[kind] if remove else np.eye(len(cols))
            V = np.zeros((len(F), local.shape[1]), complex)
            V[cols] = local
            vectors.append(V)
        spaces[f'delete_{kind}_{selection}'] = np.column_stack(vectors)
    spaces['exact_symmetry'] = np.eye(len(F))
    predictions = {}
    metadata = []
    for name, V in spaces.items():
        initial = V.conj().T @ F
        initial_error = float(np.linalg.norm(V @ initial - F))
        assert initial_error < 1e-12
        Hr = V.conj().T @ H @ V
        coefficients = expm_multiply(-1j * Hr, initial, start=0., stop=2.5, num=51,
                                     traceA=-1j*np.trace(Hr))
        states = coefficients @ V.T
        predictions[name] = states
        metadata.append({'name': name, 'dimension': V.shape[1], 'initial_error': initial_error,
                         'norm_error': float(np.max(abs(np.sum(abs(states)**2, axis=1)-1)))})
    frozen_path = out / f'L{L}_frozen_predictions.npz'
    np.savez_compressed(frozen_path, times=np.arange(51)/20, **predictions)
    record = {'L': L, 'checks': checks, 'models': metadata,
              'sha256': hashlib.sha256(frozen_path.read_bytes()).hexdigest()}
    (out / f'L{L}_frozen.json').write_text(json.dumps(record, indent=2))
    exact_full = np.load(root / 'nonlinear_channels_01' / f'L{L}_eta1_exact.npz')['states']
    exact = exact_full @ Q
    results = []
    for meta in metadata:
        rows = []
        for t, s, ref in zip(np.arange(51)/20, predictions[meta['name']], exact):
            p = np.bincount(counts, weights=abs(s)**2, minlength=model.r+1)
            p0 = np.bincount(counts, weights=abs(ref)**2, minlength=model.r+1)
            rate, rate0 = H@s, H@ref
            I = np.array([2*np.vdot(s[counts==m], rate[counts==m]).imag for m in range(model.r+1)])
            I0 = np.array([2*np.vdot(ref[counts==m], rate0[counts==m]).imag for m in range(model.r+1)])
            rows.append({'time': float(t), 'fidelity': float(abs(np.vdot(ref, s))**2),
                         'count_TV': float(.5*np.sum(abs(p-p0))),
                         'current_L1_error': float(np.sum(abs(I-I0)))})
        results.append({**meta, 'rows': rows})
    # Direct Taylor derivatives and the unique up...up/down path are computed
    # from H only, without fitting any time-series coefficients.
    fullH = model.H.tocoo()
    from scipy.sparse import coo_matrix
    up = model.m[fullH.row] - model.m[fullH.col] == 1
    Hplus = coo_matrix((fullH.data[up], (fullH.row[up], fullH.col[up])), shape=fullH.shape).tocsr()
    Hminus = Hplus.conj().T
    power = model.initial.copy()
    derivatives = []
    for order in range(1, model.r+4):
        power = model.H @ power
        P = dark.apply(power, 'either')
        derivatives.append([float(np.linalg.norm(P[indices]))/factorial(order) for _, indices in model.blocks])
    paths = []
    raised = model.initial.copy()
    for count in range(1, model.r+2):
        raised = Hplus @ raised
        m = count-1
        if m < 1 or m >= model.r: continue
        path = dark.apply(Hminus @ raised, 'either')
        exact_power = model.initial.copy()
        for _ in range(m+2): exact_power = model.H @ exact_power
        selected = model.blocks[m][1]
        actual = dark.apply(exact_power, 'either')[selected]
        expected = path[selected]
        paths.append({'m': m, 'leading_order_candidate': m+2,
                      'coefficient_norm': float(np.linalg.norm(expected))/factorial(m+2),
                      'relative_unique_path_error': float(np.linalg.norm(actual-expected)/max(1., np.linalg.norm(expected)))})
    result = {'L': L, 'frozen': record, 'models': results, 'taylor_coefficient_norms': derivatives,
              'unique_return_paths': paths,
              'max_full_reference_difference': float(np.max(np.linalg.norm(predictions['exact_symmetry']-exact, axis=1)))}
    (out / f'L{L}_evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'L': L, 'time2': [{**{k: row[k] for k in ('name','dimension')}, **row['rows'][40]}
                                     for row in results], 'return_paths': paths}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'protocol.json').write_text(json.dumps({
        'classification': 'Retrospective physical mechanism deletion; no fresh holdouts or canonical-selected space.',
        'initial_state': 'Exact same Fermi sea for all full and projected Hamiltonians.',
        'deletions': 'Either-band or both-band source-annihilator dark subspaces, separately m1 and m>=2.',
        'limitations': 'Projected Hamiltonian ablation is not an implemented experimental intervention. Dark subspaces need not be dynamically invariant.'}, indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job, (L, args.output)) for L in (14, 18)]): f.result()
