#!/usr/bin/env python3
"""Numerical cross-checks and retrospective physics consequences of the AGP bound."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.projected_count_bound import CanonicalAGPCheck, moment_fidelity_bound
from pairing_formation.model import picket_fence


def check_random(task):
    K, M, count, seed = task
    model = CanonicalAGPCheck(K, M)
    rng = np.random.default_rng(seed)
    cases = []
    for trial in range(count):
        scale = [0., .2, 1., 3., 8.][trial % 5]
        logarithms = rng.normal(0, scale, K)+1j*rng.uniform(-np.pi, np.pi, K)
        U = np.linalg.qr(rng.normal(size=(2*K, 2*K))+1j*rng.normal(size=(2*K, 2*K)))[0]
        rank = int(rng.integers(1, 2*K))
        A = np.diag(np.r_[np.ones(rank), np.zeros(2*K-rank)])
        P = U.conj().T@A@U
        row = model.check(logarithms, P)
        if trial < 3:
            state = model.rotated_full_state(logarithms, U)
            number = np.sum(model.configurations < rank, axis=1)
            mean = abs(state)**2@number
            variance = abs(state)**2@(number**2)-mean**2
            row['independent_pfaffian_mean_error'] = float(abs(mean-row['mean_number']))
            row['independent_pfaffian_variance_error'] = float(abs(variance-row['variance_number']))
        assert row['max_offdiagonal_pair_covariance'] < 1e-12
        assert row['diagonal_variance'] <= row['diagonal_bound']+1e-11
        assert row['broken_pair_variance'] <= row['broken_pair_bound']+1e-11
        assert row['diagonal_bound']+row['broken_pair_bound'] <= row['simple_bound']+1e-10
        assert row['variance_number'] <= row['universal_bound']+1e-10
        cases.append(row)
    return {'canonical_pairs': K, 'particles': 2*M, 'cases': count, 'seed': seed,
            'max_fano': max(x['pair_fano'] for x in cases),
            'max_decomposition_error': max(x['variance_decomposition_error'] for x in cases),
            'max_independent_pfaffian_mean_error': max(x.get('independent_pfaffian_mean_error', 0.) for x in cases),
            'max_independent_pfaffian_variance_error': max(x.get('independent_pfaffian_variance_error', 0.) for x in cases),
            'all_inequality_steps_passed': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]/'results'
    tasks = [(K, M, 300, 20261121+100*K+M) for K in (3, 4, 6, 8) for M in sorted({1, K//2, K-1})]
    random_checks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(check_random, task) for task in tasks]):
            result = future.result(); random_checks.append(result)
            print(json.dumps(result), flush=True)
    # Near an occupied lower-band determinant, the ratio approaches the bound.
    boundary = []
    check = CanonicalAGPCheck(4, 2)
    for small in (-1., -3., -6., -10.):
        row = check.check(np.array([0., 0., small, small]), np.diag([0.]*4+[1.]*4))
        boundary.append({'upper_log_weight': small, **row})
    (args.output/'random_checks.json').write_text(json.dumps({'scope': 'Independent full-Fock checks support the analytic proof; random checks alone are not a universal certificate',
                                                             'records': random_checks, 'boundary': boundary}, indent=2))
    # Saved first-round states: these consequences are retrospective, not new forecasts.
    physics = []
    for L in (8, 12, 16):
        for G in (2., 4.):
            folder = 'physics_L16_01' if L == 16 else 'physics_screen_01'
            source = np.load(root/folder/f'L{L}_G{G:g}_states.npz')
            model = picket_fence(L, G)
            rows = []
            for t, state in zip(source['times'], source['states']):
                obs = model.observe(state)
                probability = np.array(obs['count'])
                theta = .2
                echo_difference = float(1-abs(probability@np.exp(-1j*theta*np.arange(L//2+1)))**2)
                excess = echo_difference-theta**2*obs['mean']
                threshold = max(0., 1-np.sqrt(theta**2*obs['mean']/echo_difference)) if echo_difference > 1e-15 else 0.
                rows.append({'time': float(t), 'mean': obs['mean'], 'variance': obs['variance'], 'fano': obs['fano'],
                             'pure_QFI': 4*obs['variance'], 'free_convex_hull_QFI_bound': 4*obs['mean'],
                             'theta': theta, 'pure_echo_difference': echo_difference, 'finite_angle_witness_excess': excess,
                             'collective_dephasing_fraction_threshold': threshold,
                             'fidelity_certificate': moment_fidelity_bound(obs['mean'], obs['variance'], model.r)})
            physics.append({'L': L, 'G': G, 'source': str((root/folder).name), 'rows': rows})
    (args.output/'physics_certificates.json').write_text(json.dumps(physics, indent=2))
    print(json.dumps({'random_cases': sum(x['cases'] for x in random_checks),
                      'physics_maxima': [{'L':d['L'],'G':d['G'],'max_echo_excess':max(x['finite_angle_witness_excess'] for x in d['rows']),
                                         'max_infidelity_bound':max(x['fidelity_certificate']['infidelity_lower_bound'] for x in d['rows'])} for d in physics]}))
