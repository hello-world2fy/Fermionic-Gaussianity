#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import json
import argparse
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.replica2 import Replica2
from pairing_formation.columns import onsite_oracle, onsite_to_qp_Z, FullGaussianOracle
from pairing_formation.column_prediction import orthonormal_span


def job(task):
    family, seed, device, count, output = task
    start = time.monotonic()
    model = picket_fence(12, 2.)
    source = Path(__file__).resolve().parents[1]/'results/replica2_L12_G2_01/t1.npz'
    saved = np.load(source)
    target, u = saved['target'], saved['witness']
    energy = np.vdot(target, model.H@target).real
    first = model.H@target-energy*target
    second = model.H@first-energy*first
    response_report = {}
    if family == 'hamiltonian':
        training = np.column_stack((target, first, second))
    elif family == 'canonical_response':
        v, response_report = Replica2(model).witness_response(u, -1j*first)
        training = np.column_stack((target, u, v))
    else:
        raise ValueError(family)
    training, triangular = np.linalg.qr(training)
    oracle = FullGaussianOracle(model, f'cuda:{device}')
    atoms, params, matrices, history = [], [], [], []
    residual = training.copy()
    for k in range(count):
        left, singular, _ = np.linalg.svd(residual, full_matrices=False)
        direction = left[:, 0]
        atom, param, onsite_report = onsite_oracle(model, direction, starts=12, seed=seed+1009*k, seeds=params[-2:])
        atom, Z, full_report = oracle.optimize(direction, param, starts=2, seed=seed+1009*k)
        j = int(np.argmax(abs(direction)))
        fock = abs(direction[j]) > abs(np.vdot(direction, atom))*(1+1e-9)
        if fock:
            atom = np.eye(1, model.dim, j, dtype=complex).ravel()
            Z = np.full_like(Z, np.nan)
        atoms.append(atom)
        params.append(param)
        matrices.append(Z)
        A = np.array(atoms).T
        coefficients = np.linalg.lstsq(A, training, rcond=1e-11)[0]
        residual = training-A@coefficients
        Q, _ = orthonormal_span(A)
        state_coeff = Q.conj().T@target
        leakage = first-Q@(Q.conj().T@first)
        item = {'columns': k+1, 'fidelity': float(np.vdot(state_coeff, state_coeff).real),
                'response_residual_norms': np.linalg.norm(residual, axis=0).tolist(),
                'hamiltonian_leakage_relative': float(np.linalg.norm(leakage)/np.linalg.norm(first)),
                'elapsed_seconds': time.monotonic()-start, 'fock_selected': bool(fock),
                **onsite_report, **full_report}
        history.append(item)
        name = f'{family}_seed{seed}'
        np.savez_compressed(output/f'{name}.npz', atoms=A, parameters=np.array(params), qp_Z=np.array(matrices),
                            target=target, training=training, coefficients=coefficients, initial=state_coeff, Q=Q)
        (output/f'{name}.json').write_text(json.dumps({'family': family, 'seed': seed, 'device': device,
                       'L': 12, 'G': 2., 'train_time': 1., 'training_scope': 'only training-time state and Hamiltonian/witness derivatives',
                       'response_equation': response_report, 'history': history}, indent=2))
        if (k+1) % 8 == 0:
            print(json.dumps({'family': family, 'seed': seed, **{a: item[a] for a in
                      ('columns', 'fidelity', 'hamiltonian_leakage_relative', 'elapsed_seconds')}}), flush=True)
    return name


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--columns', type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(family, seed, device, args.columns, args.output) for device, (family, seed) in
             enumerate((f, s) for f in ('hamiltonian', 'canonical_response') for s in (20261029, 20261030))]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        for name in pool.map(job, tasks):
            print('finished '+name, flush=True)
