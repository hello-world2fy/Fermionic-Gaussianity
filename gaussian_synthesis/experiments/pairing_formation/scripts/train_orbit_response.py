#!/usr/bin/env python3
"""Training-only model selection; protocol is written before any new fits."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import time
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.multiorbit import MultiOrbitDynamics
from pairing_formation.orbit_response import response_basis, fit_response_orbits, tangent_response_score


def job(task):
    kind, weight, seed, output = task
    begin = time.monotonic()
    root = Path(__file__).resolve().parents[1]/'results'
    model = picket_fence(12, 2.)
    target = np.load(root/'interband_freeze_01/training.npz')['state']
    dynamics = MultiOrbitDynamics(model, 2)
    early = np.load(root/'multiorbit_training_refined_01'/f'target_t1_seed{seed}.npz')['coordinates']
    _, logs = dynamics.unpack(early)
    responses, operators, basis_report = response_basis(model, target, kind)
    if weight:
        coordinates, report = fit_response_orbits(model, target, responses, weight, logs, seed=seed)
    else:
        coordinates, fidelity = dynamics.encode(logs, target)
        report = {'weight': 0., 'initial_fidelity': fidelity, 'blocks': [], 'reused_direct_training': True}
    report.update({'kind': kind, 'seed': seed, 'basis': basis_report,
                   'training_tangent_response': tangent_response_score(dynamics, coordinates, operators),
                   'elapsed_seconds': time.monotonic()-begin})
    name = f'{kind}_weight{weight:g}_seed{seed}'
    path = output/(name+'.npz')
    np.savez_compressed(path, coordinates=coordinates)
    report.update({'name': name, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (output/(name+'.json')).write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {'L': 12, 'train_G': 2., 'train_time': 1., 'seeds': [20261111, 20261112],
                'weights': [0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.], 'kinds': ['hamiltonian', 'generators'],
                'initial_infidelity_budget_factor': 2.,
                'selection': 'Minimize normalized early-state Hamiltonian normal-response squared norm subject to initial infidelity <= 2 times the direct fit infidelity for that seed. No future references or metrics are used.',
                'prediction_family': 'Same two-orbit-per-count MultiOrbitDynamics; state-averaged response extraction is a conventional control, not canonical synthesis.',
                'known_grid': 'Re-evaluate previous reversal confirmation only as a post-pilot comparison.',
                'new_holdout': {'G': [-1.4, -2.2], 'kicks': {'none': 0., 'fifth': float(np.pi/5)},
                                'spectra': {'warped': 'eps + .3*eps**3', 'asymmetric': 'eps + .25*(eps**2-mean(eps**2))'},
                                'elapsed': [0., .1, .25, .5, 1., 1.5, 2.], 'primary_time': 1.,
                                'primary_metrics': ['absolute negativity error', 'unnormalized filtered trace distance'],
                                'filter': 'The prior training-only band-edge filter, unchanged'},
                'failure_rule': 'A broad canonical advantage is not established if the same-complexity selected response control matches or improves both primary metrics; retain all outcomes, including state fidelity and count errors.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    tasks = [(kind, weight, seed, args.output) for kind in protocol['kinds'] for seed in protocol['seeds'] for weight in protocol['weights']]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job, task) for task in tasks]):
            record = future.result(); records.append(record)
            print(json.dumps({k: record[k] for k in ('name', 'initial_fidelity', 'training_tangent_response', 'elapsed_seconds')}), flush=True)
    selected = []
    for kind in protocol['kinds']:
        for seed in protocol['seeds']:
            subset = [r for r in records if r['kind'] == kind and r['seed'] == seed]
            initial_infidelity = 1-next(r['initial_fidelity'] for r in subset if r['weight'] == 0.)
            allowed = [r for r in subset if 1-r['initial_fidelity'] <= protocol['initial_infidelity_budget_factor']*initial_infidelity+1e-14]
            best = min(allowed, key=lambda r: (r['training_tangent_response']['score'], r['weight']))
            selected.append({**best, 'allowed_weights': sorted(r['weight'] for r in allowed),
                             'max_initial_infidelity': protocol['initial_infidelity_budget_factor']*initial_infidelity})
    (args.output/'selection.json').write_text(json.dumps({'protocol': protocol, 'selected': selected}, indent=2))
    print(json.dumps({'selected': [{k:r[k] for k in ('name','weight','initial_fidelity')} for r in selected]}))
