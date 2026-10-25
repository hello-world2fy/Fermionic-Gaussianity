#!/usr/bin/env python3
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.column_prediction import evolve_span
from pairing_formation.conditional import energy_resolved_observables


def phase_diagonal(model):
    energy = 2*model.occ@model.energies
    return energy-energy[model.fermi_index]-2*model.m


def predict(task):
    record, G, kick, eta, manifest, frozen, output = task
    source = frozen/record['file']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256']
    saved = np.load(source)
    model = picket_fence(manifest['L'], G)
    Q = saved['Q']*np.exp(-1j*eta*phase_diagonal(model))[:, None]
    states = evolve_span(model, Q, saved['initial'], manifest['elapsed'])
    name = f"{record['name']}_G{G:g}_{kick}"
    np.savez_compressed(output/(name+'.npz'), states=states)
    return {'name': name, 'family': record['name'], 'G': G, 'kick': kick, 'eta': eta,
            'dimension': record['dimension'], 'max_norm_error': float(np.max(abs(np.linalg.norm(states, axis=1)-1)))}


def errors(model, approximate, exact):
    a, b = model.observe(approximate), model.observe(exact)
    ea, eb = energy_resolved_observables(model, approximate), energy_resolved_observables(model, exact)
    pa = np.array(ea['cross_pair_real'])+1j*np.array(ea['cross_pair_imag'])
    pb = np.array(eb['cross_pair_real'])+1j*np.array(eb['cross_pair_imag'])
    return {'fidelity': float(abs(np.vdot(approximate, exact))**2),
            'count_TV': float(.5*np.sum(abs(np.array(a['count'])-b['count']))),
            'mean_error': a['mean']-b['mean'], 'variance_error': a['variance']-b['variance'],
            'collective_cross_error': float(abs(complex(a['cross_real']-b['cross_real'], a['cross_imag']-b['cross_imag']))),
            'energy_resolved_cross_RMS': float(np.linalg.norm(pa-pb)/model.r),
            'energy_resolved_cross_max': float(np.max(abs(pa-pb))),
            'occupation_max_error': float(np.max(abs(np.array(ea['occupations'])-eb['occupations']))),
            'connected_occupation_RMS': float(np.linalg.norm(np.array(ea['connected_occupations'])-eb['connected_occupations'])/model.L),
            'transfer_energy_mean_error': ea['energy_transfer_mean']-eb['energy_transfer_mean'],
            'transfer_energy_variance_error': ea['energy_transfer_variance']-eb['energy_transfer_variance']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.frozen/'manifest.json').read_text())
    tasks = [(record, G, kick, eta, manifest, args.frozen, args.output) for record in manifest['records']
             for G in manifest['holdout_G'] for kick, eta in manifest['kicks'].items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        predictions = list(pool.map(predict, tasks))
    (args.output/'predictions_complete.json').write_text(json.dumps(predictions, indent=2))
    # Reference access is deliberately after all predictions have been saved.
    initial = np.load(args.frozen/'training.npz')['state']
    references = {}
    for G in manifest['holdout_G']:
        model = picket_fence(manifest['L'], G)
        for kick, eta in manifest['kicks'].items():
            references[G, kick] = model.evolve(manifest['elapsed'], initial*np.exp(-1j*eta*phase_diagonal(model)))
    evaluation = []
    for record in predictions:
        model = picket_fence(manifest['L'], record['G'])
        prediction = np.load(args.output/(record['name']+'.npz'))['states']
        exact = references[record['G'], record['kick']]
        rows = [{'elapsed': elapsed, **errors(model, a, b)} for elapsed, a, b in zip(manifest['elapsed'], prediction, exact)]
        evaluation.append({**record, 'rows': rows})
    (args.output/'evaluation.json').write_text(json.dumps(evaluation, indent=2))
    print(json.dumps({'predictions': len(predictions), 'references': len(references),
                      'max_norm_error': max(x['max_norm_error'] for x in predictions)}))
