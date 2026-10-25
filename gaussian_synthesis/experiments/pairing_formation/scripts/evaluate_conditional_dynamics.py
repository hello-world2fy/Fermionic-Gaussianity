#!/usr/bin/env python3
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
from pairing_formation.conditional_dynamics import ConditionalDynamics
from pairing_formation.column_prediction import evolve_span
from evaluate_conditional import errors, phase_diagonal


def predict(task):
    record, G, kick, eta, manifest, frozen, output = task
    begin = time.monotonic()
    path = frozen/record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    data = np.load(path)
    model = picket_fence(manifest['L'], G)
    report = {}
    name = f"{record['name']}_G{G:g}_{kick}"
    if record['type'] == 'tdvp':
        dynamics = ConditionalDynamics(model, record['conditioned'])
        initial = data['coordinates'].copy()
        c, z = dynamics.unpack(initial)
        kick_coefficients = np.r_[2*model.energies[model.r:]-1, -2*model.energies[:model.r]-1]
        z -= 1j*eta*kick_coefficients
        states, coordinates, report = dynamics.evolve(initial, manifest['elapsed'])
        np.savez_compressed(output/(name+'.npz'), states=states, coordinates=coordinates)
    else:
        Q = data['Q']*np.exp(-1j*eta*phase_diagonal(model))[:, None]
        states = evolve_span(model, Q, data['initial'], manifest['elapsed'])
        np.savez_compressed(output/(name+'.npz'), states=states)
    record = {'name': name, 'family': record['name'], 'type': record['type'], 'G': G, 'kick': kick, 'eta': eta,
              'elapsed_seconds': time.monotonic()-begin,
              'max_norm_error': float(np.max(abs(np.linalg.norm(states, axis=1)-1))), 'integration': report}
    (output/(name+'.json')).write_text(json.dumps(record, indent=2))
    return record


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
    predictions = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(predict, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            predictions.append(record)
            print(json.dumps({k: record[k] for k in ('name', 'elapsed_seconds', 'max_norm_error')}), flush=True)
    (args.output/'predictions_complete.json').write_text(json.dumps(predictions, indent=2))
    # Exact future states are accessed only after the entire batch is saved.
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
