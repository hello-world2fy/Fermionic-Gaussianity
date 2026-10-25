#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.column_prediction import evolve_span


def predict(task):
    record, G, phase, transport, manifest, frozen, output = task
    source = frozen/record['file']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256']
    with np.load(source) as data:
        Q, c = data['Q'], data['initial']
    model = picket_fence(manifest['L'], G)
    Q = np.exp(-1j*phase*model.m[:, None])*Q
    states = evolve_span(model, Q, c, manifest['holdout_elapsed'], transport)
    name = f"{record['name']}_G{G:g}_phase{phase:.6f}_{transport}"
    np.savez_compressed(output/(name+'.npz'), states=states)
    return {'name': name, 'family': record['name'], 'G': G, 'phase': phase, 'transport': transport,
            'dimension': record['dimension'], 'max_norm_error': float(max(abs(np.linalg.norm(states, axis=1)-1)))}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=24)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.frozen/'manifest.json').read_text())
    tasks = [(record, G, phase, transport, manifest, args.frozen, args.output)
             for record in manifest['records'] for G in manifest['holdout_G']
             for phase in manifest['phase_kicks'] for transport in manifest['transports']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        predictions = list(pool.map(predict, tasks))
    (args.output/'predictions_complete.json').write_text(json.dumps(predictions, indent=2))
    # Exact held-out states are accessed only after every prediction is saved.
    initial = np.load(args.frozen/'training.npz')['state']
    references = {}
    for G in manifest['holdout_G']:
        for phase in manifest['phase_kicks']:
            model = picket_fence(manifest['L'], G)
            states = model.evolve(manifest['holdout_elapsed'], initial*np.exp(-1j*phase*model.m))
            references[G, phase] = states
    evaluations = []
    for record in predictions:
        model = picket_fence(manifest['L'], record['G'])
        predicted = np.load(args.output/(record['name']+'.npz'))['states']
        exact = references[record['G'], record['phase']]
        rows = []
        for t, a, b in zip(manifest['holdout_elapsed'], predicted, exact):
            estimate, truth = model.observe(a), model.observe(b)
            rows.append({'elapsed': t, 'fidelity': float(abs(np.vdot(a, b))**2),
                         'count_TV': float(.5*np.sum(abs(np.array(estimate['count'])-truth['count']))),
                         'mean_error': estimate['mean']-truth['mean'],
                         'variance_error': estimate['variance']-truth['variance'],
                         'cross_error': float(abs(complex(estimate['cross_real']-truth['cross_real'],
                                                          estimate['cross_imag']-truth['cross_imag'])))})
        evaluations.append({**record, 'rows': rows})
    (args.output/'evaluation.json').write_text(json.dumps(evaluations, indent=2))
    print(json.dumps({'predictions': len(evaluations), 'references': len(references),
                       'max_norm_error': max(x['max_norm_error'] for x in predictions)}), flush=True)
