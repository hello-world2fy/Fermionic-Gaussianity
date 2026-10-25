#!/usr/bin/env python3
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.frozen import propagate


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--frozen', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.frozen/'manifest.json').read_text())
    times = np.array(manifest['holdout_times'])
    elapsed = times-manifest['train_time']
    initial = np.load(args.frozen/'training_state.npz')['state']
    predictions = []
    for G in manifest['holdout_G']:
        model = picket_fence(manifest['L'], G)
        for record in manifest['bases']:
            path = args.frozen/record['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
            with np.load(path) as data:
                Q, c = data['basis'], data['initial']
            H = Q.conj().T@(model.H@Q)
            states = propagate(H, c, elapsed)@Q.T
            name = f"{record['name']}_q{record['q']}_G{G:g}"
            # Write predictions before computing ANY holdout reference states.
            np.savez_compressed(args.output/f'{name}.npz', times=times, states=states)
            rows = [model.observe(s) for s in states]
            predictions.append({**record, 'prediction_file': f'{name}.npz', 'G': G, 'rows': rows})
    (args.output/'predictions.json').write_text(json.dumps(predictions, indent=2))
    references = {}
    for G in manifest['holdout_G']:
        model = picket_fence(manifest['L'], G)
        states = model.evolve(elapsed, initial=initial)
        references[G] = states
        np.savez_compressed(args.output/f'reference_G{G:g}.npz', times=times, states=states)
    evaluated = []
    for record in predictions:
        G = record['G']
        model = picket_fence(manifest['L'], G)
        predicted = np.load(args.output/record['prediction_file'])['states']
        rows = []
        for t, exact, approx in zip(times, references[G], predicted):
            truth, estimate = model.observe(exact), model.observe(approx)
            rows.append({'time': float(t), 'fidelity': float(abs(np.vdot(exact, approx))**2),
                         'count_TV': float(.5*np.sum(abs(np.array(truth['count'])-estimate['count']))),
                         'mean_error': estimate['mean']-truth['mean'],
                         'variance_error': estimate['variance']-truth['variance'],
                         'cross_complex_error': float(abs(complex(estimate['cross_real']-truth['cross_real'],
                                                                estimate['cross_imag']-truth['cross_imag'])))})
        evaluated.append({'source': record['name'], 'q': record['q'], 'G': G,
                          'dimension': record['dimension'], 'rows': rows})
        print(json.dumps({'source': record['name'], 'q': record['q'], 'G': G,
                          'dimension': record['dimension'], 'peak': rows[4], 'last': rows[-1]}), flush=True)
    (args.output/'evaluation.json').write_text(json.dumps(evaluated, indent=2))
