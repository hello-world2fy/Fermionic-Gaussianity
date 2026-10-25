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
from pairing_formation.multiorbit import MultiOrbitDynamics
from pairing_formation.conditional_dynamics import ConditionalDynamics
from pairing_formation.jastrow import JastrowDynamics
from pairing_formation.column_prediction import evolve_span
from pairing_formation.interband import filtered_density, tomography
from evaluate_conditional import errors


def predict(task):
    record, G, kick, alpha, manifest, frozen, output = task
    begin = time.monotonic()
    path = frozen/record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    saved = np.load(path)
    model = picket_fence(manifest['L'], G)
    name = f"{record['name']}_G{G:g}_{kick}"
    report = {}
    if record['type'] == 'static':
        Q = saved['Q']*np.exp(-1j*alpha*model.m)[:, None]
        states = evolve_span(model, Q, saved['initial'], manifest['elapsed'])
        coordinates = np.empty((0, 0))
    else:
        initial = saved['coordinates'].copy()
        if record['type'] == 'multiorbit':
            dynamics = MultiOrbitDynamics(model, record['rank'])
            c, logs = dynamics.unpack(initial)
            c *= np.exp(-1j*alpha*np.arange(model.r+1))[:, None]
        elif record['type'] == 'jastrow':
            dynamics = JastrowDynamics(model, record['jastrow_family'], features=saved['features'])
            initial = dynamics.apply_diagonal_phase(initial, alpha*model.m)
        else:
            dynamics = ConditionalDynamics(model, record['conditioned'])
            c, logs = dynamics.unpack(initial)
            c *= np.exp(-1j*alpha*np.arange(model.r+1))
        states, coordinates, report = dynamics.evolve(initial, manifest['elapsed'])
    np.savez_compressed(output/(name+'.npz'), states=states, coordinates=coordinates)
    result = {'name': name, 'family': record['name'], 'type': record['type'], 'G': G, 'kick': kick, 'alpha': alpha,
              'elapsed_seconds': time.monotonic()-begin,
              'max_norm_error': float(np.max(abs(np.linalg.norm(states, axis=1)-1))), 'integration': report}
    (output/(name+'.json')).write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.frozen/'manifest.json').read_text())
    tasks = [(record, G, kick, alpha, manifest, args.frozen, args.output) for record in manifest['records']
             for G in manifest['holdout_G'] for kick, alpha in manifest['kicks'].items()]
    predictions = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(predict, task) for task in tasks]):
            record = future.result()
            predictions.append(record)
            print(json.dumps({k: record[k] for k in ('name', 'elapsed_seconds', 'max_norm_error')}), flush=True)
    (args.output/'predictions_complete.json').write_text(json.dumps(predictions, indent=2))
    initial = np.load(args.frozen/'training.npz')['state']
    references = {}
    for G in manifest['holdout_G']:
        model = picket_fence(manifest['L'], G)
        for kick, alpha in manifest['kicks'].items():
            references[G, kick] = model.evolve(manifest['elapsed'], initial*np.exp(-1j*alpha*model.m))
    evaluation = []
    # All three independently selected training filters coincide here; retain
    # that null result in the manifest instead of labelling the same filter novel.
    chosen = manifest['filters']['target']
    for record in predictions:
        model = picket_fence(manifest['L'], record['G'])
        states = np.load(args.output/(record['name']+'.npz'))['states']
        exact = references[record['G'], record['kick']]
        rows = []
        for elapsed, a, b in zip(manifest['elapsed'], states, exact):
            ra, pa = filtered_density(model, a, chosen['particle'], chosen['hole'])
            rb, pb = filtered_density(model, b, chosen['particle'], chosen['hole'])
            oa, ob = tomography(ra, pa), tomography(rb, pb)
            rows.append({'elapsed': elapsed, **errors(model, a, b), 'filter_prediction': oa, 'filter_reference': ob,
                         'filter_probability_error': pa-pb, 'negativity_error': oa['negativity']-ob['negativity'],
                         'weighted_negativity_error': oa['weighted_negativity']-ob['weighted_negativity'],
                         'filtered_trace_distance': float(np.linalg.svd(ra-rb, compute_uv=False).sum()/2),
                         'unnormalized_filtered_trace_distance': float(np.linalg.svd(pa*ra-pb*rb, compute_uv=False).sum()/2)})
        evaluation.append({**record, 'rows': rows})
    (args.output/'evaluation.json').write_text(json.dumps(evaluation, indent=2))
    print(json.dumps({'predictions': len(predictions), 'references': len(references),
                      'max_norm_error': max(r['max_norm_error'] for r in predictions)}))
