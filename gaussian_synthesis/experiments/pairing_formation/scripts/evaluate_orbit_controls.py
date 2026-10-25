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
from pairing_formation.model import PairModel
from pairing_formation.multiorbit import MultiOrbitDynamics
from pairing_formation.conditional_dynamics import ConditionalDynamics
from pairing_formation.jastrow import JastrowDynamics
from pairing_formation.column_prediction import evolve_span
from pairing_formation.interband import filtered_density, tomography
from evaluate_conditional import errors


def predict(task):
    record, case, manifest, frozen, output = task
    begin = time.monotonic()
    path = frozen/record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    saved = np.load(path)
    model = PairModel(np.array(case['energies']), case['G'])
    name = record['name']+'__'+case['name']
    alpha = case['alpha']
    integration = {}
    if record['type'] == 'static':
        Q = saved['Q']*np.exp(-1j*alpha*model.m)[:, None]
        states = evolve_span(model, Q, saved['initial'], manifest['elapsed'])
        coordinates = np.empty((0, 0))
    else:
        initial = saved['coordinates'].copy()
        if record['type'] == 'multiorbit':
            dynamics = MultiOrbitDynamics(model, record['rank'])
            c, _ = dynamics.unpack(initial)
            c *= np.exp(-1j*alpha*np.arange(model.r+1))[:, None]
        elif record['type'] == 'jastrow':
            dynamics = JastrowDynamics(model, record['jastrow_family'], features=saved['features'])
            initial = dynamics.apply_diagonal_phase(initial, alpha*model.m)
        else:
            dynamics = ConditionalDynamics(model, record['conditioned'])
            c, _ = dynamics.unpack(initial)
            c *= np.exp(-1j*alpha*np.arange(model.r+1))
        states, coordinates, integration = dynamics.evolve(initial, manifest['elapsed'])
    np.savez_compressed(output/(name+'.npz'), states=states, coordinates=coordinates)
    report = {'name': name, 'family': record['name'], 'case': case['name'], 'G': case['G'],
              'spectrum': case['spectrum'], 'kick': case['kick'], 'alpha': alpha,
              'max_norm_error': float(np.max(abs(np.linalg.norm(states, axis=1)-1))),
              'integration': integration, 'elapsed_seconds': time.monotonic()-begin}
    (output/(name+'.json')).write_text(json.dumps(report, indent=2))
    return report


def observable_errors(model, a, b, chosen):
    ra, pa = filtered_density(model, a, chosen['particle'], chosen['hole'])
    rb, pb = filtered_density(model, b, chosen['particle'], chosen['hole'])
    oa, ob = tomography(ra, pa), tomography(rb, pb)
    return {**errors(model, a, b), 'filter_prediction': oa, 'filter_reference': ob,
            'filter_probability_error': pa-pb, 'negativity_error': oa['negativity']-ob['negativity'],
            'unnormalized_filtered_trace_distance': float(np.linalg.svd(pa*ra-pb*rb, compute_uv=False).sum()/2)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--grid', choices=['known', 'unseen'], default='known')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.frozen/'manifest.json').read_text())
    cases = manifest['case_groups'][args.grid]
    tasks = [(record, case, manifest, args.frozen, args.output) for record in manifest['records'] for case in cases]
    predictions = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(predict, task) for task in tasks]):
            record = future.result(); predictions.append(record)
            print(json.dumps({k: record[k] for k in ('name', 'elapsed_seconds', 'max_norm_error')}), flush=True)
    (args.output/'predictions_complete.json').write_text(json.dumps(predictions, indent=2))
    initial = np.load(args.frozen/'training.npz')['state']
    evaluation = []
    for case in cases:
        model = PairModel(np.array(case['energies']), case['G'])
        records = sorted([r for r in predictions if r['case'] == case['name']], key=lambda r: r['family'])
        states = [np.load(args.output/(r['name']+'.npz'))['states'] for r in records]
        # Matrix exponential of all initial states in one batch; this happens
        # only after every prediction in the grid has been saved.
        initial_columns = np.column_stack([initial*np.exp(-1j*case['alpha']*model.m)]+[x[0] for x in states])
        references = model.evolve(manifest['elapsed'], initial_columns)
        np.savez_compressed(args.output/('references__'+case['name']+'.npz'), states=references,
                            families=np.array(['true_training_state']+[r['family'] for r in records]))
        for j, (record, predicted) in enumerate(zip(records, states)):
            rows = []
            for elapsed, a, reference in zip(manifest['elapsed'], predicted, references):
                b, own = reference[:, 0], reference[:, j+1]
                total = observable_errors(model, a, b, manifest['filter'])
                start = observable_errors(model, own, b, manifest['filter'])
                dynamic = observable_errors(model, a, own, manifest['filter'])
                assert abs(total['negativity_error']-start['negativity_error']-dynamic['negativity_error']) < 1e-12
                rows.append({'elapsed': elapsed, **total, 'initial_state_only': start, 'dynamics_only': dynamic})
            evaluation.append({**record, 'rows': rows})
    (args.output/'evaluation.json').write_text(json.dumps(evaluation, indent=2))
    print(json.dumps({'predictions': len(predictions), 'physical_conditions': len(cases), 'grid': args.grid,
                      'exact_initial_state_controls': len(predictions)}))
